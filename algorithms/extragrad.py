import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import pickle

import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# distributed packages
import torch.utils.data.distributed

# locals
from utils_distributed import av_param,av_grad,av_loss
from utils import compute_gan_loss,sampler,clip,Minibatch,ProgressMeter
from optim.OptimExtragrad import ExtraAdam


def main_worker(global_rank, local_rank, world_size, netG, netD,
                dataset, nz, loss_type, sampler_option, clip_amount, results):

    # distributed stuff
    #global_rank is used for dist train_sampler and printing (only want global_rank=0 to print)
    #local_rank is the device id for the GPU associated with this process.
    #world_size is used in averaging gradients, dist train sampler
    now=datetime.now()
    dt_string = now.strftime("%d_%m_%Y::%H:%M:%S")

    tstart = time.time()

    # Set random seed for reproducibility
    set_seed = False
    if set_seed:
        manualSeed = 999
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        #random.seed(manualSeed)
        torch.manual_seed(manualSeed)



    # Number of workers for dataloader
    workers = 1
    # Batch size during training
    batch_size = 64
    results['batch_size'] = batch_size
    # Number of training epochs
    num_epochs = 600
    num_iterations = float('inf')
    # Learning rate for optimizers
    lr_dis = 2e-4
    lr_gen = 2e-5

    # Beta1, beta2 hyperparam for Adam optimizers
    beta1 = 0.5
    beta2 = 0.9

    # new means get a new minibatch for the optimizer.step,
    # stale means use the same minibatch for extrapolate and step
    stale = True

    IS_eval_freq = 12 # for FID/IS, how many epochs between calculation of IS
    # note IS calculation takes about 60 sec, so don't want to necessarily do it
    # every epoch, since epoch for cifar may be like 30sec. Better to do it every 10 epochs's or so
    n_samples = 50000 # for FID/IS
    getInceptionScore = True 

    param_setting_str = f"batch_size:{batch_size},lr_dis:{lr_dis},lr_gen:{lr_gen},beta1:{beta1},beta2:{beta2},stale:{stale},workers:{workers}"
    if global_rank==0: print(param_setting_str)

    # create the sampler used for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, global_rank)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers,
                                             pin_memory=True,sampler=train_sampler)

    torch.cuda.set_device(local_rank)

    # send the generator to GPU
    netG = netG.cuda()

    #synchronize the model weights across devices
    av_param(netG,world_size)

    netD = netD.cuda()

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.

    # average parameters across workers so they all start with the same model
    av_param(netD,world_size)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = sampler(64,nz,sampler_option).cuda()

    # Setup Adam optimizers for both G and D
    optimizerD = ExtraAdam(netD.parameters(), lr=lr_dis, betas=(beta1, beta2))
    optimizerG = ExtraAdam(netG.parameters(), lr=lr_gen, betas=(beta1, beta2))

    # Training Loop
    forward_steps = 0
    newEpoch = False
    epoch = 0
    minibatch = Minibatch(dataloader)
    errD,errG,D_on_real_data,D_on_fake_data = 4*[float("NaN")]

    if global_rank==0:
        print("Starting Training Loop...")
        progressMeter = ProgressMeter(n_samples,nz,netG,num_epochs,
                                      dataloader,results,IS_eval_freq,sampler_option,
                                      clip_amount,param_setting_str,dt_string,
                                      getInceptionScore)

        progressMeter.record(forward_steps,epoch,minibatch.epoch_progress,errD,errG,D_on_real_data,D_on_fake_data,float("NaN"))


    tepoch = time.time()
    while (forward_steps < 2*num_iterations) and (epoch < num_epochs):

        if (forward_steps%2==0) or (stale==False):
            data,newEpoch = minibatch.get()
            epoch += int(newEpoch)
        ############################
        # (1) Calculate D network gradients
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].cuda(None,non_blocking=True)
        b_size = real_cpu.size(0)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = compute_gan_loss(output,loss_type,b_size,"real","dis")
        # Calculate gradients for D in backward pass
        errD_real.backward()

        D_on_real_data = output.sum()

        D_on_real_data = av_loss(D_on_real_data, b_size)

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = sampler(b_size,nz,sampler_option).cuda()

        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D

        for p in netG.parameters():
            p.requires_grad = False

        output = netD(fake).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = compute_gan_loss(output,loss_type,b_size,"fake","dis")

        # Calculate the gradients for this batch
        errD_fake.backward(retain_graph=True)


        D_on_fake_data = output.sum()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake

        errD = av_loss(errD, 1.0)
        D_on_fake_data = av_loss(D_on_fake_data, b_size)

        # average discriminator gradients across workers
        av_grad(netD,world_size)

        ############################
        # (2) Calculate G network gradients
        ###########################
        netG.zero_grad()

        for p in netG.parameters():
            p.requires_grad = True
        for p in netD.parameters():
            p.requires_grad = False


        # Calculate G's loss based on this output
        # use real labels because generator wants to fool the discriminator
        errG = compute_gan_loss(output,loss_type,b_size,"real","gen")
        # Calculate gradients for G
        errG.backward()

        errG = av_loss(errG, 1.0)

        # average G's gradients across workers
        av_grad(netG,world_size)

        if forward_steps%2 == 0:
            #extrapolation
            # Update G
            optimizerG.extrapolate()
            # Update D
            optimizerD.extrapolate()
        else:
            #step
            # Update G
            optimizerG.step()
            # Update D
            optimizerD.step()

        # clip the discriminator tensors
        # for extragradient, clip after both extrapolate() and the step()
        clip(netD,clip_amount)

        forward_steps += 1
        for p in netD.parameters():
            p.requires_grad = True

        if (global_rank==0) and newEpoch:
            newEpoch = False
            tepoch = time.time()-tepoch
            if (epoch+1) % IS_eval_freq == 0:
                progressMeter.record(forward_steps,epoch,minibatch.epoch_progress,errD,errG,D_on_real_data,D_on_fake_data,float("NaN"))
                ttot = time.time() - tstart
                progressMeter.save(ttot,tepoch)
            print(f"epoch {epoch} time = {tepoch}")

            tepoch = time.time()
