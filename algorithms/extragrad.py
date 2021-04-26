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
from utils_distributed import av_param,av_loss
from utils import compute_gan_loss,sampler,clip,Minibatch,ProgressMeter
from optim.OptimExtragrad import ExtraAdam, ExtraSGD


def main_worker(global_rank, local_rank, world_size, netG, netD,
                dataset, nz, loss_type, sampler_option, clip_amount, results
                , args, params):

    now=datetime.now()
    dt_string = now.strftime("%d_%m_%Y::%H:%M:%S")

    tstart = time.time()

    if params.num_iterations == "None":
        num_iterations = float('inf')
    else:
        num_iterations = params.num_iterations

    results['batch_size'] = params.batch_size
    param_setting_str = f"batch_size:{params.batch_size},lr_dis:{params.lr_dis},"
    param_setting_str += f"lr_gen:{params.lr_gen},beta1:{params.beta1},beta2:{params.beta2},stale:{params.stale},workers:{params.workers},"
    param_setting_str += f"adam_updates:{params.adam_updates}\n"
    param_setting_str += f"chunk_reduce:{args.chunk_reduce}"

    if params.set_seed:
        manualSeed = 999
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        #random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    if global_rank==0:
        print("\n\n")
        print(param_setting_str)
        print("\n\n")

    # create the sampler used for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, global_rank)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,
                                             shuffle=False, num_workers=params.workers,
                                             pin_memory=True,sampler=train_sampler)

    torch.cuda.set_device(local_rank)

    # send the generator to GPU
    netG = netG.cuda()

    #synchronize the model weights across devices
    av_param(netG,world_size)

    netD = netD.cuda()


    if args.chunk_reduce:
        from utils_distributed import GradAverager as GA
        ga_G = GA(netG)
        ga_D = GA(netD)
    else:
        from utils_distributed import av_grad

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.

    # average parameters across workers so they all start with the same model
    av_param(netD,world_size)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = sampler(64,nz,sampler_option).cuda()

    # Setup optimizers for both G and D

    if params.adam_updates == True:
        optimizerD = ExtraAdam(netD.parameters(), lr=params.lr_dis, betas=(params.beta1, params.beta2))
        optimizerG = ExtraAdam(netG.parameters(), lr=params.lr_gen, betas=(params.beta1, params.beta2))
    else:
        optimizerD = ExtraSGD(netD.parameters(), lr=params.lr_dis)
        optimizerG = ExtraSGD(netG.parameters(), lr=params.lr_gen)


    # Training Loop
    forward_steps = 0
    newEpoch = False
    epoch = 0
    minibatch = Minibatch(dataloader)
    errD,errG,D_on_real_data,D_on_fake_data = 4*[float("NaN")]

    if global_rank==0:
        print("Starting Training Loop...")
        progressMeter = ProgressMeter(params.n_samples,nz,netG,params.num_epochs,
                                      dataloader,results,params.IS_eval_freq,sampler_option,
                                      clip_amount,param_setting_str,dt_string,
                                      params.getISscore,args.results,params.getFIDscore,params.path2FIDstats,
                                      args.moreFilters,args.paramTuning)

        if not params.getFirstScore:
            progressMeter.getISscore = False
            progressMeter.getFIDscore = False

        progressMeter.record(forward_steps,epoch,errD,errG,netG)
        progressMeter.getISscore = params.getISscore
        progressMeter.getFIDscore = params.getFIDscore

    tepoch = time.time()
    while (forward_steps < 2*num_iterations) and (epoch < params.num_epochs):

        if (forward_steps%2==0) or (params.stale==False):
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
        if args.chunk_reduce:
            ga_D.av_grad(netD,world_size)
        else:
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
        if args.chunk_reduce:
            ga_G.av_grad(netG,world_size)
        else:
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
            if (epoch+1) % params.IS_eval_freq == 0:
                progressMeter.record(forward_steps,epoch,errD,errG,netG)
                ttot = time.time() - tstart
                progressMeter.save(ttot,tepoch)
            print(f"epoch {epoch} time = {tepoch}")
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch,params.num_epochs,errD,errG,D_on_real_data,D_on_fake_data,float("NaN")))

            tepoch = time.time()


    if global_rank == 0:
        tepoch = time.time()-tepoch
        ttot = time.time() - tstart
        progressMeter.record(forward_steps,epoch,errD,errG,netG,final=True)
        progressMeter.save(ttot,tepoch,final=True,paramTuneVal=args.tuneVal)
