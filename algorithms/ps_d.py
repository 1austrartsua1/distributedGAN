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
#from optim.OptimExtragrad import ExtraAdam
from optim.OptimPSd import PS_Adam, PS_SGD


def main_worker(global_rank, local_rank, world_size, netG, netD,
                dataset, nz, loss_type, sampler_option, clip_amount,
                results, args, params):
                
    now=datetime.now()
    dt_string = now.strftime("%d_%m_%Y::%H:%M:%S")

    tstart = time.time()



    if params.num_iterations == "None":
        num_iterations = float('inf')
    else:
        num_iterations = params.num_iterations
    # Learning rate for optimizers
    if params.lr_dis_extrap=="None":
        lr_dis_extrap = None
    else:
        lr_dis_extrap = params.lr_dis_extrap
    if params.lr_gen_extrap=="None":
        lr_gen_extrap = None
    else:
        lr_gen_extrap = params.lr_gen_extrap


    lr_dis_dual = params.gamma*params.lr_dis_step
    lr_gen_dual = params.gamma*params.lr_gen_step



    results['batch_size'] = params.batch_size

    param_setting_str = f"batch_size:{params.batch_size},lr_dis_step:{params.lr_dis_step},"
    param_setting_str+=f"lr_dis_extrap:{lr_dis_extrap},\n"
    param_setting_str += f"lr_gen_step:{params.lr_gen_step},"
    param_setting_str+=f"lr_gen_extrap:{lr_gen_extrap},"
    param_setting_str += f"stale:{params.stale},workers:{params.workers},av_reduce:{params.av_reduce}\n"
    param_setting_str += f"clip_on_extrapolate:{params.clip_on_extrapolate},adam_updates:{params.adam_updates}\n"
    param_setting_str += f"beta1:{params.beta1},beta2:{params.beta2},AdamForDuals:{params.AdamForDuals},gamma:{params.gamma}\n"
    param_setting_str+=f"chunk_reduce:{args.chunk_reduce},"
    param_setting_str+=f"moreFilters:{args.moreFilters},"

    if global_rank==0: print(param_setting_str)

    if params.set_seed:
        manualSeed = 999
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        #random.seed(manualSeed)
        torch.manual_seed(manualSeed)

    if params.av_reduce:
        divide_by = world_size
    else:
        divide_by = 1.0

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

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.

    # average parameters across workers so they all start with the same model
    av_param(netD,world_size)

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = sampler(64,nz,sampler_option).cuda()

    # Setup Adam optimizers for both G and D
    if params.adam_updates:
        optimizerD = PS_Adam(netD.parameters(), lr_step=params.lr_dis_step,
                             lr_extrap=lr_dis_extrap, betas=(params.beta1, params.beta2),
                             AdamForDuals=params.AdamForDuals,lr_dual=lr_dis_dual)
        optimizerG = PS_Adam(netG.parameters(), lr_step=params.lr_gen_step,
                             lr_extrap=lr_gen_extrap, betas=(params.beta1, params.beta2),
                             AdamForDuals=params.AdamForDuals,lr_dual=lr_gen_dual)
    else:
        optimizerD = PS_SGD(netD.parameters(), lr_step=params.lr_dis_step,lr_extrap=lr_dis_extrap)
        optimizerG = PS_SGD(netG.parameters(), lr_step=params.lr_gen_step,lr_extrap=lr_gen_extrap)

    # Training Loop
    forward_steps = 0
    newEpoch = False
    epoch = 0
    minibatch = Minibatch(dataloader)
    errD,errG,D_on_real_data,D_on_fake_data = 4*[float("NaN")]

    if global_rank==0:
        print("Starting Training Loop...")
        progressMeter = ProgressMeter(params.n_samples,nz,netG,args.num_epochs,
                                      dataloader,results,params.IS_eval_freq,sampler_option,
                                      clip_amount,param_setting_str,dt_string,
                                      params.getISscore, args.results,params.getFIDscore,params.path2FIDstats
                                      ,args.moreFilters,args.paramTuning)


        if (not params.getFirstScore) or args.paramTuning:
            progressMeter.getISscore = False
            progressMeter.getFIDscore = False

        progressMeter.record(forward_steps,epoch,errD,errG,netG)
        progressMeter.getISscore = params.getISscore or args.paramTuning
        progressMeter.getFIDscore = params.getFIDscore


    tepoch = time.time()
    while (forward_steps < 2*num_iterations) and (epoch < args.num_epochs):

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



        if forward_steps%2 == 0:

            #extrapolation
            # Update G
            optimizerG.extrapolate()
            # Update D
            optimizerD.extrapolate()

            if params.clip_on_extrapolate:
                clip(netD,clip_amount)
        else:
            # for sync projective splitting, only perform the all-reduce
            # before a step() update, not before extrapolate().

            # average discriminator gradients across workers
            if args.chunk_reduce:
                optimizerG.reduce(divide_by)
                optimizerD.reduce(divide_by)
            else:
                av_grad(netG,divide_by)
                # average G's gradients across workers
                av_grad(netD,divide_by)

            #step
            # Update G
            optimizerG.primal_step()
            # Update D
            optimizerD.primal_step()
            # clip the discriminator tensors
            # for projective splitting, clip only after the step()
            clip(netD,clip_amount)

            optimizerG.dual_step(world_size,args.chunk_reduce)
            optimizerD.dual_step(world_size,args.chunk_reduce)



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
            % (epoch,args.num_epochs,errD,errG,D_on_real_data,D_on_fake_data,float("NaN")))
            tepoch = time.time()


    if global_rank==0:
        tepoch = time.time()-tepoch
        ttot = time.time() - tstart
        progressMeter.record(forward_steps,epoch,errD,errG,netG,final=True)
        progressMeter.save(ttot,tepoch,final=True,paramTuneVal=args.tuneVal)
