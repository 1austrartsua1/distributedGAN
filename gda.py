from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data


import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# distributed packages
import torch.utils.data.distributed


#from IPython.display import HTML

# locals
from utils_distributed import av_param,av_grad,av_loss




def main_worker(global_rank, local_rank, world_size, netG, netD,
                dataset, nz, criterion):

    # distributed stuff
    #global_rank is used for dist train_sampler and printing (only want global_rank=0 to print)
    #local_rank is the device id for the GPU associated with this process.
    #world_size is used in averaging gradients, dist train sampler


    # Set random seed for reproducibility
    set_seed = False
    if set_seed:
        manualSeed = 999
        #manualSeed = random.randint(1, 10000) # use if you want new results
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)



    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 128

    # Number of training epochs
    num_epochs = 5

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # simult: whether to use simultaneous or alternating GDA
    simult = False


    # create the sampler used for distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, global_rank)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers,
                                             pin_memory=True,sampler=train_sampler)

    torch.cuda.set_device(local_rank)




    # send the generator to GPU
    netG = netG.cuda()


    #dist.barrier()

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.


    #synchronize the model weights across devices
    av_param(netG,world_size)

    netD = netD.cuda()


    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.

    av_param(netD,world_size)

    # Initialize the loss on the GPU

    criterion = criterion.cuda()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1).cuda()



    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    if global_rank==0: print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].cuda(None,non_blocking=True)
            b_size = real_cpu.size(0)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()

            D_x = output.sum()

            D_x = av_loss(D_x, b_size)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1).cuda()
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            if simult:
                for p in netG.parameters():
                    p.requires_grad = False

                output = netD(fake).view(-1)
            else:
                output = netD(fake.detach()).view(-1)


            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch

            if simult:
                errD_fake.backward(retain_graph=True)
            else:
                errD_fake.backward()

            D_G_z1 = output.sum()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake #XX should this be an average?

            errD = av_loss(errD, 1.0)
            D_G_z1 = av_loss(D_G_z1, b_size)

            # average discriminator gradients across workers
            av_grad(netD,world_size)

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, in the alternating version (simult=False),
            # perform another forward pass of all-fake batch through D
            # in simultaneous GDA (simult=True), you don't need another forward pass

            if simult:
                for p in netG.parameters():
                    p.requires_grad = True
                for p in netD.parameters():
                    p.requires_grad = False
            else:
                # Update D
                optimizerD.step()
                #pass the fake data through the updated discriminator
                output = netD(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()

            D_G_z2 = output.sum()
            D_G_z2 = av_loss(D_G_z2, b_size)

            # average G's gradients across workers
            av_grad(netG,world_size)

            # Update G
            optimizerG.step()

            if simult:
                # Update D
                optimizerD.step()
                for p in netD.parameters():
                    p.requires_grad = True

            # Output training stats
            if (global_rank == 0) and (i % 50 == 0):
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD, errG, D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG)
                D_losses.append(errD)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (global_rank==0) and ((iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1))):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
