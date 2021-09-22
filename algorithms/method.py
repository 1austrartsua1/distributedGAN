import argparse
import torch
#import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# distributed packages
import torch.utils.data.distributed

# locals
from utils_distributed import av_param,av_loss
from utils import compute_gan_loss,sampler,clip,Minibatch,ProgressMeter

#from optim.OptimExtragrad import ExtraAdam, ExtraSGD


'''
        METHOD
          /\
    1STEP   2STEP
    /  |     |   |
   GDA OGM   EG  PS



'''

class Method:
    # Generic Method
    # Base class, can be specialized to a specific Method
    # such as GDA, Extragrad, projective splitting etc.
    def __init__(self):
        self.forward_steps = 0

    def print_net(self,netG,netD):
        #print(f"netG params:")
        norms = 0.0
        for p in netG.parameters():
            norms += torch.norm(p)**2

        #print(f"netD params:")
        for p in netD.parameters():
            norms += torch.norm(p)**2

        print(f"net norm {norms}")

    def print_grad(self,netG,netD):

        norms = 0.0
        for p in netG.parameters():
            norms += torch.norm(p.grad)**2


        for p in netD.parameters():
            norms += torch.norm(p.grad)**2

        print(f"grad norm {norms}")


    def setup_optimizer(self,params,netD,netG):
        # must overwrite
        pass

    def update_iteration_counter(self,iteration):
        # must overwrite
        pass

    def get_new_data(self,params,minibatch):
        # must overwrite
        pass

    def optimizer_step(self,optimizerG,optimizerD,netD,netG,clip_amount):
        # must overwrite
        pass

    def communicate(self,netD,netG,world_size,ch_D,ch_G,args):
        # must overwrite
        pass

    def dis_real_batch(self,netD,data,loss_type,commLoss=True):
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].cuda(None, non_blocking=True)
        b_size = real_cpu.size(0)

        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = compute_gan_loss(output, loss_type, b_size, "real", "dis")
        # Calculate gradients for D in backward pass
        errD_real.backward()

        D_on_real_data = output.sum()

        if commLoss:
            D_on_real_data = av_loss(D_on_real_data, b_size)

        return D_on_real_data,errD_real,b_size,real_cpu

    def dis_fake_batch(self,b_size,nz,sampler_option,netG,netD,loss_type,commLoss=True):
        # simultaneous version...
        # alternating version needs to be overloaded

        # Generate batch of latent vectors
        noise = sampler(b_size, nz, sampler_option).cuda()

        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D

        for p in netG.parameters():
            p.requires_grad = False

        output = netD(fake).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = compute_gan_loss(output, loss_type, b_size, "fake", "dis")

        # Calculate the gradients for this batch
        errD_fake.backward(retain_graph=True)

        D_on_fake_data = output.sum()
        if commLoss:
            D_on_fake_data = av_loss(D_on_fake_data, b_size)

        return D_on_fake_data, errD_fake,output,fake

    def gen_grads(self,netG,netD,loss_type,b_size,output,optimizerD,optimizerG,world_size,clip_amount,fake,commLoss=True):
        netG.zero_grad()

        for p in netG.parameters():
            p.requires_grad = True
        for p in netD.parameters():
            p.requires_grad = False

        # Calculate G's loss based on this output
        # use real labels because generator wants to fool the discriminator
        errG = compute_gan_loss(output, loss_type, b_size, "real", "gen")
        # Calculate gradients for G
        errG.backward()
        if commLoss:
            errG = av_loss(errG, 1.0)

        return errG

    def main_preamble(self,params,results,global_rank):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y::%H:%M:%S")

        tstart = time.time()

        if params.num_iterations == "None":
            num_iterations = float('inf')
        else:
            num_iterations = params.num_iterations

        results['batch_size'] = params.batch_size



        if global_rank == 0:
            print("\n\n")
            print(f"params\n\n{params}")
            print("\n\n")

        return dt_string,tstart,num_iterations

    def main(self,global_rank, local_rank, world_size, netG, netD,
                dataset, nz, loss_type, sampler_option, clip_amount, results
                ,args, params):

        dt_string, tstart, num_iterations = self.main_preamble(params,results,global_rank)

        # create the sampler used for distributed training
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, global_rank)

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size,
                                                 shuffle=False, num_workers=params.workers,
                                                 pin_memory=True,sampler=train_sampler)

        torch.cuda.set_device(local_rank)

        # send the generator weights to GPU
        netG = netG.cuda()

        #synchronize the model weights across devices
        av_param(netG,world_size)

        netD = netD.cuda()


        if args.chunk_reduce:
            from utils_distributed import Chunker as CH
            ch_G = CH(netG)
            ch_D = CH(netD)
        else:
            ch_G = None
            ch_D = None


        # average parameters across workers so they all start with the same model
        av_param(netD,world_size)


        # Setup optimizers for both G and D
        optimizerD,optimizerG = self.setup_optimizer(params,netD,netG)

        # Training Loop
        self.forward_steps = 0
        self.epoch = 0
        minibatch = Minibatch(dataloader)
        errD,errG,D_on_real_data,D_on_fake_data = 4*[float("NaN")]

        if global_rank==0:
            print("Starting Training Loop...")
            progressMeter = ProgressMeter(params.n_samples,nz,netG,args.num_epochs,
                                          dataloader,results,params.IS_eval_freq,sampler_option,
                                          clip_amount,params,dt_string,
                                          params.getISscore,args.results,params.getFIDscore,params.path2FIDstats,
                                          args.moreFilters,args.paramTuning)

            if not params.getFirstScore:
                progressMeter.getISscore = False
                progressMeter.getFIDscore = False

            progressMeter.record(self.forward_steps,self.epoch,errD,errG,netG)
            progressMeter.getISscore = params.getISscore
            progressMeter.getFIDscore = params.getFIDscore

        tepoch = time.time()
        iteration = 0




        while (iteration < num_iterations) and (self.epoch < args.num_epochs):
            dataNew,newEpoch = self.get_new_data(params,minibatch)
            if dataNew is not None:
                data = dataNew

            ############################
            # (1) Calculate D network gradients
            ###########################
            ## (1a) Train with all-real batch
            ###########################
            D_on_real_data, errD_real,b_size,real = self.dis_real_batch(netD, data, loss_type)

            ############################
            ## (1b) Train with all-fake batch
            ############################
            D_on_fake_data, errD_fake,output,fake = self.dis_fake_batch(b_size, nz, sampler_option, netG, netD, loss_type)


            errD = errD_real + errD_fake

            errD = av_loss(errD, 1.0)

            ############################
            # (2) Calculate G network gradients
            ###########################
            errG = self.gen_grads(netG, netD, loss_type, b_size,output,optimizerD,optimizerG,world_size,clip_amount,fake)


            ############################
            # (2) Communicate (if necessary)
            ###########################
            self.communicate(netD,netG,world_size,ch_D,ch_G,args)


            self.optimizer_step(optimizerG, optimizerD, netD, netG, clip_amount)

            #if (global_rank==0) and (iteration % 2 == 1):
            #    # update step:
            #    print("net after update")
            #    self.print_net(netG,netD)


            iteration = self.update_iteration_counter(iteration)


            if (global_rank==0) and newEpoch:
                tepoch = time.time()-tepoch
                if (self.epoch+1) % params.IS_eval_freq == 0:
                    progressMeter.record(self.forward_steps,self.epoch,errD,errG,netG)
                    ttot = time.time() - tstart
                    progressMeter.save(ttot,tepoch)
                print(f"epoch {self.epoch} time = {tepoch}")
                print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (self.epoch,args.num_epochs,errD,errG,D_on_real_data,D_on_fake_data,float("NaN")))

                tepoch = time.time()


        if global_rank == 0:
            tepoch = time.time()-tepoch
            ttot = time.time() - tstart
            progressMeter.record(self.forward_steps,self.epoch,errD,errG,netG,final=True)
            progressMeter.save(ttot,tepoch,final=True,paramTuneVal=args.tuneVal)


class OneForwardStep(Method):
    def __init__(self):
        super(OneForwardStep,self).__init__()

    def get_new_data(self,params,minibatch):
        data,newEpoch = minibatch.get()
        self.epoch += int(newEpoch)
        return data,newEpoch

    def update_iteration_counter(self,iteration):
        self.forward_steps += 1
        iteration += 1
        return iteration


class TwoForwardStep(Method):
    def __init__(self):
        super(TwoForwardStep,self).__init__()


    def get_new_data(self,params,minibatch):
        if (self.forward_steps%2==0) or (params.stale==False):
            data,newEpoch = minibatch.get()
            self.epoch += int(newEpoch)
        else:
            data = None
            newEpoch = False


        return data,newEpoch

    def update_iteration_counter(self,iteration):
        self.forward_steps += 1
        if self.forward_steps%2 == 0:
            iteration += 1
        return iteration




