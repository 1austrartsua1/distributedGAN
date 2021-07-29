import torch.optim as optim
import numpy as np
import torch
import os
import sys
import torch.distributed as dist


# locals
from utils_distributed import Chunker as CH
from utils_distributed import av_param
from algorithms.extragrad import Extragrad
from utils import Minibatch,ProgressMeter, clip,sampler

def print_some(thing):
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open('print_output.txt', 'a') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(thing)
        sys.stdout = original_stdout  # Reset the standard output to its original value


class AsyncEG(Extragrad):
    def __init__(self):
        super(Extragrad,self).__init__()
        self.CONTINUE = torch.tensor([1])
        self.FINISH = torch.tensor([0])
        self.continue_flag = torch.tensor([-1])

    def main(self, global_rank, local_rank, world_size, netG, netD,
         dataset, nz, loss_type, sampler_option, clip_amount, results
         , args, params):
        if global_rank==0:
            if os.path.exists("print_output.txt"):
                os.remove("print_output.txt")
            print_some("entered main()")
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.dataset = dataset
        self.nz = nz
        self.loss_type = loss_type
        self.sampler_option = sampler_option
        self.clip_amount = clip_amount
        self.results = results
        self.args = args
        self.params = params

        dt_string, tstart, self.num_iterations = self.main_preamble(params, results, global_rank)

        torch.cuda.set_device(local_rank)

        # send the generator weights to GPU
        self.netG = netG.cuda()
        self.netD = netD.cuda()

        # synchronize the model weights across devices
        av_param(self.netG, world_size)
        av_param(self.netD, world_size)

        # set up the chunker which creates a buffer we can use to store either the network params
        # or gradients to send a single message rather than having to send each param tensor
        # individually. This makes code simpler, especially for async, as you only need
        # to keep one request object for the buffer rather than a list for all param tensors
        # Also may improve speed as there is less overhead for the communication.

        isMaster = (global_rank==0)
        self.ch_G = CH(netG,world_size,isMaster)
        self.ch_D = CH(netD,world_size,isMaster)

        # Setup optimizers for both G and D
        self.setup_optimizer(params, self.netD, self.netG)

        if global_rank == 0:
            dl = torch.utils.data.DataLoader(self.dataset)
            data_loader_iter = iter(dl)
            data = next(data_loader_iter)
            data = data[0].cuda()
            outD = torch.sum(0*self.netD(data))
            outD.backward()

            noise = sampler(10, self.nz, self.sampler_option).cuda()

            # Generate fake image batch with G
            fake = netG(noise)
            outG = torch.sum(0*fake)
            outG.backward()

            self.master()
        else:
            self.worker()


    def worker(self):
        print_some(f"worker {self.global_rank} in worker()")
        # create the sampler used for distributed training
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, self.world_size - 1, self.global_rank - 1)
        # Create the dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.params.batch_size,
                                                 shuffle=False, num_workers=self.params.workers,
                                                 pin_memory=True, sampler=train_sampler)

        self.worker_loop()

    def worker_get_grads(self,data):
        ############################
        # (1) Calculate D network gradients
        ###########################
        ## (1a) Train with all-real batch
        ###########################

        D_on_real_data, errD_real, b_size = self.dis_real_batch(self.netD, data, self.loss_type,commLoss=False)

        ############################
        ## (1b) Train with all-fake batch
        ############################
        D_on_fake_data, errD_fake, output, fake = self.dis_fake_batch(b_size, self.nz, self.sampler_option, self.netG,
                                                                      self.netD,
                                                                      self.loss_type,commLoss=False)

        errD = errD_real + errD_fake


        ############################
        # (2) Calculate G network gradients
        ###########################
        errG = self.gen_grads(self.netG, self.netD, self.loss_type, b_size, output, None, None, None,
                              None, None,commLoss=False)


        # need to set requires_grad back to True for discriminator after it was set to False in the gen_grads function
        for p in self.netD.parameters():
            p.requires_grad = True


    def worker_loop(self):
        print_some(f"worker {self.global_rank} in worker_loop()")
        ngrads = 0
        minibatch = Minibatch(self.dataloader)
        self.epoch = 0
        loopNumber = 0
        while True:
            loopNumber += 1
            print_some(f"LOOP NUMBER: {loopNumber}")
            #get new data
            print_some(f"worker {self.global_rank} getting new data")
            data, newEpoch = self.get_new_data(minibatch)

            # calculate disc and generator gradients
            print_some(f"worker {self.global_rank} calculating grads")
            self.worker_get_grads(data)

            # perform an update using these gradients
            # Update G
            print_some(f"worker {self.global_rank} optimizer step")
            self.optimizerG.step()
            # Update D
            self.optimizerD.step()

            # clip params here
            clip(self.netD, self.clip_amount)

            # get new data
            if self.params.stale==False:
                data, newEpoch = self.get_new_data(minibatch)

            # once again, calculate disc and generator gradients
            print_some(f"worker {self.global_rank} calculating grads second time...")
            self.worker_get_grads(data)

            # communicate them to the master
            print_some(f"worker {self.global_rank} sending isend()s")
            self.ch_G.grad_isend()
            self.ch_D.grad_isend()
            print_some(f"worker {self.global_rank} waiting...")
            self.ch_G.grad_wait()
            self.ch_D.grad_wait()

            ngrads += 1
            dist.recv(self.continue_flag, 0)
            if self.continue_flag[0] == 0:
                break
            # get updates params from master
            self.ch_G.param_recv()
            self.ch_D.param_recv()
            print_some(f"worker {self.global_rank + 1} got new params")

        print_some(f"worker {self.global_rank + 1} exiting after computing {ngrads} gradients")


    def master(self):
        print_some("master in master()")
        self.master_setup()
        print_some("master finishes with setup")
        self.master_loop()
        self.master_cleanup()

    def master_setup(self):
        print_some("master in master_setup()")
        # initial setup of irecv from every worker
        for i in range(1, self.world_size):
            print_some(i)
            self.ch_G.grad_irecv(i)
            self.ch_D.grad_irecv(i)
        print_some("finished with setup...!")

    def master_update(self,workeri):
        # copy workeri gradients accross to netG and netD
        # perform a step on optimizerG and optimizerD
        self.ch_G.copy_grad(workeri)
        self.ch_D.copy_grad(workeri)

        # Update G
        self.optimizerG.step()
        # Update D
        self.optimizerD.step()

        # clip params here
        clip(self.netD, self.clip_amount)

    def master_loop(self):
        # loop to check for an irecv completed
        print_some("master in master_loop()")
        updates = 0
        i = 0
        while True:
            if self.ch_G.grad_ready(i) and self.ch_D.grad_ready(i):
                self.master_update(i)
                updates += 1
                print_some(f"num updates {updates}")
                if updates >= self.num_iterations:
                    # send message to others to exit...
                    break
                print_some(f"master sending continue")
                dist.send(self.CONTINUE, i + 1)
                print_some(f"master sending params")
                self.ch_G.param_send(i + 1)
                self.ch_D.param_send(i + 1)

                print_some(f"master setting up irecv")
                self.ch_G.grad_irecv(i + 1)
                self.ch_D.grad_irecv(i + 1)
            i = (i + 1) % (self.world_size - 1)

    def master_cleanup(self):
        # clean-up (tell other processes to shut down)
        finishes_sent = 0
        not_sent = [1 for i in range(self.world_size - 1)]
        i = 0
        while True:
            if self.ch_G.grad_ready(i) and self.ch_D.grad_ready(i) and not_sent[i]:
                not_sent[i] = 0
                dist.send(self.FINISH, i + 1)
                finishes_sent += 1
                if finishes_sent >= self.world_size - 1:
                    break
            i = (i + 1) % (self.world_size - 1)

    def setup_optimizer(self,params,netD,netG):
        self.optimizerD = optim.Adam(netD.parameters(), lr=params.lr_dis, betas=(params.beta1, params.beta2))
        self.optimizerG = optim.Adam(netG.parameters(), lr=params.lr_gen, betas=(params.beta1, params.beta2))


    def get_new_data(self, minibatch):
        data, newEpoch = minibatch.get()
        self.epoch += int(newEpoch)
        return data, newEpoch





