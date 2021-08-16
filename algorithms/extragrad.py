import torch.optim as optim
import numpy as np

# locals

from utils_distributed import av_grad
from utils import clip
from optim.OptimExtragrad import ExtraAdam, ExtraSGD
from algorithms.method import TwoForwardStep

class Extragrad(TwoForwardStep):
    def __init__(self):
        super(Extragrad,self).__init__()


    def setup_optimizer(self,params,netD,netG):

        if params.adam_updates == True:
            optimizerD = ExtraAdam(netD.parameters(), lr=params.lr_dis,
                                   betas=(params.beta1, params.beta2))
            optimizerG = ExtraAdam(netG.parameters(), lr=params.lr_gen,
                                   betas=(params.beta1, params.beta2))
        else:
            optimizerD = ExtraSGD(netD.parameters(), lr=params.lr_dis)
            optimizerG = ExtraSGD(netG.parameters(), lr=params.lr_gen)

        return optimizerD,optimizerG

    def optimizer_step(self,optimizerG,optimizerD,netD,netG,clip_amount):

        if self.forward_steps%2 == 0:
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
        for p in netD.parameters():
            p.requires_grad = True

    def communicate(self,netD, netG, world_size,ch_D,ch_G,args):
        # average discriminator gradients across workers
        if args.chunk_reduce:
            ch_D.av_grad(netD, world_size)
        else:
            av_grad(netD, world_size)

        # average G's gradients across workers
        if args.chunk_reduce:
            ch_G.av_grad(netG,world_size)
        else:
            av_grad(netG,world_size)


