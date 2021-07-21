import torch.optim as optim
import numpy as np

# locals

from utils_distributed import av_grad,av_loss
from utils import clip,sampler,compute_gan_loss
from algorithms.method import Method,OneForwardStep


class GDA(OneForwardStep):
    def __init__(self,params):
        self.simult = params.simult
        super(GDA,self).__init__()

    def setup_optimizer(self,params,netD,netG):
        optimizerD = optim.Adam(netD.parameters(), lr=params.lr_dis, betas=(params.beta1, params.beta2))
        optimizerG = optim.Adam(netG.parameters(), lr=params.lr_gen, betas=(params.beta1, params.beta2))
        return optimizerD,optimizerG

    def dis_fake_batch(self,b_size,nz,sampler_option,netG,netD,loss_type):
        noise = sampler(b_size, nz, sampler_option).cuda()

        # Generate fake image batch with G
        fake = netG(noise)
        # Classify all fake batch with D
        if self.simult:
            for p in netG.parameters():
                p.requires_grad = False

            output = netD(fake).view(-1)
        else:
            output = netD(fake.detach()).view(-1)

        # Calculate D's loss on the all-fake batch
        errD_fake = compute_gan_loss(output, loss_type, b_size, "fake", "dis")

        # Calculate the gradients for this batch
        if self.simult:
            errD_fake.backward(retain_graph=True)
        else:
            errD_fake.backward()

        D_on_fake_data = output.sum()
        D_on_fake_data = av_loss(D_on_fake_data, b_size)

        return D_on_fake_data, errD_fake, output,fake

    def gen_grads(self, netG, netD, loss_type, b_size, output,optimizerD,optimizerG,world_size,clip_amount,fake):
        # in GDA, we do communication at every iteration
        # bundled in here is the netD grad comms for convenience
        # with the alt version (sim.simult=False) we need to update
        # netD before getting gen grads, so we need to communicate first
        av_grad(netD, world_size)

        netG.zero_grad()
        # Since we just updated D, in the alternating version (simult=False),
        # perform another forward pass of all-fake batch through D
        # in simultaneous GDA (simult=True), you don't need another forward pass

        if self.simult:
            for p in netG.parameters():
                p.requires_grad = True
            for p in netD.parameters():
                p.requires_grad = False
        else:
            # Update D
            optimizerD.step()
            # clip the discriminator params
            clip(netD, clip_amount)

            # pass the fake data through the updated discriminator
            output = netD(fake).view(-1)

        # Calculate G's loss based on this output
        # use real labels because generator wants to fool the discriminator
        errG = compute_gan_loss(output, loss_type, b_size, "real", "gen")
        # Calculate gradients for G
        errG.backward()

        errG = av_loss(errG, 1.0)

        # average G's gradients across workers
        av_grad(netG, world_size)

        optimizerG.step()

        if self.simult:
            # Update D
            optimizerD.step()
            # clip the discriminator tensors
            clip(netD, clip_amount)

            for p in netD.parameters():
                p.requires_grad = True

        return errG

