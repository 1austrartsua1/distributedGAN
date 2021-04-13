import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import time
import pickle


#locals
import inception_score as iscore
import randomDataLoaderv2 as rd
import models



def get_models(which_model):

    if which_model == "resnet_fbf_paper":
        nz = 128
        nc = 3
        ngf = 128
        batch_norm_g = True

        ndf = 128
        batch_norm_d = False
        netG = models.ResNet32Generator(nz, nc, ngf, batch_norm_g)
        netD = models.ResNet32Discriminator(nc, 1, ndf, batch_norm_d)

        netG.apply(weight_init_fbf_paper)
        netD.apply(weight_init_fbf_paper)

    elif which_model == "dcgan_fbf_paper":
        nz = 128
        nc = 3
        ngf = 64
        batch_norm_g = True

        ndf = 64
        batch_norm_d = True

        netG = models.DCGAN32Generator(nz, nc, ngf, batchnorm=batch_norm_g)
        netD = models.DCGAN32Discriminator(nc, 1, ndf, batchnorm=batch_norm_d)

        netG.apply(weight_init_fbf_paper)
        netD.apply(weight_init_fbf_paper)


    else:

        # Number of channels in the training images. For color images this is 3
        nc = 3

        # Size of z latent vector (i.e. size of generator input)
        nz = 100

        # Size of feature maps in generator
        ngf = 64

        # Size of feature maps in discriminator
        ndf = 64

        netG = models.pytorch_tutorialGenerator(nz, ngf, nc)

        netD = models.pytorch_tutorialDiscriminator(ndf,nc)

    return netG,netD,nz

def sampler(b_size,nz,sampler_option):

    if sampler_option=="pytorch_tutorial":
        noise = torch.randn(b_size, nz, 1, 1)
    else:
        noise = torch.zeros((b_size,nz)).normal_()
    return noise

bce_criterion = nn.BCELoss()
def compute_gan_loss(output,loss_type,b_size,realOrFake,genOrDis):
    if loss_type == "BCE":
        criterion = bce_criterion
        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.
        if realOrFake=="real":
            label = torch.full((b_size,), real_label, dtype=torch.float).cuda()
        else:
            label = torch.full((b_size,), fake_label, dtype=torch.float).cuda()

        err = criterion(output, label)

    elif loss_type == "wgan":
        if (genOrDis=="gen") or (realOrFake=="real"):
            err = - output.mean()
        else:
            #fake data passing to discriminator
            err = output.mean()

    else:
        raise NotImplementedError()

    return err

def get_data(which_data):

    if which_data == "celebra":
        # Spatial size of training images. All images will be resized to this
        # size using a transformer.
        image_size = 64



        # Root directory for dataset
        dataroot = "data/celebra"

        # We can use an image folder dataset the way we have it setup.
        # Create the dataset

        dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    elif which_data == "cifar":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        dataset = dset.CIFAR10(root='./data/cifar', train=True, transform=transform, download=True)
    else:
        image_size = 64
        dim1 = image_size
        dim2 = image_size
        numDataPoints = 1000
        numLabels = 10

        dataset = rd.RandomDataSet(dim1,dim2,numDataPoints,numLabels)

    return dataset


def clip(net,clip_amount):
    if clip_amount is None:
        return

    for p in net.parameters():
        p.data.clamp_(-clip_amount, clip_amount)


def get_inception_score(n_samples,nz,netG):
    all_samples = []
    samples = torch.randn(n_samples, nz)
    for i in range(0, n_samples, 100):
        batch_samples = samples[i:i+100].cuda()
        all_samples.append(netG(batch_samples).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    t0 = time.time()
    out = iscore.inception_score(torch.from_numpy(all_samples), resize=True, cuda=True)[0]
    t0 = time.time()-t0
    return out,t0



def weight_init_fbf_paper(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.)


def weights_init_pytorch_tutorial(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        netG.apply(weights_init_pytorch_tutorial)
        netD.apply(weights_init_pytorch_tutorial)


class Minibatch:
    def __init__(self,dataloader):
        self.dataloader = dataloader
        self.data_loader_iter = iter(self.dataloader)
        self.epoch_progress = 0
    def get(self):

        try:
            data = next(self.data_loader_iter)
            self.epoch_progress += 1
            newEpoch = False
        except:
            self.data_loader_iter = iter(self.dataloader)
            data = next(self.data_loader_iter)
            self.epoch_progress = 0
            newEpoch = True
        return data,newEpoch


class ProgressMeter:
    def __init__(self,n_samples,nz,netG,num_epochs,dataloader,results,eval_freq,
                 sampler_option,clip_amount,param_setting_str,dt_string,getInceptionScore):
        self.n_samples = n_samples
        self.nz = nz
        self.netG = netG
        self.num_epochs = num_epochs
        self.lendataloader = len(dataloader)
        self.results = results
        self.results['eval_freq'] = eval_freq
        self.results['sampler_option']=sampler_option
        self.results['clip_amount'] = clip_amount
        self.results['param_setting_str'] = param_setting_str
        self.dt_string = dt_string
        self.getInceptionScore = getInceptionScore


        self.epoch_running_times = []
        self.iscores = []
        self.timestamps = [0.0]
        self.forwardStepStamps = []
        self.epochStamps = []
        self.G_losses = []
        self.D_losses = []

        self.t0 = time.time()

    def record(self,forward_steps,epoch,i,errD,errG,D_x,D_G_z1,D_G_z2):
        self.t0 = time.time() - self.t0
        if self.getInceptionScore:
            iscore,t_iscore = get_inception_score(self.n_samples,self.nz,self.netG)
        else:
            iscore,t_iscore = -1,-1
        self.iscores.append(iscore)
        print(f"time to get IS: {t_iscore}")
        print(f"time since last print out: {self.t0}")
        self.timestamps.append(self.timestamps[-1]+self.t0)
        self.forwardStepStamps.append(forward_steps)
        self.epochStamps.append(epoch)
        self.t0 = time.time()

        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t IS: %.4f'
              % (epoch, self.num_epochs, i, self.lendataloader,
                 errD, errG, D_x, D_G_z1, D_G_z2,self.iscores[-1]))

        # Save Losses for plotting later
        self.G_losses.append(errG)
        self.D_losses.append(errD)

    def save(self,ttot,tepoch):
        self.epoch_running_times.append(tepoch)
        self.results['epoch_running_times'] = self.epoch_running_times
        self.results['forwardStepStamps']=self.forwardStepStamps
        self.results['iscores']=self.iscores
        self.results['timestamps']=self.timestamps[1:]
        self.results['num_epochs']=self.num_epochs
        self.results['G_losses']=self.G_losses
        self.results['D_losses']=self.D_losses
        self.results['epochStamps']=self.epochStamps
        self.results['total_running_time']=ttot

        with open('results/'+self.results['algorithm']+'/results_'+self.dt_string, 'wb') as handle:
            pickle.dump(self.results, handle)
