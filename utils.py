import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import time
import pickle
import json
import argparse

#locals
import inception_score as iscore
import randomDataLoaderv2 as rd
import models

from pytorch_fid.fid_score import calculate_activation_statistics_kaggle, calculate_frechet_distance

def read_config_file(algorithm):
    config = "config/"+algorithm+'/active_'+algorithm+'.json'
    with open(config) as f:
        data = json.load(f)
    params = argparse.Namespace(**data)

    # also read param tune file
    param_tune = "config/"+algorithm+'/param_tune_'+algorithm+'.json'
    with open(param_tune) as f:
        tune_vals = json.load(f)
    return params,tune_vals


def get_param_count(net):
    nparams = 0
    for p in net.parameters():
        nparams += np.prod(p.shape)
    return nparams


def get_models(which_model,moreFilters):

    if which_model == "resnet_fbf_paper":
        nz = 128
        nc = 3
        if moreFilters:
            ngf = 256
            ndf = 256
        else:
            ngf = 128
            ndf = 128

        batch_norm_g = True


        batch_norm_d = False
        netG = models.ResNet32Generator(nz, nc, ngf, batch_norm_g)
        netD = models.ResNet32Discriminator(nc, 1, ndf, batch_norm_d)

        netG.apply(weight_init_fbf_paper)
        netD.apply(weight_init_fbf_paper)

    elif which_model == "dcgan_fbf_paper":
        if moreFilters:
            ngf = 256
            ndf = 256
        else:
            ngf = 64
            ndf = 64

        nz = 128
        nc = 3

        batch_norm_g = True
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


def get_samples(n_samples,nz,netG):
    all_samples = []
    samples = torch.randn(n_samples, nz)
    for i in range(0, n_samples, 100):
        batch_samples = samples[i:i+100].cuda()
        all_samples.append(netG(batch_samples).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    return torch.from_numpy(all_samples)


def get_inception_score(n_samples,nz,netG):
    all_samples = get_samples(n_samples,nz,netG)
    t0 = time.time()
    out = iscore.inception_score(all_samples, resize=True, cuda=True)[0]
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
                 sampler_option,clip_amount,params,dt_string,getISscore,
                 resultsFileName,getFIDscore,path2FIDstats,moreFilters,paramTuning):
        self.paramTuning = paramTuning
        self.n_samples = n_samples
        self.nz = nz
        self.netG = netG
        self.num_epochs = num_epochs
        self.lendataloader = len(dataloader)
        self.results = results
        self.results['eval_freq'] = eval_freq
        self.results['sampler_option']=sampler_option
        self.results['clip_amount'] = clip_amount
        self.results['param_setting_str'] = params
        if moreFilters:
            startFile = 'results/moreFilters/'
        else:
            startFile = 'results/'

        if paramTuning:
            startFile += 'paramTune/'

        if resultsFileName is None:
            self.resultsFileName = startFile+'results_'+dt_string
        else:
            self.resultsFileName = startFile+self.results['algorithm']+'/'+resultsFileName

        self.getISscore = getISscore
        self.getFIDscore = getFIDscore
        if self.getFIDscore:
            from pytorch_fid.inception import InceptionV3
            # below taken from https://www.kaggle.com/ibtesama/gan-in-pytorch-with-fid
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            self.modelInception = InceptionV3([block_idx])
            self.modelInception.cuda()

            # get statistics for cifar
            with np.load(path2FIDstats) as f:
                self.mu_real, self.sigma_real = f['mu'][:], f['sigma'][:]

        self.epoch_running_times = []
        self.iscores = []
        self.fidscores = []
        self.timestamps = [0.0]
        self.forwardStepStamps = []
        self.epochStamps = []
        self.G_losses = []
        self.D_losses = []

        self.t0 = time.time()

    def record(self,forward_steps,epoch,errD,errG,netG,final=False):
        if (not final) and self.paramTuning:
            return

        self.t0 = time.time() - self.t0
        if self.getISscore:
            iscore,t_iscore = get_inception_score(self.n_samples,self.nz,self.netG)
        else:
            iscore,t_iscore = -1,-1

        if self.getFIDscore:
            fid,t_fid = self.calculateFID(netG)
        else:
            fid,t_fid = -1,-1


        self.fidscores.append(fid)
        self.iscores.append(iscore)

        print(f"time to get IS: {t_iscore:.4f}")
        print(f"time to get FID: {t_fid:.4f}")
        print(f"time since last print out: {self.t0:.4f}")
        self.timestamps.append(self.timestamps[-1]+self.t0)
        self.forwardStepStamps.append(forward_steps)
        self.epochStamps.append(epoch)
        self.t0 = time.time()

        if iscore>=0:
            print('IS: %.4f'% (iscore))
        if fid>=0:
            print('FID: %.4f'% (fid))

        # Save Losses for plotting later
        self.G_losses.append(errG)
        self.D_losses.append(errD)

    def save(self,ttot,tepoch,final=False,paramTuneVal=None,reduceFracAv=-1):
        if (not final) and self.paramTuning:
            return

        if self.paramTuning:
            self.results['forwardStepStamps_'+str(paramTuneVal)]=self.forwardStepStamps
            self.results['iscores_'+str(paramTuneVal)]=self.iscores
            self.results['timestamps_'+str(paramTuneVal)]=self.timestamps[1:]
        else:

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
            self.results['reduceFracAv'] = reduceFracAv



        with open(self.resultsFileName, 'wb') as handle:
            pickle.dump(self.results, handle)

    def calculateFID(self,netG):
        tfid = time.time()
        # get fake images
        images_fake = get_samples(self.n_samples,self.nz,netG)
        #images_fake
        mu_fake,sigma_fake=calculate_activation_statistics_kaggle(images_fake,self.modelInception,batch_size=1)

        fid_value = calculate_frechet_distance(self.mu_real, self.sigma_real, mu_fake, sigma_fake)

        return fid_value,time.time()-tfid
