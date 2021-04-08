import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import time

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
