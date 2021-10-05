import argparse
import os
import time
tmainGan = time.time()


#distributed
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

#locals
from distributed import init_workers



from utils import *

parser = argparse.ArgumentParser(description='Distributed GAN training')
parser.add_argument('-d', '--distributed-backend', choices=['mpi', 'nccl', 'nccl-lsf', 'gloo'],
                    help='Specify the distributed backend to use, (default nccl)',default='nccl')
parser.add_argument('-a','--algorithm',choices=['fbf','gda','extragrad','ps','psd','asyncEG'],default='extragrad',help='default (extragrad)')
parser.add_argument('-r','--results',default=None,help="results file name")
parser.add_argument('--which_data',choices=['cifar','celebra','random'],default='cifar',help='default (cifar)')
parser.add_argument('--which_model',choices=["dcgan_fbf_paper","resnet_fbf_paper","pytorch_tutorial"],
                    default="dcgan_fbf_paper",help='default (dcgan_fbf_paper)')
parser.add_argument('--loss_type',choices=["BCE", "wgan"],default="wgan",help='default (wgan)')
parser.add_argument('--sampler_option',choices=["pytorch_tutorial", "fbf_paper"],default="fbf_paper")
parser.add_argument('--moreFilters', action='store_true')
parser.add_argument('--num_epochs', type=int,default=600)
parser.add_argument('--chunk_reduce', action='store_true')
parser.add_argument('--debug',action='store_true',help='uses a small dummy model and data, default (off)')


args = parser.parse_args()
params,_,_ = read_config_file(args.algorithm)
if params.set_seed:
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Manually-set Seed: ", manualSeed)
    # random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

args.paramTuning = False
args.tuneVal = None

if args.algorithm == "fbf":
    from algorithms.fbf import FBF
    method = FBF()
elif args.algorithm == "gda":
    from algorithms.gda import GDA
    method = GDA(params)
elif args.algorithm == "extragrad":
    from algorithms.extragrad import Extragrad
    method = Extragrad()
elif args.algorithm == "ps":
    from algorithms.ps import PS
    method = PS()
elif args.algorithm == "psd":
    from algorithms.ps_d import PSD
    method = PSD()
elif args.algorithm == "asyncEG":
    from algorithms.async_eg import AsyncEG
    method = AsyncEG()
else:
    raise NotImplementedError()


global_rank, world_size = init_workers(args.distributed_backend)

#ranks_per_node = torch.cuda.device_count()
ranks_per_node=1
local_rank = global_rank % ranks_per_node
node_num = world_size // ranks_per_node

results = {}
if global_rank==0:
    print('\n\n')
    print('pytorch version : ', torch.__version__)
    print('WORLD SIZE:', world_size)
    print('The number of nodes : ', node_num)
    #print('Device Count : ', torch.cuda.device_count())
    print(f"which data: {args.which_data}")
    print(f"which model: {args.which_model}")
    print(f"loss type: {args.loss_type}")
    print(f"sampler option: {args.sampler_option}")
    print(f"clip amount: {params.clip_amount}")
    print(f"algorithm: {args.algorithm}")
    print(f"distributed backend: {args.distributed_backend}")
    print(f"moreFilters: {args.moreFilters}")
    print(f"results file: {args.results}")
    print('\n\n')
    results['which_data']=args.which_data
    results['which_model']=args.which_model
    results['loss_type'] = args.loss_type
    results['algorithm'] = args.algorithm
    results['world_size'] = world_size
    results['node_num'] = node_num
    results['torch_version'] = torch.__version__
    results['distributed_backend']=args.distributed_backend
    results['moreFilters']=args.moreFilters




print('Local Rank : ', local_rank)
print('Global Rank : ', global_rank)

if args.debug:
    netG, netD, nz = getDummyModels()
    dataset = getDummyData()
else:
    netG,netD,nz = get_models(args.which_model,args.moreFilters)
    dataset = get_data(args.which_data)
if global_rank == 0:
    nParamsD = get_param_count(netD)
    nParamsG = get_param_count(netG)
    print(f"num params D: {nParamsD}")
    print(f"Discriminator size (MB): {4*nParamsD/(1e6):.4f}")
    print(f"num params G: {get_param_count(netG)}")
    print(f"Generator size (MB): {4*nParamsG/(1e6):.4f}")


method.main(global_rank,local_rank,world_size,netG,netD,
            dataset,nz,args.loss_type,args.sampler_option,params.clip_amount,results,
            args,params)
tmainGan = time.time() - tmainGan
if global_rank == 0:
    print(f"time in mainGan.py: {tmainGan}")



