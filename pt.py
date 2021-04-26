import argparse
import os
import numpy as np
import time


#distributed
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

#locals
from distributed import init_workers



from utils import *

parser = argparse.ArgumentParser(description='Parameter tuning code')
parser.add_argument('-d', '--distributed-backend', choices=['mpi', 'nccl', 'nccl-lsf', 'gloo'], help='Specify the distributed backend to use',default='nccl')
parser.add_argument('-a','--algorithm',choices=['fbf','gda','extragrad','ps','psd'],default='extragrad')
parser.add_argument('-r','--results',default=None)
parser.add_argument('--which_data',choices=['cifar','celebra','random'],default='cifar')
parser.add_argument('--which_model',choices=["dcgan_fbf_paper","resnet_fbf_paper","pytorch_tutorial"],default="dcgan_fbf_paper")
parser.add_argument('--loss_type',choices=["BCE", "wgan"],default="wgan")
parser.add_argument('--sampler_option',choices=["pytorch_tutorial", "fbf_paper"],default="fbf_paper")
parser.add_argument('--clip_amount',default=0.01,type=float)
parser.add_argument('--moreFilters', action='store_true')
parser.add_argument('--num_epochs', type=int,default=50)



args = parser.parse_args()


if args.algorithm == "fbf":
    from algorithms.fbf import main_worker
elif args.algorithm == "gda":
    from algorithms.gda import main_worker
elif args.algorithm == "extragrad":
    from algorithms.extragrad import main_worker
elif args.algorithm == "ps":
    from algorithms.ps import main_worker
elif args.algorithm == "psd":
    from algorithms.ps_d import main_worker
else:
    raise NotImplementedError()

params,tune_vals = read_config_file(args.algorithm)

for key in tune_vals:
    args.tuning_variable = key
    break

def main():
    t_pt = time.time()

    global_rank, world_size = init_workers(args.distributed_backend)

    ranks_per_node = torch.cuda.device_count()
    local_rank = global_rank % ranks_per_node
    node_num = world_size // ranks_per_node

    results = {}
    if global_rank==0:
        print("parameter tuning")
        print(f"tuning variable: {args.tuning_variable}")
        print('pytorch version : ', torch.__version__)
        print('WORLD SIZE:', world_size)
        print('The number of nodes : ', node_num)
        print('Device Count : ', torch.cuda.device_count())
        print(f"which data: {args.which_data}")
        print(f"which model: {args.which_model}")
        print(f"loss type: {args.loss_type}")
        print(f"sampler option: {args.sampler_option}")
        print(f"clip amount: {args.clip_amount}")
        print(f"algorithm: {args.algorithm}")
        print(f"distributed backend: {args.distributed_backend}")
        print(f"moreFilters: {args.moreFilters}")
        print(f"results file: {args.results}")
        results['which_data']=args.which_data
        results['which_model']=args.which_model
        results['loss_type'] = args.loss_type
        results['algorithm'] = args.algorithm
        results['world_size'] = world_size
        results['node_num'] = node_num
        results['torch_version'] = torch.__version__
        results['distributed_backend']=args.distributed_backend
        results['moreFilters']=args.moreFilters
        results['num_epochs']=args.num_epochs


    tuneVarVals = tune_vals[args.tuning_variable]
    results['tuneVarVals'] = tuneVarVals
    results['tuning_variable'] = args.tuning_variable

    netG,netD,nz = get_models(args.which_model,args.moreFilters)
    dataset = get_data(args.which_data)
    if global_rank == 0:
        nParamsD = get_param_count(netD)
        nParamsG = get_param_count(netG)
        print(f"num params D: {nParamsD}")
        print(f"Discriminator size (MB): {4*nParamsD/(1e6):.4f}")
        print(f"num params G: {get_param_count(netG)}")
        print(f"Generator size (MB): {4*nParamsG/(1e6):.4f}")


    args.paramTuning = True
    for tuneVal in tuneVarVals:
        if args.tuning_variable == "gamma":
            params.gamma = tuneVal
        elif args.tuning_variable == "lr_dis":
            params.lr_dis_step = tuneVal
            params.lr_gen_step = 0.1*tuneVal

        print(f"\n\n starting tuning val {tuneVal} for {args.tuning_variable}...\n\n")

        args.tuneVal = tuneVal
        main_worker(global_rank,local_rank,world_size,netG,netD,
                dataset,nz,args.loss_type,args.sampler_option,args.clip_amount,
                results,args,params)


    if global_rank==0:
        t_pt = time.time()-t_pt
        print(f"\n\ntotal param tuning time: {t_pt}\n\n")




if __name__ == '__main__':
    main()
