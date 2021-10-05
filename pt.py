import argparse
import os
import numpy as np
import time


import itertools


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
parser.add_argument('-a','--algorithm',choices=['fbf','gda','extragrad','ps','psd','asyncEG'],default='extragrad')
parser.add_argument('-r','--results',default=None,help="results file name")
parser.add_argument('--which_data',choices=['cifar','celebra','random'],default='cifar')
parser.add_argument('--which_model',choices=["dcgan_fbf_paper","resnet_fbf_paper","pytorch_tutorial"],default="dcgan_fbf_paper")
parser.add_argument('--loss_type',choices=["BCE", "wgan"],default="wgan")
parser.add_argument('--sampler_option',choices=["pytorch_tutorial", "fbf_paper"],default="fbf_paper")
parser.add_argument('--moreFilters', action='store_true')
parser.add_argument('--num_epochs', type=int,default=50)
parser.add_argument('--chunk_reduce', action='store_true')



args = parser.parse_args()

def get_method(algo):
    if algo == "fbf":
        from algorithms.fbf import FBF
        method = FBF()
    elif algo == "gda":
        from algorithms.gda import GDA
        method = GDA(params)
    elif algo == "extragrad":
        from algorithms.extragrad import Extragrad
        method = Extragrad()
    elif algo == "ps":
        from algorithms.ps import PS
        method = PS()
    elif algo == "psd":
        from algorithms.ps_d import PSD
        method = PSD()
    elif algo == "asyncEG":
        from algorithms.async_eg import AsyncEG
        method = AsyncEG()
    else:
        raise NotImplementedError()
    return method


params,tune_vals,params_as_dict = read_config_file(args.algorithm,pt=True)
if params.set_seed:
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Manually-set Seed: ", manualSeed)
    # random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)


starting_point = read_progress_file(args.algorithm, args.results)


if starting_point == -1:
    print("tuning already done!!")
    exit(0)

results = read_pickle_file(args.algorithm, args.results,starting_point)


args.tuning_variables = []
totalSettings = 1
numVars = 0
tuningVarListOLists = []
for key in tune_vals:
    args.tuning_variables.append(key)
    totalSettings *= len(tune_vals[key])
    numVars += 1
    tuningVarListOLists.append(tune_vals[key])

tuningIter = itertools.product(*tuningVarListOLists)

def main():
    t_pt = time.time()

    global_rank, world_size = init_workers(args.distributed_backend)

    ranks_per_node = torch.cuda.device_count()
    local_rank = global_rank % ranks_per_node
    node_num = world_size // ranks_per_node

    if global_rank==0:
        print("parameter tuning")
        print('pytorch version : ', torch.__version__)
        print('WORLD SIZE:', world_size)
        print('The number of nodes : ', node_num)
        print('Device Count : ', torch.cuda.device_count())
        print(f"which data: {args.which_data}")
        print(f"which model: {args.which_model}")
        print(f"loss type: {args.loss_type}")
        print(f"sampler option: {args.sampler_option}")
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
        results['paramsListOfLists'] = tuningVarListOLists



    print(f"starting at starting point: {starting_point+1} / {totalSettings}")



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

    #for i in range(starting_point,totalSettings):
    i = -1
    for settings in tuningIter:
        print(settings)
        i += 1
        if i < starting_point:
            continue
        method = get_method(args.algorithm)

        netG, netD, nz = get_models(args.which_model, args.moreFilters)
        dataset = get_data(args.which_data)

        j = 0
        for variable in args.tuning_variables:
            params_as_dict[variable] = settings[j]
            j+= 1

        params_as_dict['lr_gen'] = 10.0*params_as_dict['lr_dis']

        params = argparse.Namespace(**params_as_dict)
        args.tuneVal = i

        print(f"\n\n starting tuning setting {i+1} / {totalSettings}...\n\n")

        method.main(global_rank,local_rank,world_size,netG,netD,
                dataset,nz,args.loss_type,args.sampler_option,params.clip_amount,
                results,args,params)



        if global_rank==0:
            write_to_progress_file(args.algorithm, args.results, i+1)

    if global_rank==0:
        t_pt = time.time()-t_pt
        print(f"\n\ntotal param tuning time: {t_pt}\n\n")
        write_to_progress_file(args.algorithm, args.results, -1)





if __name__ == '__main__':
    main()
