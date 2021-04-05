import argparse
import os


#distributed
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

#locals
from distributed import init_workers

from gda import main_worker

from utils import get_models,get_data,get_criterion




parser = argparse.ArgumentParser(description='Distributed skeleton (init worker)')
parser.add_argument('-d', '--distributed-backend', choices=['mpi', 'nccl', 'nccl-lsf', 'gloo'], help='Specify the distributed backend to use',default='nccl')

def main():
    args = parser.parse_args()
    which_data = "random"
    which_model = "custom_dcgan"
    which_criterion = "BCE"

    global_rank, world_size = init_workers(args.distributed_backend)

    ranks_per_node = torch.cuda.device_count()
    local_rank = global_rank % ranks_per_node
    node_num = world_size // ranks_per_node

    if global_rank==0:
        print('pytorch version : ', torch.__version__)
        print('cuDNN version : ', torch.backends.cudnn.version())
        print('WORLD SIZE:', world_size)
        print('The number of nodes : ', node_num)
        print('Device Count : ', torch.cuda.device_count())

    print('Local Rank : ', local_rank)
    print('Global Rank : ', global_rank)


    netG,netD,nz = get_models(which_model)
    dataset = get_data(which_data)
    criterion = get_criterion(which_criterion)

    main_worker(global_rank,local_rank,world_size,netG,netD,
                dataset,nz,criterion)


if __name__ == '__main__':
    main()
