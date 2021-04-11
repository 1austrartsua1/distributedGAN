import argparse
import os


#distributed
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

#locals
from distributed import init_workers



from utils import get_models,get_data

algorithm = "fbf"

if algorithm == "fbf":
    from fbf import main_worker
elif algorithm == "gda":
    from gda import main_worker
else:
    raise NotImplementedError()



parser = argparse.ArgumentParser(description='Distributed skeleton (init worker)')
parser.add_argument('-d', '--distributed-backend', choices=['mpi', 'nccl', 'nccl-lsf', 'gloo'], help='Specify the distributed backend to use',default='nccl')

def main():
    args = parser.parse_args()
    which_data = "cifar" #celebra,cifar,random
    which_model = "dcgan_fbf_paper" #resnet_fbf_paper,dcgan_fbf_paper,pytorch_tutorial
    loss_type = "wgan" #BCE, wgan
    sampler_option = "fbf_paper" #pytorch_tutorial, fbf_paper
    clip_amount = 0.01 #>0 or None to turn off

    global_rank, world_size = init_workers(args.distributed_backend)

    ranks_per_node = torch.cuda.device_count()
    local_rank = global_rank % ranks_per_node
    node_num = world_size // ranks_per_node

    results = {}
    if global_rank==0:
        print('pytorch version : ', torch.__version__)
        print('WORLD SIZE:', world_size)
        print('The number of nodes : ', node_num)
        print('Device Count : ', torch.cuda.device_count())
        print(f"which data: {which_data}")
        print(f"which model: {which_model}")
        print(f"loss type: {loss_type}")
        print(f"sampler option: {sampler_option}")
        print(f"clip amount: {clip_amount}")
        results['which_data']=which_data
        results['which_model']=which_model
        results['loss_type'] = loss_type

    print('Local Rank : ', local_rank)
    print('Global Rank : ', global_rank)


    netG,netD,nz = get_models(which_model)
    dataset = get_data(which_data)

    main_worker(global_rank,local_rank,world_size,netG,netD,
                dataset,nz,loss_type,sampler_option,clip_amount,results)


if __name__ == '__main__':
    main()
