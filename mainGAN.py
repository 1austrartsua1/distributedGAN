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

parser = argparse.ArgumentParser(description='Distributed GAN training')
parser.add_argument('-d', '--distributed-backend', choices=['mpi', 'nccl', 'nccl-lsf', 'gloo'], help='Specify the distributed backend to use',default='nccl')
parser.add_argument('-a','--algorithm',choices=['fbf','gda','extragrad','ps'],default='fbf')
parser.add_argument('-r','--results',default=None)
parser.add_argument('--which_data',choices=['cifar','celebra','random'],default='cifar')
parser.add_argument('--which_model',choices=["dcgan_fbf_paper","resnet_fbf_paper","pytorch_tutorial"],default="dcgan_fbf_paper")
parser.add_argument('--loss_type',choices=["BCE", "wgan"],default="wgan")
parser.add_argument('--sampler_option',choices=["pytorch_tutorial", "fbf_paper"],default="fbf_paper")
parser.add_argument('--clip_amount',default=0.01,type=float)


args = parser.parse_args()

if args.algorithm == "fbf":
    from algorithms.fbf import main_worker
elif args.algorithm == "gda":
    from algorithms.gda import main_worker
elif args.algorithm == "extragrad":
    from algorithms.extragrad import main_worker
elif args.algorithm == "ps":
    from algorithms.ps import main_worker
else:
    raise NotImplementedError()

def main():
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
        print(f"which data: {args.which_data}")
        print(f"which model: {args.which_model}")
        print(f"loss type: {args.loss_type}")
        print(f"sampler option: {args.sampler_option}")
        print(f"clip amount: {args.clip_amount}")
        print(f"algorithm: {args.algorithm}")
        print(f"distributed backend: {args.distributed_backend}")
        print(f"results file: {args.results}")
        results['which_data']=args.which_data
        results['which_model']=args.which_model
        results['loss_type'] = args.loss_type
        results['algorithm'] = args.algorithm
        results['world_size'] = world_size
        results['node_num'] = node_num
        results['torch_version'] = torch.__version__
        results['distributed_backend']=args.distributed_backend



    print('Local Rank : ', local_rank)
    print('Global Rank : ', global_rank)


    netG,netD,nz = get_models(args.which_model)
    dataset = get_data(args.which_data)

    main_worker(global_rank,local_rank,world_size,netG,netD,
                dataset,nz,args.loss_type,args.sampler_option,args.clip_amount,results,args)


if __name__ == '__main__':
    main()
