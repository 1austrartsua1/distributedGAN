#!/bin/bash
#SBATCH -J distDCGAN
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time 0:10:00
#SBATCH --account=m1759
#SBATCH -o distDCGAN.out

module load cgpu
module load pytorch

srun python mainDCGAN.py --distributed-backend 'nccl'
