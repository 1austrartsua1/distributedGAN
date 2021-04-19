#!/bin/bash
#SBATCH -J ps8mc
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time 02:59:00
#SBATCH --account=m1759
#SBATCH -o outputs/ps8mc.out

# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch

srun python mainGAN.py -a psd -r ps8 --moreChannels

