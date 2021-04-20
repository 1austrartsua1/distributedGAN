#!/bin/bash
#SBATCH -J ps16mc
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time 01:59:00
#SBATCH --account=m1759
#SBATCH -o outputs/ps16mc.out

# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch

srun python mainGAN.py -a ps -r ps16 --moreChannels

