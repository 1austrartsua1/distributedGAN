#!/bin/bash
#SBATCH -J ps32
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time 01:30:00
#SBATCH --account=m1759
#SBATCH -o outputs/ps32.out

# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch

srun python mainGAN.py -a ps -r ps32

