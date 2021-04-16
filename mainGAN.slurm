#!/bin/bash
#SBATCH -J extra16
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time 02:00:00
#SBATCH --account=m1759
#SBATCH -o outputs/extra16.out

# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch

srun python mainGAN.py -a extragrad -r extragrad16

