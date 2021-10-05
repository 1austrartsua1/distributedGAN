#!/bin/bash
#SBATCH -J SPSp8_GP
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time 01:00:00
#SBATCH --account=m1759
#SBATCH -o outputs/SPSp8_GP.out


# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch

srun python mainGAN.py -a asyncEG --num_epochs 500 -r SPSp8_GP 






