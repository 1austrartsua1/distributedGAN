#!/bin/bash
#SBATCH -J gda2
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --time 03:59:00
#SBATCH --account=m1759
#SBATCH -o outputs/gda2.out
#SBATCH -q special

# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch

srun python mainGAN.py -a gda -r gda2 --num_epochs 500


