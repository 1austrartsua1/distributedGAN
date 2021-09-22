#!/bin/bash
#SBATCH -J ptFBFp8
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --time 03:59:00
#SBATCH --account=m1759
#SBATCH -o outputs/ptFBFp8.out

# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch


resultsFile=ptFBFp8
algorithm=asyncEG

#if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
#then
#   rm results/paramTune/${algorithm}/${resultsFile}_progress
#fi


srun python pt.py -r $resultsFile -a $algorithm --num_epochs 150


