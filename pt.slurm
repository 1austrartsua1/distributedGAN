#!/bin/bash
#SBATCH -J ptSPSp2GP
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --time 03:59:00
#SBATCH --account=m1759
#SBATCH -o outputs/ptSPSpGP.out

# -N will change number of nodes
# ntasks-per-node will change number of GPUS per node

module load cgpu
module load pytorch


resultsFile=ptSPSpGP
algorithm=asyncEG

#if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
#then
#   rm results/paramTune/${algorithm}/${resultsFile}_progress
#fi


srun python pt.py -r $resultsFile -a $algorithm --num_epochs 150


