#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus=a100_80gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --job-name=multiscale_interp
#SBATCH --output=logs/%x_%a_%A.out
#SBATCH --error=logs/%x_%a_%A.err
#SBATCH --array=7-11%1


#module load cuda

./build/parallel_multiscale_times ${SLURM_ARRAY_TASK_ID}
#./build/parallel_multiscale ${SLURM_ARRAY_TASK_ID} 
#./build/parallel_multiscale_build ${SLURM_ARRAY_TASK_ID}

