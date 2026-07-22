#!/bin/bash
#SBATCH --partition=parallel
#SBATCH -A lisik33
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=48
#SBATCH --array=1-351
#SBATCH --job-name=parallel
#SBATCH --output=logs/parallel/parallel_%a.log

ml load anaconda3/2024.02-1
conda activate pose

python -u -m scripts.03_dnn_encoding --task_id $SLURM_ARRAY_TASK_ID --max_tasks 351