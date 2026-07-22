#!/bin/bash
#SBATCH --partition=shared
#SBATCH -A lisik33
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=encoding
#SBATCH --output=/dev/null

SCRIPT="figure_2"

exec > logs/${SCRIPT//\./_}.log 2>&1

ml load anaconda3/2024.02-1
conda activate pose

python -u -m scripts.${SCRIPT}
