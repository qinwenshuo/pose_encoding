#!/bin/bash
#SBATCH --partition=shared
#SBATCH -A lisik33
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=jupyter
#SBATCH --output=logs/jupyter.log

ml load anaconda3/2024.02-1
conda activate pose

# Get the compute node hostname and a random port
NODE=$(hostname)
PORT=$(shuf -i 8000-9999 -n 1)

echo "============================================"
echo "Node: $NODE"
echo "Port: $PORT"
echo "SSH tunnel command (run on your laptop):"
echo "ssh -N -L ${PORT}:${NODE}:${PORT} <YourRockfishID>@login.rockfish.jhu.edu"
echo "============================================"

jupyter notebook --no-browser --port=${PORT} --ip=0.0.0.0