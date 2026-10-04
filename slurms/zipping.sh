#!/bin/bash
#SBATCH --partition=parallel
#SBATCH -A lisik33
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --job-name=zip_files
#SBATCH --output=logs/zip/zip_%j.log

# Load pigz if available as a module, otherwise use system install
# ml load pigz  # uncomment if your cluster has it as a module

# Set number of threads for pigz to match allocated CPUs
THREADS=$SLURM_CPUS_PER_TASK

# Source and destination
SRC_DIR="experiments/SOTA_layers"
OUT_FILE="SOTA_layers.tar.gz"

# Create log directory if it doesn't exist
mkdir -p logs/zip

echo "Starting compression with $THREADS threads..."
echo "Source: $SRC_DIR"
echo "Output: $OUT_FILE"

# Use pigz with all allocated CPUs
tar -I "pigz -p $THREADS" -cvf "$OUT_FILE" "$SRC_DIR"

echo "Done! Archive created at $OUT_FILE"