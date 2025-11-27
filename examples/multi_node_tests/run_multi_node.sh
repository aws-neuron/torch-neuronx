#!/bin/bash
#SBATCH --partition=no_bots
#SBATCH --job-name=training_example
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=logs/slurm_%j.out

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the data parallel example
srun ./run_parallel_training.sh
