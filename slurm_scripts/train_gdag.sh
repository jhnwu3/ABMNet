#!/bin/bash
#SBATCH --time=0-20:10:00 
#SBATCH --job-name=SpatialABMNetTraining
#SBATCH --partition=general,das,himem
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/spatial_train%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100

python3 spatial_train_moments.py
# mprof plot
# python3 -m memory_profiler spatial.py
