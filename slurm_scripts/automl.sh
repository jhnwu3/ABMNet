#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=AUTOML
#SBATCH --partition=general,das,himem
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/automl%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu

python3 automl.py
# mprof plot
# python3 -m memory_profiler spatial.py
