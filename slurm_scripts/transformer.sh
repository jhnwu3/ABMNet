#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=TransformerTest
#SBATCH --partition=general,das,himem
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/transformer%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu

python3 temporal_transformer.py
# mprof plot
# python3 -m memory_profiler spatial.py
