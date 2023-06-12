#!/bin/bash
#SBATCH --time=0-10:10:00 
#SBATCH --job-name=SpatialABMNetEvaluation
#SBATCH --partition=general,das,himem
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/spatial_eval%j.txt
#SBATCH --cpus-per-task=16

python3 spatial_evaluate.py
# mprof plot
# python3 -m memory_profiler spatial.py
