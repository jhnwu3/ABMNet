#!/bin/bash
#SBATCH --time=0-60:10:00 
#SBATCH --job-name=SpatialABMNetTraining
#SBATCH --partition=general,das,himem
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/temporal_generate%j.txt
#SBATCH --cpus-per-task=2
cd data/John_Indrani_data/zeta/training_kon_koff/
python3 training_data_generation.py
# mprof plot
# python3 -m memory_profiler spatial.py
