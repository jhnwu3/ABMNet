#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=EvaluationOfIndrani
#SBATCH --partition=general,das,himem
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/temporal_rnn_interpret%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu

python3 temporal_interpret.py
# mprof plot
# python3 -m memory_profiler spatial.py
