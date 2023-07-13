#!/bin/bash
#SBATCH --time=0-99:10:00 
#SBATCH --job-name=abmnet_soham
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/soham%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu

python3 soham.py