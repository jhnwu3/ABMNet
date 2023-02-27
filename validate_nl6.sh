#!/bin/bash
#SBATCH --time=0-80:10:00 
#SBATCH --job-name=ABMNet_GPU
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/nl6_val%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100

python3 main.py -i 'data/static/NL6P.csv' --cross -o 'nl6_val_large' --gpu --normalize --normalize_out --save