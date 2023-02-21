#!/bin/bash
#SBATCH --time=0-20:10:00 
#SBATCH --job-name=ABMNet_GPU
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/nl6_res_means%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100

python3 main.py -i 'data/gdag_1300ss_covs.csv' -h 64 --epochs 50 -d 6 -o 'gdag_default_input' --gpu --normalize_out --save

