#!/bin/bash
#SBATCH --time=0-10:10:00 
#SBATCH --job-name=ABMNet_GPU
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/gdagn_res%j.txt
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:a100

#python3 main.py -i 'data/NL6P.csv' --epochs 110 -h 128 -d 6 -o 'nl6_normalize_large' --gpu --normalize

python3 main.py -i 'data/gdag_1300ss.csv' -h 128 --epochs 120 -d 32 -o 'gdag1300ss_large_norm' --gpu --normalize --type res_nn
