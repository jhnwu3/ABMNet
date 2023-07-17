#!/bin/bash
#SBATCH --time=0-99:10:00 
#SBATCH --job-name=abmnet_indrani
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/indrani_zeta_ca_triangle%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu

python3 main.py -i 'data/static/indrani/indrani_triangle_features.csv' -o 'ixr3k_zeta_ca_h_triangle' --save --gpu --normalize --normalize_out --type res_nn --batch 500 --cross
