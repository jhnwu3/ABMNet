#!/bin/bash
#SBATCH --time=0-99:10:00 
#SBATCH --job-name=abmnet_indrani
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/indrani_zeta_ca_biased%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu

python3 main.py -i 'data/static/ixr_biased_t333.csv' -o 'ixr_biased' -d 10 -h 512 --epochs 5000 --save --gpu --normalize --normalize_out --type res_nn --batch 500
