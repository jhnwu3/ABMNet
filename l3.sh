#!/bin/bash
#SBATCH --time=0-20:10:00 
#SBATCH --job-name=ABMNet_GPU
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/l3%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100

# python3 main.py -i 'data/static/l3p_t1.csv' -d 4 -h 64 --epochs 50 -o 'l3p_t1' --save --gpu
# python3 main.py -i 'data/static/l3p_t2.csv' -d 4 -h 64 --epochs 50 -o 'l3p_t2' --save --gpu
# python3 main.py -i 'data/static/l3p_t3.csv' -d 4 -h 64 --epochs 50 -o 'l3p_t3' --save --gpu
# python3 main.py -i 'data/time_series/l3pt_i.csv' -d 4 -h 64 --epochs 50 -o 'l3p_i' --save --gpu
python3 main.py -i 'data/static/l3p_10k_t3_5kss.csv' -d 6 -h 128 --epochs 50 -o 'l3p_i' --save --gpu
#python3 main.py -i 'data/NL6_means.csv' -h 128 -d 5 -o 'nl6means_h128_d5.csv' --epochs 100 --normalize --type res_n --gpu

#python3 main.py -i 'data/gdag_1300ss_covs.csv' -h 128 --epochs 200 -d 20 -o 'gdag1300ss_large_norm' --gpu --normalize --type res_nn

#python3 main.py -i 'data/NL6_in.csv' -h 64 -d 10 -o 'nl6in' --epochs 200 --gpu

#python3 main.py -i 'data/NL6_means.csv' -h 512 -d 32 -o 'nl6means_large' --epochs 1000 --gpu --type res_nn --normalize
