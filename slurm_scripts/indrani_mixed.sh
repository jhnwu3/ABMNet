#!/bin/bash
#SBATCH --time=0-99:10:00 
#SBATCH --job-name=abmnet_indrani
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --output=./data/slurm_outputs/indrani_zeta_ca%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu

# python3 main.py -i 'data/static/l3p_t1.csv' -d 4 -h 64 --epochs 50 -o 'l3p_t1' --save --gpu
# python3 main.py -i 'data/static/l3p_t2.csv' -d 4 -h 64 --epochs 50 -o 'l3p_t2' --save --gpu
# python3 main.py -i 'data/static/l3p_t3.csv' -d 4 -h 64 --epochs 50 -o 'l3p_t3' --save --gpu
#python3 main.py -i 'data/time_series/l3pt_i.csv' -d 4 -h 64 --epochs 50 -o 'l3p_i' --save --gpu
# python3 main.py -i 'data/static/l3p_10k_t3_5kss.csv' -d 6 -h 128 --epochs 50 -o 'l3p_10k_5kss' --save --gpu

# python3 main.py -i 'data/static/l3p_10k_t3_5kss.csv' -d 4 -h 64 --epochs 100 -o 'l3p_10k_small_res_t3' --save --gpu --type res_nn --normalize --normalize_out
# python3 main.py -i 'data/static/l3p_100k.csv' -d 8 -h 128 --epochs 70 -o 'l3p_100k_medium_res' --save --gpu --normalize --normalize_out --type res_nn
#python3 main.py -i 'data/NL6_means.csv' -h 128 -d 5 -o 'nl6means_h128_d5.csv' --epochs 100 --normalize --type res_n --gpu
# python3 main.py -i 'data/static/l3p_100k.csv' -d 10 -h 256 --epochs 400 -o 'l3p_100k_large_batch_normed' --save --gpu --normalize_out
#python3 main.py -i 'data/gdag_1300ss_covs.csv' -h 128 --epochs 200 -d 20 -o 'gdag1300ss_large_norm' --gpu --normalize --type res_nn

#python3 main.py -i 'data/NL6_in.csv' -h 64 -d 10 -o 'nl6in' --epochs 200 --gpu

#python3 main.py -i 'data/NL6_means.csv' -h 512 -d 32 -o 'nl6means_large' --epochs 1000 --gpu --type res_nn --normalize
# python3 main.py -i 'data/static/l3p_pso.csv' -d 4 -h 64 --epochs 70 -o 'l3p_pso' --save --gpu --normalize --normalize_out 
# python3 main.py -i 'data/static/l3p_pso.csv' -d 4 -h 64 --epochs 50 -o 'l3p_pso_early_tsteps' --save --gpu --normalize_out 


python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t10.csv' -o 'ixr_1k_zeta_ca_t10' --save --gpu --normalize --cross
python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t250.csv' -o 'ixr_1k_zeta_ca_t250' --save --gpu --normalize --cross
python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t750.csv' -o 'ixr_1k_zeta_ca_t750' --save --gpu --normalize --cross
python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t1750.csv' -o 'ixr_1k_zeta_ca_t1750' --save --gpu --normalize --cross

# python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t750.csv' -d 4 -h 64 --epochs 50 -o 'ixr_1k_zeta_ca_t750' --save --gpu --normalize --normalize_out