git pull

python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t750.csv' -h 128 -d 5 --epochs 1000 -o 'ixr_1k_zeta_ca_t750_res_batch' --save --gpu --normalize --normalize_out --type res_nn --batch 500
python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t250.csv' -h 128-d 5 --epochs 1000 -o 'ixr_1k_zeta_ca_t250_res_batch' --save --gpu --normalize --normalize_out --type res_nn --batch 500
python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t10.csv' -h 128 -d 5 --epochs 1000 -o 'ixr_1k_zeta_ca_t10_res_batch' --save --gpu --normalize --normalize_out --type res_nn --batch 500
python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t1750.csv' -h 128 -d 5 --epochs 1000 -o 'ixr_1k_zeta_ca_t1750_res_batch' --save --gpu --normalize --normalize_out --type res_nn --batch 500

git add .
git commit -m "$"
git push

# test mse: 0.01 for d=10, 256=h, 150=epochs
# mse: 0.022 for d=6 h=64, 150 epochs