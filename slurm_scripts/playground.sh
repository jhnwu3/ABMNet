git pull

python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t750.csv' -h 64 -d 6 --epochs 150 -o 'ixr_1k_zeta_ca_t750' --save --gpu --normalize 


git add .
git commit -m "$"
git push

# test mse: 0.01 for d=10, 256=h, 150=epochs
# mse: 0.