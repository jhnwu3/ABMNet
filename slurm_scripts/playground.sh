git pull

python3 main.py -i 'data/static/indrani/indrani_zeta_ca_t750.csv' -h 512 -d 10 --epochs 150 -o 'ixr_1k_zeta_ca_t750' --save --gpu --normalize 


git add .
git commit -m "$"
git push

# test mse: 0.01 for d=10, 256=h, 150=epochs
# mse: 0.022 for d=6 h=64, 150 epochs