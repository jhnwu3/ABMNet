python3 main.py -i 'data/static/l3p.csv' -h 100 --epochs 100 -d 2 -o 'linear3protein'
python3 main.py -i 'data/static/l3p.csv' -h 100 --epochs 10 -d 2 -o 'l3p_norm_out' --normalize --normalize_out
python3 main.py -i 'data/static/gdag_400ss.csv' -h 1000 --epochs 1000 -d 3 -o 'gdagABM_more_avg'
python3 main.py -i 'data/static/NL6P.csv' -h 1000 --epochs 1000 -d 5 -o 'nl6'
python3 main.py -i 'data/static/NL6S.csv' -h 64 --epochs 100 -d 5 -o 'nl6s'
python3 main.py -i 'data/static/gdag_1300ss.csv' -h 100 --epochs 1000 -d 5 -o 'gdag1300' --normalize
python3 main.py -i 'data/static/gdag_1300ss_covs.csv' -h 64 --epochs 100 -d 5 -o 'gdag1300_covs' --normalize
python3 main.py -i 'data/static/gdag_1300ss.csv' -h 100 --epochs 100 -d 5 -o 'gdag1300n' --gpu --normalize
python3 main.py -i 'data/static/gdag_1300ss.csv' -h 100 --epochs 100 -d 5 -o 'gdag1300_n_out' --normalize --normalize_out
python3 main.py -i 'data/static/NL6_means.csv' -h 200 --epochs 60 -d 32 -o 'nl6_means' --normalize --type res_nn
python3 main.py -i 'data/static/NL6_in.csv' -h 64 --epochs 100 -d 5 -o 'nl6in' 
python3 main.py -i 'data/static/NL6_2k.csv' -h 64 --epochs 100 -d 3 -o 'nl6s_now' --type res_nn
python3 main.py -i 'data/static/gdag_1300ss_covs.csv' -h 64 --epochs 100 -d 5 -o 'gdag1300_covs' --normalize --normalize_out
python3 main.py -i 'data/static/indrani.csv' -d 2 -h 32 --epochs 70 -o 'indrani' --save --gpu --normalize --normalize_out