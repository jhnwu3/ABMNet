python3 main.py -i 'data/linear.csv' -h 100 --epochs 100 -d 2 -o 'linear3protein'
python3 main.py -i 'data/gdag_400ss.csv' -h 1000 --epochs 1000 -d 3 -o 'gdagABM_more_avg'
python3 main.py -i 'data/NL6P.csv' -h 1000 --epochs 1000 -d 5 -o 'nl6'
python3 main.py -i 'data/gdag_1300ss.csv' -h 100 --epochs 1000 -d 2 -o 'nl6' --transform