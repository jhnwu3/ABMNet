import pickle
import torch
from modules.data.temporal import *

file1 = "data/time_series/indrani_zeta_ca_no_zeroes_2500.pickle"
file2 = "data/time_series/indrani_zeta_ca_no_zeroes.pickle"
path = "data/time_series/indrani_zeta_ca_no_zeroes_3500.pickle"
combined = combine_temporal_pickles(file1, file2, save=True, path=path)
print(len(combined["outputs"]))
print(len(combined["rates"]))

seq_data = TemporalDataset(path,min_max_scale=False, standardize_inputs=False)
# find 4 time points
time_points = [10,250,750,1750]
# process these into a new dataset, and then train
for t in time_points:
    data = generate_static_dataset(seq_data,t)
    data.write_to_csv("data/static/indrani/indrani_zeta_ca_t" +str(t) + ".csv")