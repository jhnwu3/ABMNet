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
