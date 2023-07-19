from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
from modules.utils.evaluate import *
from sklearn.model_selection import KFold
import torch

dataset = TemporalDataset("data/time_series/indrani_zeta_ca_h_std_norm.pickle", 
                                   standardize_inputs=False, min_max_scale=False)

# dataset.save_to_pickle("data/time_series/indrani_zeta_ca_h_std_norm.pickle")
# output dimension is the same as input dimension (I believe)
train_temporal_transformer(dataset=dataset, n_rates = dataset.n_rates, hidden_dim=128, 
                           output_dim= dataset.input_size, nEpochs=5, batch_size=10)