import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch 
import gc
from torch.cuda.amp import autocast
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.spatial import *
from modules.utils.train import *



data = SingleInitialMomentsDataset("../gdag_data/gdag_spatial_moments.pickle")
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

SpatialModel.train_moments(train_data, 5,n_inputs=data.n_inputs,n_outputs=data.n_outputs,n_rates=data.n_rates, path="model/gdag_gnn.pt")