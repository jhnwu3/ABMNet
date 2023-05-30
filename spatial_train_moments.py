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

model, device = SpatialModel.train_moments(train_data, 45, n_inputs=data.n_inputs,n_outputs=data.n_outputs,n_rates=data.n_rates,
                           initial_graph=data.initial_graph, edges=data.edges, path="model/gdag_gnn.pt")


test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=True)

criterion = torch.nn.MSELoss()
model.eval()
test_loss = 0
with torch.no_grad():
    for rates, output_graph in test_dataloader:
        out = model(data.initial_graph.to(device), data.edges.to(device), rates.to(device))
        test_loss += criterion(out.detach(), output_graph.to(device))

print("Test Average MSE:", test_loss.item() / len(test_data))