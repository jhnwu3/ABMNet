# problem with the code is that the spatial data can only be run on the cluster, because it's so massive, which means we can't run anything locally
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from textwrap import wrap
import sys
import os
import pickle
import torch 
import torch.nn.functional as F
import torch.nn as nn
import resource
from torch_geometric.nn import GCNConv
from scipy import spatial
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.spatial import *

data = SingleInitialConditionDataset("../gdag_data/gdag_graph_data.pickle")
nEpochs = 5
# for manual testing, load everything at once, and train
# model = GCN(n_features=input_graph.size()[1], n_classes=output_graphs_chunk[0].size()[1])
model = GCNComplex(n_features=data.n_inputs, n_classes= data.n_outputs, n_rates=data.n_rates, hidden_channels=32)
model.train()
model = model.double()
optimizer = torch.optim.AdamW(model.parameters())
criterion = torch.nn.MSELoss()

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.cuda()
    criterion = criterion.cuda()
    using_gpu = True
else:
    device = torch.device("cpu")
    using_gpu = False
    
dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=True) 
for epoch in range(nEpochs):
    loss_per_epoch = 0
    for rates, output_graph in range(dataloader):
        optimizer.zero_grad()
        out = model(data.initial_graph.to(device), data.edges.to(device), rates.to(device))
        loss = criterion(out, output_graph.to(device))
        loss.backward()
        loss_per_epoch+= float(loss)
        optimizer.step()
      
    if epoch % 1 == 0:
        print("Epoch:", epoch, " Loss:", loss_per_epoch)   


# for i in range(len(output_graphs_chunk)):
#     print(output_graphs_chunk[i].size())
# train_profiled(input_graph, output_graphs_chunk, rates_chunk, edges)




# model = train_giuseppe_surrogate_pkl(loaded_data, nEpochs=20)


