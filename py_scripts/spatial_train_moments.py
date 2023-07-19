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



data = SingleInitialCorrelationDataset("../gdag_data/gdag_autocorr_r1.pickle")
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
print("Training Dataset Size:", train_size)
print("Test Dataset Size:",test_size)
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])

# model, device = SpatialModel.train_moments(train_data, 45, n_inputs=data.n_inputs,n_outputs=data.n_outputs,n_rates=data.n_rates,
#                            initial_graph=data.initial_graph, edges=data.edges, hidden_channels=128, path="model/gdag_gat.pt")
print(data.n_outputs)
model, device = SpatialModel.train_gat(train_data, 45, n_inputs=data.n_inputs,n_outputs=data.n_outputs,n_rates=data.n_rates,
                           initial_graph=data.initial_graph, edges=data.edges, hidden_channels=128, path="model/gdag_gat_corr.pt")



test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=True)
print("Edge Length:", data.edges.size())
criterion = torch.nn.MSELoss()
model.eval()
test_loss = 0
predictions = []
ground_truth = []
with torch.no_grad():
    for rates, output_graph in test_dataloader:
        out = model(data.initial_graph.to(device), data.edges.to(device), rates.to(device))
        test_loss += criterion(out.detach(), output_graph.to(device))
        predictions.append(out.cpu().numpy())
        ground_truth.append(output_graph.cpu().numpy())

predictions = np.array(predictions).squeeze()
ground_truth = np.array(ground_truth)
print("Test Average MSE:", test_loss.item() / len(test_data))
plot_histograms(test_dataset=ground_truth,predictions=predictions, output="graphs/gnn/test_gat_corr")
plot_scatter(true=ground_truth, predictions=predictions, output="graphs/gnn/test_gat_corr")