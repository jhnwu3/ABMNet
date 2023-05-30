import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch 
import torch_geometric
from torch.cuda.amp import autocast
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.spatial import *
from modules.utils.train import *


# load the model in
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get predictions and get ground truth stuff from entire dataset, might as well look at it holistically first.
data = SingleInitialMomentsDataset("../gdag_data/gdag_spatial_moments.pickle")
dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=True)
criterion = torch.nn.MSELoss()
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
print("Training Dataset Size:", train_size)
print("Test Dataset Size:",test_size)
train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
predictions = []
ground_truth = []
overall_loss = 0

model = GCNComplexMoments(n_inputs=data.n_inputs, n_outputs= data.n_outputs, n_rates=data.n_rates, hidden_channels=32)
model.load_state_dict(torch.load("model/gdag_gnn.pt"))  # Replace "path_to_model.pth" with the path to your model file
model.to(device)
model = model.float()
with torch.no_grad():
    for rates, output in dataloader:
        out = model(data.initial_graph.to(device).float(), data.edges.to(device), rates.to(device).float())
        overall_loss += criterion(out.detach(), output.to(device))
        predictions.append(out.cpu().numpy())
        ground_truth.append(output.cpu().numpy())

print("Overall Average Test MSE:", overall_loss / len(data))
predictions = np.array(predictions).squeeze()
ground_truth = np.array(ground_truth)
# plot histograms and plot scatter plots
print(predictions.shape)
print(ground_truth.shape)
plot_histograms(test_dataset=ground_truth,predictions=predictions, output="graphs/gnn/out")
plot_scatter(true=ground_truth, predictions=predictions, output="graphs/gnn/out")



