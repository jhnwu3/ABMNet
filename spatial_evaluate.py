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

hidden_channels=128
data = SingleInitialCorrelationDataset("../gdag_data/gdag_spatial_moments.pickle")
model = GATComplex(input_dim=data.n_inputs, n_rates=data.n_rates, hidden_dim=hidden_channels,
                                  num_classes=data.n_outputs)
model.load_state_dict(torch.load("model/gdag_gat.pt"))

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
model = model.to(device)
    
test_dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=True)
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
print("Test Average MSE:", test_loss.item() / len(data))
# plot_histograms(test_dataset=ground_truth,predictions=predictions, output="graphs/gnn/all_gat_moms")
plot_scatter(true=ground_truth, predictions=predictions, output="graphs/gnn/all_gat_moms", nSpecies=5)