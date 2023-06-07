from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
import torch

data = TemporalDataset("data/time_series/indrani.pickle")
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")

# loading a model state dict for now, just so we can look at what we got, in the future, we will just save entire models
hidden_size = 256
lr = 0.0001
n_layers = 5
model = TemporalComplexModel(input_size=data.input_size, hidden_dim=hidden_size, n_layers=n_layers, n_rates = data.n_rates)
model.load_state_dict(torch.load("indrani_searched.pt"))
model = model.to(device)
# we just want some example to look at
with torch.no_grad():
    rates, input, true = data[0]
    predicted = model(input.to(device), rates.to(device))
    plt.plot(data.times, true.cpu().numpy(), label="ground truth")
    plt.plot(data.times, predicted.cpu().numpy(), label="surrogate")
    plt.savefig("graphs/temporal/indrani.png")