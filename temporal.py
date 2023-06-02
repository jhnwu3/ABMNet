from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
import torch
data = TemporalDataset("data/time_series/indrani.pickle")
print(len(data))
model, device, test_data = train_temporal_model(data,path="model/indrani_temporal.pt")

criterion = torch.nn.MSELoss()

# evaluate on test_data and see how much error we get.
