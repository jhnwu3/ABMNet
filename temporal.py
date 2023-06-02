from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
import torch


data = TemporalDataset("data/time_series/indrani.pickle")
print(len(data))
hidden_size=256
lr=0.001
n_epochs=50
n_layers=5
model, device, test_data = train_temporal_model(data, hidden_size, lr, n_epochs, n_layers, path="model/indrani_temporal.pt")

criterion = torch.nn.MSELoss().to(device)

# evaluate on test_data and see how much error we get. In particular, we want to 
# display the average and standard deviation of error of each time-series in the test-dataset
model.eval()
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=None, shuffle=True)
test_loss = 0
predicted = []
truth = []
with torch.no_grad():
    hidden = (torch.zeros(n_layers, hidden_size).detach(), torch.zeros(n_layers, hidden_size).detach())
    for rates, input, output in test_dataloader:
        out, hidden = model(input.to(device).float(), (hidden[0].detach().to(device), hidden[1].detach().to(device)), rates.to(device).float())
        test_loss += criterion(out.squeeze(), output.to(device)).cpu().detach()
        predicted.append(out.cpu().numpy())
        truth.append(output.cpu().numpy())
        
print("Average Test Loss:", test_loss / len(test_data))
plot_time_series_errors(truth, predicted, data.times[1:])