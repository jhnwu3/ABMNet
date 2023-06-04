from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
import torch


data = TemporalDataset("data/time_series/indrani.pickle")
print(len(data))
# REMINDER: no cross-validation just yet, we will cross-validate next week!!
# NEED TO REMIND OURSELVES TO WRITE MORE MODULAR CODE SUCH THAT IT IS EASY FOR US TO RUN CROSSVALIDATION

hidden_sizes = [64, 128, 256]
lrs = [0.01, 0.001, 0.0001]
range_epochs = [25, 50, 75]
range_layers = [2,3,5]

# original just to test parameters
# hidden_size=128
# lr=0.001
# n_epochs=50
# n_layers=3
best_hidden_size = 0
best_lr = 0
best_epochs = 0
best_layers = 0
min_test_loss = 2114218412401248 # cheap max
for hidden_size in hidden_sizes:
    for lr in lrs:
        for n_epochs in range_epochs:
            for n_layers in range_layers:
                model, device, test_data = train_temporal_model(data, hidden_size, lr, n_epochs, n_layers, path="")
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

                if test_loss / len(test_data) < min_test_loss:
                    print("New Minimum Average Test Loss:", test_loss / len(test_data))
                    print("For hsize, lr, n_epochs, n_layers")
                    print(hidden_size, " ",lr , " ",n_epochs, " ", n_layers)
                    best_hidden_size = hidden_size
                    best_lr = lr 
                    best_epochs = n_epochs
                    best_layers = n_layers
                    plot_time_series_errors(truth, predicted, data.times[1:], path="graphs/temporal/validation/errors_h" + str(hidden_size) +"lr" + str(lr) + "nEpc" + str(n_epochs) +"nlay" +str(n_layers) +".png")

# do some final training
model, device, test_data = train_temporal_model(data, best_hidden_size, best_lr, best_epochs, best_layers, "model/indrani_searched.pt")