from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
from modules.utils.evaluate import *
from sklearn.model_selection import KFold
import torch

batch_size = 64
future_steps = 4
data = TemporalChunkedDataset("data/time_series/indrani_zeta_ca_no_zeroes.pickle",time_chunk_size=20, batch_size=batch_size, steps=future_steps)
print(len(data))
hidden_sizes = [256]
lrs = [0.01, 0.001]
range_epochs = [25, 50, 75]
range_layers = [2,3,5]

hidden_sizes = [256]
lrs=[0.001]
range_epochs = [75]
range_layers = [4]

K = 2
kf = KFold(n_splits=K, shuffle=True, random_state=42) # seed it, shuffle it again, and n splits it.
best_hidden_size = 512
best_lr = 0.0001
best_epochs = 40
best_layers = 5

# split into train and test
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_data, test_data = tc.utils.data.random_split(data, [train_size, test_size])

device = ""
criterion = torch.nn.MSELoss()
min_val_loss = 2114218412401248 # cheap max
# for hidden_size in hidden_sizes:
#     for lr in lrs:
#         for n_epochs in range_epochs:
#             for n_layers in range_layers:
#                 k_val_loss = 0
#                 for fold, (train_index, test_index) in enumerate(kf.split(train_data)):
#                     k_train = tc.utils.data.Subset(train_data, train_index)
#                     k_val = tc.utils.data.Subset(train_data, test_index)
#                     model, device = train_temporal_model(k_train, data.input_size, data.n_rates, hidden_size, lr, n_epochs, n_layers, path="")
#                     criterion = torch.nn.MSELoss().to(device)
#                     # evaluate on test_data and see how much error we get. In particular, we want to 
#                     # display the average and standard deviation of error of each time-series in the test-dataset
#                     val_loss, truth, predicted = evaluate_temporal(k_val, model, criterion, device)
#                     k_val_loss += val_loss / K
                    
#                 # we want to save a checkpoint of the model's state dict, optimizers state dict, val_loss, epoch, n_layers, hidden_size
#                 torch.save({
#                     'n_epochs': n_epochs,
#                     'model_state_dict': model.state_dict(),
#                     'loss': val_loss,
#                     'n_layers': n_layers,
#                     'lr':lr,
#                     'hidden_size':hidden_size
#                     }, 'checkpoints/indrani_model_zeta_chkpt.pth')
                
#                 if k_val_loss  < min_val_loss:
#                     print("New Minimum Average Test Loss:", k_val_loss)
#                     print("For hsize, lr, n_epochs, n_layers")
#                     print(hidden_size, " ",lr , " ",n_epochs, " ", n_layers)
#                     best_hidden_size = hidden_size
#                     best_lr = lr 
#                     best_epochs = n_epochs
#                     best_layers = n_layers
#                     # plot_time_series_errors(truth, predicted, data.times[1:], path="graphs/temporal/validation/errors_h" + str(hidden_size) +"lr" + str(lr) + "nEpc" + str(n_epochs) +"nlay" +str(n_layers) +".png")

# do some final training
print(best_layers)
print(data.input_size)
model, device = train_temporal_model(train_data, input_size=data.input_size,
                                     hidden_size= int(best_hidden_size), 
                                     lr= best_lr, n_rates=data.n_rates, 
                                     n_epochs= best_epochs, 
                                     n_layers=int(best_layers), 
                                     path="model/indrani_zeta_ca_chunked_fs" + str(future_steps) +".pt",
                                     batch_size=batch_size)
# now we evaluate!
test_loss, truth, predicted = evaluate_temporal(test_data, model, criterion, device, batch_size=batch_size)
print("Test MSE:", test_loss)
# plot_time_series_errors(truth, predicted, data.times[1:], path="graphs/temporal/gamma_no_zero_errors_chunked.png")
plot_scatter(truth, predicted, output="graphs/temporal/zeta_ca_chunked_fs" + str(future_steps))     