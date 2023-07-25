# import sys
# sys.path.append('../') # important to adjust what path the libraries are loaded from
from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
from modules.utils.evaluate import *
from modules.utils.graph import *
from sklearn.model_selection import KFold
import torch


fs = 800
dataset = TemporalDatasetEncoder("data/time_series/indrani_zeta_ca_h_std_norm.pickle", 
                                   standardize_inputs=False, min_max_scale=False, steps=fs)

# train, test split.
train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size

# split dataset, into training set and test
train_dataset, test_dataset = tc.utils.data.random_split(dataset, [train_size, test_size])

# output dimension is the same as input dimension (I believe)
model = train_transformer_encoder(dataset=train_dataset, n_rates = dataset.n_rates, hidden_dim=128, 
                           output_dim=dataset.input_size, nEpochs=50, batch_size=10)
tc.save(model, 'model/indrani_transformer_encoder' + str(int(fs)) + '.pt')
# model = tc.load("model/indrani_transformer_" +str(int(fs)) + ".pt")

criterion = torch.nn.MSELoss() 
device = tc.device("cpu")
if tc.cuda.is_available():
    device = tc.device("cuda")
    model = model.cuda()
    criterion = criterion.cuda()
    using_gpu = True

model.eval()
avg_test_mse, truth, predictions, runtime = evaluate_transformer_encoder(dataset, model, criterion, device, batch_size=20)
print("Finished Evaluating "+ str(len(dataset)) + " in time (s):" + str(runtime))
print("With AVG MSE:", avg_test_mse)

# plot_scatter(truth, predictions, output="transformer" + str(fs))

# evaluate with indrani's estimates, and observed dataset's trajectories. 
# let's see what we get out. 
# get indrani's estimates 
data_CD3z_46L_50k = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/CD3z_46L_50k.dat")
# convert trajectory into tensor
indrani_estimates = TemporalDatasetEncoder("data/time_series/ixr_est_copies.pickle", min_max_scale=True, standardize_inputs=True, steps=fs)
# torch.utils.data.DataLoader(indrani_estimates, batch_size=5, shuffle=False, drop_last=False)
fig, axes = plt.subplots(6, 5, figsize=(15, 15))
i = 0
for r in range(6):
    for c in range(5):
        rates, outputs = indrani_estimates[i]
        rates = rates.unsqueeze(dim=1).t()
        predictions = model(rates.to(device))
        axes[r,c].plot(indrani_estimates.times, predictions.cpu().detach().numpy(), c='orange', label="predictions")
        axes[r,c].plot(indrani_estimates.times, outputs.cpu().detach().numpy(), c='blue', label="truth")
        i+=1
plt.legend()
plt.savefig("transformer_validation_ixr_est_enc"+ str(fs) + ".png")

# good sanity check, run random wildly different parameter sets with the same set of trajectories, what do we get? Ideally, should be a different output of parameter sets.

# other sanity check, run same parameter sets, different trajectories
# firstTrajectory = dataset.outputs[0][:-fs]
rates, output = dataset[0]
pseudoRates = torch.zeros(1,5).double()
# pseudoRates -= torch.ones(1,5).double()
prediction = model(pseudoRates.to(device))
plt.figure()
plt.plot(prediction.detach().cpu().numpy(), c='orange', label="Prediction With Zero")
plt.plot(output.detach().cpu().numpy(), c='blue', label="Ground Truth")
plt.legend()
plt.savefig("TransformerEncSanityCheck" +str(fs) + ".png")


# perform parameter estimation, this should fail if the rates have no value. 