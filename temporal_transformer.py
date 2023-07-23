# import sys
# sys.path.append('../') # important to adjust what path the libraries are loaded from
from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
from modules.utils.evaluate import *
from modules.utils.graph import *
from sklearn.model_selection import KFold
import torch

dataset = TemporalDataset("data/time_series/indrani_zeta_ca_h_std_norm.pickle", 
                                   standardize_inputs=False, min_max_scale=False)

# train, test split.
train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size

# split dataset, into training set and test
# train_dataset, test_dataset = tc.utils.data.random_split(dataset, [train_size, test_size])

# output dimension is the same as input dimension (I believe)
# model = train_temporal_transformer(dataset=train_dataset, n_rates = dataset.n_rates, hidden_dim=128, 
                        #    output_dim=dataset.input_size, nEpochs=50, batch_size=10)
# tc.save(model, 'model/indrani_transformer.pt')
model = tc.load("model/indrani_transformer.pt")
criterion = torch.nn.MSELoss() 
device = tc.device("cpu")
if tc.cuda.is_available():
    device = tc.device("cuda")
    model = model.cuda()
    criterion = criterion.cuda()
    using_gpu = True

# avg_test_mse, truth, predictions, runtime = evaluate_temporal_transformer(dataset, model, criterion, device, batch_size=20)
# print("Finished Evaluating "+ str(len(dataset)) + " in time (s):" + str(runtime))
# print("With AVG MSE:", avg_test_mse)

# plot_scatter(truth, predictions, output="transformer_full_dataset")

# evaluate with indrani's estimates, and observed dataset's trajectories. 
# let's see what we get out. 

# get indrani's estimates 
data_CD3z_46L_50k = np.loadtxt("data/John_Indrani_data/zeta_Ca_signal/test_data_experiments/CD3z_46L_50k.dat")
indrani_estimates = TemporalDataset("data/time_series/ixr_est_copies.pickle", min_max_scale=True, standardize_inputs=True)
# torch.utils.data.DataLoader(indrani_estimates, batch_size=5, shuffle=False, drop_last=False)
for i in range(30):
    rates, inputs, outputs = indrani_estimates[i]
    rates = torch.transpose(rates.unsqueeze(dim=1))
    predictions = model(rates.to(device), inputs.to(device))
    plt.plot(indrani_estimates.times, predictions.cpu().numpy())
    plt.plot(indrani_estimates.times, outputs.cpu().numpy())

plt.savefig("Transformer_validation_ixr_est.png")
