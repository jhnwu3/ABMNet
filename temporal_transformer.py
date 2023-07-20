# import sys
# sys.path.append('../') # important to adjust what path the libraries are loaded from
from modules.data.temporal import *
from modules.models.temporal import *
from modules.utils.train import *
from modules.utils.evaluate import *
from sklearn.model_selection import KFold
import torch

dataset = TemporalDataset("data/time_series/indrani_zeta_ca_h_std_norm.pickle", 
                                   standardize_inputs=False, min_max_scale=False)

# train, test split.
train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size

# split dataset, into training set and test
train_dataset, test_dataset = tc.utils.data.random_split(dataset, [train_size, test_size])


# dataset.save_to_pickle("data/time_series/indrani_zeta_ca_h_std_norm.pickle")
# output dimension is the same as input dimension (I believe)
model = train_temporal_transformer(dataset=train_dataset, n_rates = dataset.n_rates, hidden_dim=128, 
                           output_dim= dataset.input_size, nEpochs=100, batch_size=10)
tc.save(model, 'model/indrani_transformer.pt')
criterion = torch.nn.MSELoss() 
device = tc.device("cpu")
if tc.cuda.is_available():
    device = tc.device("cuda")
    model = model.cuda()
    criterion = criterion.cuda()
    using_gpu = True

avg_test_mse, truth, predictions, time = evaluate_temporal_transformer(test_dataset, model, criterion, device)
print("Finished Evaluating "+ str(len(test_dataset)) + "in time (s):" + str(time))