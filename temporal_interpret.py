from modules.utils.interpret import *
from modules.utils.evaluate import *
import pickle 
import torch 
path = "data/time_series/indrani_zeta.pickle"
data = pickle.load(open(path, "rb"))

# interpret_temporal(data, threshold = 0.00001, path="graphs/temporal/zeta")

# load a model in, and now let's see can it do it from scratch? 
# how many data points in the future can it predict?
# and how many data points does it need??
data = TemporalDataset(path)
t_observed = 2 # can I just do it with 2 data points?
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    using_gpu = True
else:
    device = torch.device("cpu")
    using_gpu = False

criterion = torch.nn.MSELoss().to(device)
model = torch.load("model/indrani/indrani_zeta_small.pt", map_location=torch.device('cpu'))
model = model.to(device)
test_loss, truth, predicted = generate_temporal(data, model, criterion, device, t_observed)
print(truth.shape)
print(predicted.shape)


# save these matrices for future use.

np.savetxt("data/time_series/gen_i2t_zeta_surrogate.csv", predicted, delimiter=",")