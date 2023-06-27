from modules.utils.interpret import *
from modules.utils.evaluate import *
from modules.utils.graph import *
from modules.utils.temporal import *
import torch 
# data = pickle.load(open(path, "rb"))

# interpret_temporal(data, threshold = 0.00001, path="graphs/temporal/zeta")
path = "data/time_series/indrani_gamma_no_zeroes.pickle"
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    using_gpu = True
else:
    device = torch.device("cpu")
    using_gpu = False

criterion = torch.nn.MSELoss().to(device)
model = torch.load("model/indrani_gamma_nzero_chunked_fs4.pt", map_location=torch.device('cpu'))
print(model)
model = model.to(device)
output_path = "graphs/temporal/gamma_chunked_fs4"
generate_time_series(path, model, device, criterion, t_observed=20, out = output_path, fs=4)
