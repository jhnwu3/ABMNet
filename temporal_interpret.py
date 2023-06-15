from modules.utils.interpret import *
from modules.utils.evaluate import *
from modules.utils.graph import *
import pickle 
import torch 
# data = pickle.load(open(path, "rb"))

# interpret_temporal(data, threshold = 0.00001, path="graphs/temporal/zeta")

# load a model in, and now let's see can it do it from scratch? 
# how many data points in the future can it predict?
# and how many data points does it need??
def generate_time_series(path, model, device, criterion, t_observed): 
    data = TemporalDataset(path)
    print(data.times.shape)
    test_loss, truth, predicted = generate_temporal(data, model, criterion, device, t_observed)
    print(truth.shape)
    print(predicted.shape)
    print("MSE:", test_loss)
    # save these matrices for future use.
    np.savetxt("data/time_series/gen_it" + str(t_observed) + "_zeta_surrogate.csv", predicted, delimiter=",")

    generated = np.loadtxt("data/time_series/gen_it"+ str(t_observed) + "_zeta_surrogate.csv", delimiter=",")
    for i in range(0, len(data), 300):
        rates, input, output = data[i]
        print(output.size())
        print(output.shape)
        print(generated.shape)
        plot_time_series(output[t_observed:].cpu().numpy(), generated[i,:], data.times[t_observed:], 
                        path="graphs/temporal/zeta_gen_it"+ str(t_observed) + "_set" + str(i) + ".png")
    
    # plot_scatter(output[t_observed:].cpu().numpy(), generated[i,:], output="graphs/temporal/zeta_gen_it")


path = "data/time_series/indrani_zeta.pickle"
device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    using_gpu = True
else:
    device = torch.device("cpu")
    using_gpu = False

criterion = torch.nn.MSELoss().to(device)
model = torch.load("model/indrani/indrani_zeta_small.pt", map_location=torch.device('cpu'))
print(model)
model = model.to(device)

generate_time_series(path, model, device, criterion, t_observed=50)
generate_time_series(path, model, device, criterion, t_observed=100)
# generate_time_series(path, model, device, criterion, t_observed=10)
# generate_time_series(path, model, device, criterion, t_observed=2)