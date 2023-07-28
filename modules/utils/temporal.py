from modules.utils.interpret import *
from modules.utils.evaluate import *
from modules.utils.graph import *
import torch 


# Given the first t_observed inputs, generate future trajectories using its own predictions. Then compare and contrast.
# returns a matrix of predicted sequences, the mse, and the ground truth sequence, the times used as input for generation.
def generate_temporal(data, model : TemporalComplexModel, criterion, device, t_observed, batch_size = None, fs=1):
    predicted = []
    ground_truth = []
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=False)
    test_loss = 0
    model.eval()
    i = 1
    with torch.no_grad():
        time_start = time.time()
        for rates, input, output in test_dataloader:
            truth, pred, loss = generate_temporal_single(input, rates,
                                                         output, model,
                                                         criterion, device, 
                                                         t_observed, fs=fs)
            ground_truth.append(truth) # matrix of just ground truths.
            test_loss += loss / len(data)
            predicted.append(pred)
            if i % 10 == 0:
                print("Time For 10 Sequences of Generation:", time.time() - time_start, " With Average MSE:", test_loss)
                time_start = time.time()
            i += 1
    return test_loss.cpu().detach().numpy(), np.array(ground_truth), np.array(predicted)


# recursively generates data and computes a loss
def generate_temporal_single(input, rates, output, model, criterion, device, t_observed, fs = 4):
    test_loss = 0
    predicted = []
    truth = [] # let's actually make our lives easier by simply just returning the matching truth element
    hidden = (torch.zeros(model.n_layers, model.hidden_dim).detach(), torch.zeros(model.n_layers, model.hidden_dim).detach())
    # feed it first t_observed time points
    curr = input[:t_observed].to(device).float() # current input of time points that the RNN might see.
    out, hidden = model(curr, (hidden[0].detach().to(device), hidden[1].detach().to(device)), rates.to(device).float())

    tru = output[fs - 1:t_observed + fs - 1]
    test_loss += criterion(out.squeeze(), tru.to(device)).cpu().detach() # keep in mind we're 1 off because of what out and in are. 

    truth.append(tru[-fs:].cpu().numpy())
    predicted.append(out[-fs:].cpu().numpy())
    # now let's recursively generate given the new out and hidden units until we run out of space. Keep in mind, this only works for the output here.
    # print("Sequence Length:", output.size()[0])
    # note we need to offset by 1 to make sure the MSEs are aligned
    for t in range(fs, output.size()[0] - t_observed, fs): # account for the missing one.
        # print(curr.size())
        # print(out[-1].size())
        curr = torch.cat([curr[fs:].squeeze(),out[-fs:].squeeze()])
        out, hidden = model(curr.float().squeeze(), (hidden[0].detach().to(device), hidden[1].detach().to(device)), rates.to(device).float())
        predicted.append(out[-fs:].cpu().numpy()) # so I can keep track of all the predictions
        tru = output[t-1: t+t_observed - 1]
        test_loss += criterion(out.squeeze(), tru.to(device)).cpu().detach()
        truth.append(tru[-fs:].numpy())
        # t_observed=3
        # fs = 2
        # 1 2 3 4 5 6
        # input: 1 2 3
        # prediction: 3 4 5 
        # output: 3 4 5, a[2:5]
        # print(tru.size())
    predicted = np.array(predicted)
    predicted = predicted.flatten()
    truth = np.array(truth)
    truth = truth.flatten()
    return truth, predicted, test_loss 
    


# load a model in, and now let's see can it do it from scratch? 
# how many data points in the future can it predict?
# and how many data points does it need??
def generate_time_series(path, model, device, criterion, t_observed, out, fs=1): 
    data = TemporalDataset(path)
    
    test_loss, truth, predicted = generate_temporal(data, model, criterion, device, t_observed, fs=fs)

    print("MSE:", test_loss)
    # save these matrices for future use.
    np.savetxt(out + "_prediction.csv", predicted, delimiter=",")
    np.savetxt(out + "_truth.csv", truth, delimiter=",")

    # generated = np.loadtxt("data/time_series/gen_it"+ str(t_observed) + "_zeta_surrogate.csv", delimiter=",")
    # truth = np.loadtxt("data/time_series/tru_it"+ str(t_observed) + "_zeta_surrogate.csv", delimiter=",")
    # for i in range(0, len(data), 300):
    #     # print(output.size())
    #     # print(output.shape)
    #     print(truth.shape)
    #     print(predicted.shape)
    #     plot_time_series(truth[i,:], predicted[i,:], data.times[t_observed+1:], 
    #                     path=out + str(t_observed) + "_set" + str(i) + ".png")
    
    plot_scatter(truth, predicted, output=out + str(t_observed))


def find_closest_index(arr, value):
    closest_index = np.abs(arr - value).argmin()
    return closest_index

# observed data is in the format of  (t,y), where chosen_times are the indices
def get_corresponding_y(observed_data, chosen_time_indices):
    y = []
    for t in chosen_time_indices:
        tdx = find_closest_index(observed_data[:,0], t)
        y.append(observed_data[tdx, 1])
    return y 

# path = "data/time_series/indrani_gamma_no_zeroes.pickle"
# device = ""
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     using_gpu = True
# else:
#     device = torch.device("cpu")
#     using_gpu = False

# criterion = torch.nn.MSELoss().to(device)
# model = torch.load("model/indrani_gamma_nzero_chunked_fs4.pt", map_location=torch.device('cpu'))
# print(model)
# model = model.to(device)
# output_path = "graphs/temporal/gamma_chunked_fs4"
# generate_time_series(path, model, device, criterion, t_observed=20, out = output_path, fs=4)
