import torch
import time
from modules.models.temporal import *
from modules.models.spatial import *
from modules.models.simple import *



# return MSE metric of moments for simple MLPs
# %%
def evaluate(model, dataset, use_gpu = False, batch_size=None):
    
    criterion = nn.MSELoss()
    if tc.cuda.is_available() and use_gpu:
        device = tc.device("cuda")
        model = model.cuda()
        criterion = criterion.cuda()
        using_gpu = True
    else:
        device = tc.device("cpu")
        using_gpu = False

    print(f"Using GPU: {using_gpu}")
    model.eval()
    loss = 0
    start_time = time.time()
    predicted = []
    tested = []
    for ex in range(len(dataset)):
        sample = dataset[ex]
        input = sample[0].squeeze()
        output = sample[1].squeeze()
        if batch_size is not None: 
            input = input.unsqueeze(dim=0)
            output = output.unsqueeze(dim=0)
        prediction = model.forward(input.to(device))
        loss += criterion(prediction, output.to(device))
        tested.append(output.squeeze().cpu().detach().numpy())
        predicted.append(prediction.squeeze().cpu().detach().numpy())
        
    return loss.cpu().detach().numpy() / len(dataset), time.time() - start_time, np.array(predicted), np.array(tested)


# as in evaluate an RNN
def evaluate_temporal(data, model : TemporalComplexModel, criterion, device, batch_size=None):
    predicted = []
    truth = []
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        h_0 = torch.zeros(model.n_layers, batch_size, model.hidden_dim)
        c_0 = torch.zeros(model.n_layers, batch_size, model.hidden_dim)

        # Initialize the LSTM hidden state
        hidden = (h_0.to(device), c_0.to(device))
        for rates, input, output in test_dataloader:
            if len(input.size()) > 3:
                input = input.squeeze()
                output = output.squeeze()
            out, hidden = model(input.to(device).float(), (hidden[0].detach().to(device), hidden[1].detach().to(device)), rates.to(device).float())
            test_loss += criterion(out, output.to(device)).cpu().item() / len(data)
            predicted.append(out.squeeze().cpu().numpy())
            truth.append(output.squeeze().cpu().numpy())
    # Now stack each of the numpy arrays if truth and predicted along the correct dimension
    
        
    # print(truth)
    return test_loss, np.concatenate(truth, axis=0), np.concatenate(predicted, axis=0)
    
# evaluates a transformer model on indrani's model.
def evaluate_temporal_transformer(data, model, criterion, device, batch_size=None):
    predicted = []
    truth = []
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    avg_loss = 0
    model.eval()
    start_time = time.time()
    with torch.no_grad():

        for rates, input, output in test_dataloader:
            if len(input.size()) > 3:
                input = input.squeeze()
                output = output.squeeze()
            if batch_size is not None: 
                rates = rates.unsqueeze(dim=1)
            out = model(rates.to(device), input.to(device))
            avg_loss += criterion(out, output.to(device)).cpu().item() / len(data) # get average
            predicted.append(out.squeeze().cpu().numpy())
            truth.append(output.squeeze().cpu().numpy())
    # Now stack each of the numpy arrays if truth and predicted along the correct dimension
    return avg_loss, np.concatenate(truth, axis=0), np.concatenate(predicted, axis=0), time.time() - start_time


def evaluate_transformer_encoder(data, model, criterion, device, batch_size=None):
    predicted = []
    truth = []
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    avg_loss = 0
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for rates, output in test_dataloader:
            if len(input.size()) > 3:
                output = output.squeeze()
            if batch_size is not None: 
                rates = rates.unsqueeze(dim=1)
            out = model(rates.to(device))
            avg_loss += criterion(out, output.to(device)).cpu().item() / len(data) # get average
            predicted.append(out.squeeze().cpu().numpy())
            truth.append(output.squeeze().cpu().numpy())
    # Now stack each of the numpy arrays if truth and predicted along the correct dimension
    return avg_loss, np.concatenate(truth, axis=0), np.concatenate(predicted, axis=0), time.time() - start_time