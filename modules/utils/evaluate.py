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


def evaluate_temporal(data, model : TemporalComplexModel, criterion, device, batch_size=None):
    predicted = []
    truth = []
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        h_0 = torch.zeros(model.n_layers, batch_size, model.hidden_dim)
        c_0 = torch.zeros(model.n_layers, batch_size, model.hidden_dim)

        # Initialize the LSTM hidden state
        hidden = (h_0.to(device), c_0.to(device))
        for rates, input, output in test_dataloader:
            out, hidden = model(input.to(device).float(), (hidden[0].detach().to(device), hidden[1].detach().to(device)), rates.to(device).float())
            test_loss += criterion(out, output.to(device)).cpu().detach()
            predicted.append(out.cpu().numpy())
            truth.append(output.cpu().numpy())
    
    
    return test_loss.cpu().detach().numpy(), np.array(truth), np.array(predicted)
    
