import torch
from modules.models.temporal import *
from modules.models.spatial import *
from modules.models.simple import *



# return MSE metric of moments for simple MLPs
def evaluate(model, dataset, use_gpu = False):
    
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
        input = sample[0]
        output = sample[1]
        prediction = model.forward(input.to(device))
        loss += criterion(prediction.squeeze(), output.squeeze().to(device))
        tested.append(output.cpu().detach().numpy())
        predicted.append(prediction.cpu().detach().numpy())
        
    return loss.cpu().detach().numpy() / len(dataset), time.time() - start_time, np.array(predicted), np.array(tested)


def evaluate_temporal(data, model : TemporalComplexModel, criterion, device):
    predicted = []
    truth = []
    test_dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=False)
    test_loss = 0
    model.eval()
    with torch.no_grad():
        hidden = (torch.zeros(model.n_layers, model.hidden_dim).detach(), torch.zeros(model.n_layers, model.hidden_dim).detach())
        for rates, input, output in test_dataloader:
            out, hidden = model(input.to(device).float(), (hidden[0].detach().to(device), hidden[1].detach().to(device)), rates.to(device).float())
            test_loss += criterion(out.squeeze(), output.to(device)).cpu().detach()
            predicted.append(out.cpu().numpy())
            truth.append(output.cpu().numpy())
    
    
    return test_loss.cpu().detach().numpy(), np.array(truth), np.array(predicted)
    
            