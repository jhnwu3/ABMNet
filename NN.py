import torch as tc 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ABM import *
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import time
import mesa

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 120)
        self.fc3 = nn.Linear(120, output_size)
        # self.fc4 = nn.Linear(10,output_size)
        # self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input):
        fc1 = self.fc1(input)
        fc1 = F.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = F.relu(fc2)
        fc3 = self.fc3(fc2)
        # fc3 = F.relu(fc3)
        # fc3 = self.fc4(fc3)
        # output = self.softmax(line)
        return fc3

def train_nn(dataset : ABMDataset, input_size, hidden_size, output_size, nEpochs, use_gpu = False):
    
    model = NeuralNetwork(input_size, hidden_size, output_size).double()
    optimizer = optim.AdamW(model.parameters())
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
    epoch_start = time.time()
    for epoch in range(nEpochs):
        
        loss_this_epoch = 0
        for ex in range(len(dataset)):
            optimizer.zero_grad()
            loss = 0
            sample = dataset[ex]
            input = sample['params']
            output = sample['moments']

            prediction = model.forward(input.to(device))
            loss += criterion(prediction.squeeze(), output.squeeze().to(device))
            loss_this_epoch += loss.item() 
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(repr(f"Finished epoch {epoch} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
            epoch_start = time.time()
            
    return model  


# return MSE metric of moments
def evaluate(model : NeuralNetwork, dataset, use_gpu = False):
    
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
    for ex in range(len(dataset)):
        sample = dataset[ex]
        input = sample['params']
        output = sample['moments']
        prediction = model.forward(input.to(device))
        loss += criterion(prediction.squeeze(), output.squeeze().to(device))
        predicted.append(prediction.cpu().detach().numpy())
        
    return loss.cpu().detach().numpy() / len(dataset), time.time() - start_time, np.array(predicted)