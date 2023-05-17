import torch as tc 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from modules.data.mixed import *
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import time
import mesa

class ReLuBlock(nn.Module):
    def __init__(self, i, o):
        super(ReLuBlock, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.ff.forward(x)
        return output


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.input_ff = nn.Linear(input_size, hidden_size)
        hidden_layers = []
        for i in range(depth):
            hidden_layers.append(ReLuBlock(hidden_size, hidden_size))
        
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(hidden_size, output_size)
        # self.fc4 = nn.Linear(10,output_size)
        # self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input):
        out = self.input_ff(input)
        out = self.hidden(out)
        out = self.output(out)
        # output = self.softmax(line)
        return out 


class ResidualReLuBlock(nn.Module):
    def __init__(self, i, o): # i == o
        super(ResidualReLuBlock, self).__init__()
        self.fc1 = nn.Linear(i,o)
        self.fc2 = nn.Linear(i,o)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.fc2(output)
        output = output + x
        return output


class ResidualNN(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size):
        super(ResidualNN, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        print("input", input_size)
        print("hidden", hidden_size)
        self.input_ff = nn.Linear(input_size, hidden_size)
        hidden_layers = []
        for i in range(depth):
            hidden_layers.append(ResidualReLuBlock(hidden_size, hidden_size))
        
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(hidden_size, output_size)
        # self.fc4 = nn.Linear(10,output_size)
        # self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input):
        out = self.input_ff(input)
        out = self.hidden(out)
        out = self.output(out)
        # output = self.softmax(line)
        return out 


class RecurrentNN(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size):
        super(RecurrentNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=depth)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, input):
        out, (h, c) = self.lstm(input)
        out = self.linear(out.squeeze())
        return (out, (h,c))
    
def train_nn(dataset : ABMDataset, input_size, hidden_size, depth, output_size, nEpochs, use_gpu = False):
    
    model = NeuralNetwork(input_size, hidden_size, depth, output_size).double()
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
    model.train()
    epoch_start = time.time()
    
    # enable shuffling so we can 
    loader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)  
    for epoch in range(nEpochs):
        loss_this_epoch = 0
           
        for input, output in loader:
            optimizer.zero_grad()
            loss = 0
          
            prediction = model.forward(input.to(device))
            loss += criterion(prediction.squeeze(), output.squeeze().to(device))
            loss_this_epoch += loss.item() 
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(repr(f"Finished epoch {epoch} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
            epoch_start = time.time()
            
    return model  


def train_res_nn(dataset : ABMDataset, input_size, hidden_size, depth, output_size, nEpochs, use_gpu = False):
    
    model = ResidualNN(input_size, hidden_size, depth, output_size).double()
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
    model.train()
    epoch_start = time.time()
    loader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)
    for epoch in range(nEpochs):
        
        loss_this_epoch = 0
        for input, output in loader:
            optimizer.zero_grad()
            loss = 0
            # sample = dataset[ex]
            # input = sample[0] # no more keys, justt values input is first index, output is second
            # output = sample[1] 
            prediction = model.forward(input.to(device))
            loss += criterion(prediction.squeeze(), output.squeeze().to(device))
            loss_this_epoch += loss.item() 
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(repr(f"Finished epoch {epoch} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
            epoch_start = time.time()
            
    return model  


def train_rnn(dataset : ABMDataset, input_size, hidden_size, depth, output_size, nEpochs, use_gpu = False):
    model = RecurrentNN(input_size, hidden_size, depth, output_size).double()
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
    model.train()
    epoch_start = time.time()
    for epoch in range(nEpochs):
        
        loss_this_epoch = 0
        for s in range(len(dataset)):
            optimizer.zero_grad()
            loss = 0
            
            # start with first sample and then iterate through each time step. 
            
            # sample = dataset[ex]
            # input = sample[0] # no more keys, justt values input is first index, output is second
            # output = sample[1] 
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
