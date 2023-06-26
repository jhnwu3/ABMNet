import torch as tc 
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from modules.data.mixed import *
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import time
import resource
class ReLuBlock(nn.Module):
    def __init__(self, i, o, initialize_kaiming=True):
        super(ReLuBlock, self).__init__()

        self.ff = nn.Sequential(
            nn.Linear(i, o),
            nn.ReLU(),
        )
        if initialize_kaiming:
            for layer in self.ff: 
                if isinstance(layer, nn.Linear):
                    init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                
    def forward(self, x):
        output = self.ff.forward(x)
        return output


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, depth, output_size, initialize_kaiming=True):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.input_ff = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        hidden_layers = []
        for i in range(depth):
            hidden_layers.append(ReLuBlock(hidden_size, hidden_size, initialize_kaiming))
        
        self.hidden = nn.Sequential(*hidden_layers)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        if initialize_kaiming:
            init.kaiming_normal_(self.input_ff.weight, mode="fan_in", nonlinearity="relu")
            init.kaiming_normal_(self.output.weight,mode="fan_in", nonlinearity="relu")
        # self.fc4 = nn.Linear(10,output_size)
        # self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input):
        out = self.input_ff(input)
        out = self.bn1(out)
        out = self.hidden(out)
        out = self.bn2(out)
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
