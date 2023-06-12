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
import resource
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
