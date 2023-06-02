import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from modules.models.simple import *
from modules.models.spatial import *
from modules.data.temporal import *
class TemporalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(TemporalModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=False)  
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
   
    def forward(self, x, hidden):

        # Initializing hidden state for first input using method defined below
        # hidden = self.init_hidden()

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
       
        return out, hidden

class TemporalComplexModel(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, n_rates):
        super(TemporalComplexModel, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_size = input_size
        # Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=False)  
        # Fully connected MLP layer
        self.encoder = EncoderLayer(n_rates, hidden_dim, input_size)
        
        self.fc = NeuralNetwork(input_size*2, hidden_dim ,output_size=input_size)
   
    def forward(self, x, hidden, rates):

        # Initializing hidden state for first input using method defined below
        # hidden = self.init_hidden()

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.lstm(x, hidden)
        out = torch.cat((out, self.rates_encoder(rates)), dim=0)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        # out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
       
        return out, hidden

def train_temporal_model(data, hidden_size, lr, n_epochs, n_layers):
    hidden_size = 256
    learning_rate = 0.1
    num_epochs = 100
    input_size = 3
    output_size = 3
    layer_size = 5

    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        using_gpu = True
    else:
        device = torch.device("cpu")
        using_gpu = False
            
    
    model = TemporalComplexModel(input_size=input_size, output_size=output_size, hidden_dim=hidden_size, n_layers=layer_size)
    model = model.float()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=True) 
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    for epoch in range(1, num_epochs + 1):
        loss_per_epoch = 0
        hidden = (torch.zeros(layer_size, hidden_size).detach(), torch.zeros(layer_size, hidden_size).detach())
        for i in range(len(train_input)):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            out, hidden = model(train_input[i].float(), (hidden[0].detach(), hidden[1].detach()))
            loss = criterion(out.float(), train_output[i].float())
            loss_per_epoch += loss
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
        if(epoch % 10 == 0):
            print(loss_per_epoch)

    model.eval()

    plot_expected = []
    plot_predicted = []

    hidden = (torch.zeros(layer_size, hidden_size).detach(), torch.zeros(layer_size, hidden_size).detach())
    test_loss = 0
    for i in range(len(test_input)):
        out, hidden = model(test_input[i].float(), (hidden[0].detach(), hidden[1].detach()))
        plot_expected.append(test_output[i][9].detach().cpu().numpy())
        plot_predicted.append(out[9].detach().cpu().numpy())
        loss = criterion(out.float(), test_output[i].float())
        test_loss += loss
        loss.backward() # Does backpropagation and calculates gradients

    print("This is the test loss: ", test_loss.cpu().detach().numpy())
