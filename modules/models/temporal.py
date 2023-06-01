import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(TemporalComplexModel, self).__init__()

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

def train_temporal_model():
    hidden_size = 256
    learning_rate = 0.1
    num_epochs = 100
    input_size = 3
    output_size = 3
    layer_size = 5

    tdata = np.loadtxt("3linyt.csv", delimiter=",", dtype=float)
    tdata = tdata[:, [0,1,2]]

    train_series = tdata[0:80]
    test_series = tdata[80:100]

    train_series[:] = (train_series[:] - train_series[:].min()) / (train_series[:].max() - train_series[:].min())

    train_input = []
    train_output = []

    for i in range(train_series.shape[0] - 10):
        row1 = train_series[i:i+10]
        row2 = train_series[i+1:i+10+1]
        train_input.append(torch.from_numpy(row1).float())
        train_output.append(torch.from_numpy(row2).float())

    test_series[:] = (test_series[:] - test_series[:].min()) / (test_series[:].max() - test_series[:].min())

    test_input = []
    test_output = []

    for i in range(test_series.shape[0] - 10):
        row1 = test_series[i:i+10]
        row2 = test_series[i+1:i+10+1]
        test_input.append(torch.from_numpy(row1).float())
        test_output.append(torch.from_numpy(row2).float())

    model = TemporalModel(input_size=input_size, output_size=output_size, hidden_dim=hidden_size, n_layers=layer_size)
    model = model.float()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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

    # plot_expected = np.array(plot_expected)
    # plot_predicted = np.array(plot_predicted)
    # plot_expected[:] = ((plot_expected[:] * (test_series[:].max() - test_series[:].min())) + test_series[:].min())
    # plot_predicted[:] = ((plot_predicted[:] * (test_series[:].max() - test_series[:].min())) + test_series[:].min())

    plt.plot(plot_expected, c="b")
    plt.plot(plot_predicted, c="r")
    plt.show()