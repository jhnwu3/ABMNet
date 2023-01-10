import torch as tc 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import time
import mesa

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.input_size = input_size 
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        # self.softmax = nn.Softmax(dim=0)
        
    def forward(self, input):
        line = self.linear(input)
        # output = self.softmax(line)
        return line

def train_nn(input, expected_output, nEpochs):
    N = input.shape[0]
    D = input.shape[1]
    output_dim = 1
    if expected_output.ndim > 1:
        output_dim = expected_output.shape[1] # 
        
    model = LogisticRegression(D, output_dim)
    optimizer = optim.AdamW(model.parameters())
    loss_fcn = nn.BCEWithLogitsLoss()
    for epoch in range(nEpochs):
        epoch_start = time.time()
        loss_this_epoch = 0
        for ex in range(N):
            optimizer.zero_grad()
            loss = 0
            
            tc_input = tc.tensor(input[ex,:])
            tc_output = tc.tensor(expected_output[ex])
            
            # print(tc_input.shape)
            prediction = model.forward(tc_input.float())
            loss += loss_fcn(prediction.squeeze(), tc_output.squeeze().float())
            loss_this_epoch += loss.item() 
            loss.backward()
            optimizer.step()
            
        print(repr(f"Finished epoch {epoch} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
            
    return model  

