import torch as tc 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import time
import mesa
from ABM import *
from NN import *

if __name__ == '__main__':
    
    # Get Data
    csv_file = "data/linear.csv"
    abm_dataset = ABMDataset(csv_file, root_dir="data/")
    train_size = int(0.8 * len(abm_dataset))
    test_size = len(abm_dataset) - train_size
    train_dataset, test_dataset = tc.utils.data.random_split(abm_dataset, [train_size, test_size])
    
    print("Length of Training:",train_size)
    print("Length of Test", test_size)

    # Train Neural Network.
    ABMNet = train_nn(train_dataset,input_size= 5,hidden_size=100, output_size= 9,nEpochs = 100)
    
    # Cross Validate Using Training Dataset to Find Best Parameters.
    
    # Validate On Test
    mse, time_to_run = evaluate(ABMNet, test_dataset)
    print('Final Average MSE On Test Dataset:', mse, ', Time For Inference:', time_to_run)