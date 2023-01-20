import torch as tc 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import sys
from ABM import *
from NN import *
from GRPH import *

def get_data(): 
    return sys.argv[sys.argv.index('-d') + 1]

def get_epochs():
    return int(sys.argv[sys.argv.index('--epochs') + 1])

def get_output(): 
    return sys.argv[sys.argv.index('-o') + 1]

if __name__ == '__main__':
    
    # Get Data
    nEps = get_epochs()
    csv_file = get_data()
    abm_dataset = ABMDataset(csv_file, root_dir="data/")
    train_size = int(0.8 * len(abm_dataset))
    test_size = len(abm_dataset) - train_size
    train_dataset, test_dataset = tc.utils.data.random_split(abm_dataset, [train_size, test_size])

    sample = train_dataset[0]
    input_len = sample['params'].detach().numpy().shape[0]
    output_len = sample['moments'].detach().numpy().shape[0]
        
    print("Length of Training:",train_size)
    print("Length of Test:", test_size)
    print("Input Dimension:", input_len)
    print("Output Dimension:", output_len)
    # Train Neural Network.
    ABMNet = train_nn(train_dataset, input_size=input_len, hidden_size=100, output_size=output_len, nEpochs=nEps)
    
    # Cross Validate Using Training Dataset to Find Best Parameters. Will do later just to see how it is.
    
    
    # Validate On Test
    mse, time_to_run, predictions = evaluate(ABMNet, test_dataset)
    print('Final Average MSE On Test Dataset:', mse, ', Time For Inference:', time_to_run)
    np.savetxt('data/nn_output/' + get_output() + '.csv', predictions, delimiter=',')
    plot_histograms(test_dataset, predictions, output='data/graphs/' + get_output())