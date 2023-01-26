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


def get_depth():
    return int(sys.argv[sys.argv.index('-d') + 1])

def get_data(): 
    return sys.argv[sys.argv.index('-i') + 1]

def get_epochs():
    return int(sys.argv[sys.argv.index('--epochs') + 1])

def get_output(): 
    return sys.argv[sys.argv.index('-o') + 1]

def get_hidden():
    return int(sys.argv[sys.argv.index('-h') + 1])

def use_gpu():
    return '--gpu' in sys.argv

def save_model():
    return '--save' in sys.argv
 
if __name__ == '__main__':
    
    # Get Data and Parameters
    using_GPU = use_gpu()
    n_epochs = get_epochs()
    hidden_size = get_hidden()
    csv_file = get_data()
    saving_model = save_model()
    output_name = get_output()
    depth = get_depth()
    abm_dataset = ABMDataset(csv_file, root_dir="data/")
    train_size = int(0.3 * len(abm_dataset))
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
    ABMNet = train_nn(train_dataset, input_size=input_len, hidden_size=hidden_size, depth=depth, output_size=output_len, nEpochs=n_epochs, use_gpu=using_GPU)
    if saving_model:
        tc.save(ABMNet, 'model/' + output_name)
    # Cross Validate Using Training Dataset to Find Best Parameters. Will do later just to see how it is.
    
    
    # Validate On Test
    mse, time_to_run, predictions = evaluate(ABMNet, test_dataset, use_gpu=using_GPU)
    print('Final Average MSE On Test Dataset:', mse, ', Time For Inference:', time_to_run)
    np.savetxt('data/nn_output/' + output_name + '.csv', predictions, delimiter=',')
    plot_histograms(test_dataset, predictions, output='data/graphs/' + output_name)