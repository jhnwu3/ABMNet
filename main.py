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

def transform_data():
    return '--transform' in sys.argv

def normalize_input():
    return '--normalize' in sys.argv
 
def normalize_output():
    return '--normalize_out' in sys.argv
 
 
def get_nn_type():
    if '--type' in sys.argv:
        return sys.argv[sys.argv.index('--type') + 1]
    else: 
        return -1 
 
if __name__ == '__main__':
    
    # run params
    using_GPU = use_gpu()
    n_epochs = get_epochs()
    hidden_size = get_hidden()
    csv_file = get_data()
    saving_model = save_model()
    output_name = get_output()
    depth = get_depth()
    is_transform = transform_data()
    normalize = normalize_input()
    model_type = get_nn_type()
    normalize_out = normalize_output()
    
    # data
    abm_dataset = ABMDataset(csv_file, root_dir="data/", transform=is_transform, standardize=normalize, norm_out=normalize_out)
    train_size = int(0.8 * len(abm_dataset))
    test_size = len(abm_dataset) - train_size
    train_dataset, test_dataset = tc.utils.data.random_split(abm_dataset, [train_size, test_size])

    sample = train_dataset[0]
    input_len = sample['params'].detach().numpy().shape[0]
    output_len = sample['moments'].detach().numpy().shape[0]
    
    print("Dataset:", csv_file)
    print("Length of Training:",train_size)
    print("Length of Test:", test_size)
    print("Input Dimension:", input_len)
    print("Output Dimension:", output_len)
    print("Depth of NN:", depth)
    print("Hidden Neurons:", hidden_size)
    print("Model Type:", model_type)
    
    # Train Neural Network.
    ABMNet = None
    if model_type == 'res_nn':
        ABMNet = train_res_nn(train_dataset, input_size=input_len, hidden_size=hidden_size, depth=depth, output_size=output_len, nEpochs=n_epochs, use_gpu=using_GPU)
    else: 
        ABMNet = train_nn(train_dataset, input_size=input_len, hidden_size=hidden_size, depth=depth, output_size=output_len, nEpochs=n_epochs, use_gpu=using_GPU)
        
    if saving_model:
        tc.save(ABMNet, 'model/' + output_name)
    # Cross Validate Using Training Dataset to Find Best Parameters. Will do later just to see how it is.
    
    
    # Validate On Test
    mse, time_to_run, predictions, tested = evaluate(ABMNet, test_dataset, use_gpu=using_GPU)
    print('Final Average MSE On Test Dataset:', mse, ', Time For Inference:', time_to_run)
    if is_transform:
        print('Final MSE Untransformed:', numpy_mse(np.matmul(predictions, np.linalg.inv(abm_dataset.transform_mat)), np.matmul(convert_dataset_output_to_numpy(test_dataset), np.linalg.inv(abm_dataset.transform_mat))))
    if normalize_out:
        scale_factor = abm_dataset.output_maxes - abm_dataset.output_mins
        # print(scale_factor)
        unnormalized_predictions = (predictions * scale_factor) + abm_dataset.output_mins
        unnormalized_actual = (tested * scale_factor) + abm_dataset.output_mins
        # np.savetxt("predictions.csv", unnormalized_predictions, delimiter=',')
        # np.savetxt("actual.csv", unnormalized_actual, delimiter=',')
        print("Unnormalized Max:", unnormalized_actual.max())
        print("Unnormalized Min:", unnormalized_actual.min())
        print('Final Average Unnormalized MSE:', numpy_mse(unnormalized_predictions, unnormalized_actual))
        print("Final Average Percent Error:", avg_percent_error(unnormalized_predictions, unnormalized_actual))
        plot_histograms(unnormalized_actual, unnormalized_predictions,output='data/graphs/' + output_name + '_og')
        
    np.savetxt('data/nn_output/' + output_name + '_predicted.csv', predictions, delimiter=',')
    np.savetxt('data/nn_output/' + output_name + '_test.csv', tested, delimiter=',')
    plot_histograms(tested, predictions, output='data/graphs/' + output_name, transform=is_transform)