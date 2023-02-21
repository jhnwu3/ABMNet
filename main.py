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
from sklearn.model_selection import KFold


def get_depth():
    if '-d' in sys.argv:
        return int(sys.argv[sys.argv.index('-d') + 1])
    return -1

def get_data(): 
    return sys.argv[sys.argv.index('-i') + 1]

def get_epochs():
    if '--epochs' in sys.argv:
        return int(sys.argv[sys.argv.index('--epochs') + 1])
    return -1

def get_output(): 
    return sys.argv[sys.argv.index('-o') + 1]

def get_hidden():
    if '-h' in sys.argv:
        return int(sys.argv[sys.argv.index('-h') + 1])
    return -1

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

def cross_validate():
    return '--cross' in sys.argv
 
def get_nn_type():
    if '--type' in sys.argv:
        return sys.argv[sys.argv.index('--type') + 1]
    else: 
        return -1 
 
if __name__ == '__main__':
    
    # run parameters - will convert layer into a class.
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
    cross = cross_validate()
    # data
    abm_dataset = ABMDataset(csv_file, root_dir="data/", transform=is_transform, standardize=normalize, norm_out=normalize_out)
    train_size = int(0.8 * len(abm_dataset))
    test_size = len(abm_dataset) - train_size
    
    # split dataset, into training set and test
    train_dataset, test_dataset = tc.utils.data.random_split(abm_dataset, [train_size, test_size])

    sample = train_dataset[0]
    input_len = sample[0].detach().numpy().shape[0]
    output_len = sample[1].detach().numpy().shape[0]
    
    print("Dataset:", csv_file)
    print("Length of Training:",train_size)
    print("Length of Test:", test_size)
    print("Input Dimension:", input_len)
    print("Output Dimension:", output_len)
    print("Model Type:", model_type)
    
    # Train Neural Network. Cross Validation, also maybe, repeat training process 30 times, and save best with least test
    # 5-fold cross validation so we can tune our parameters effectively. In this case, how many epochs to use in training overall, then full-train using that number.
    # Find best number out of 100 epochs.
    ABMNet = None
    best_n_epochs = 0
    best_hidden_size = 0
    best_depth = 0
    if cross:
        kf = KFold(n_splits=5, shuffle=True,random_state=42) # seed it, shuffle it again, and n splits it.
        print(kf)
        depths_to_search = [4,6,8,10]
        hidden_sizes_to_search = [32,64,128,267] # just go up to some reasonable number I guess.
        epochs_to_search = [25, 50, 100, 150] # number of epochs to search and train for
        best_val_mse = np.Inf
        for d_len in depths_to_search:
            for h_size in hidden_sizes_to_search:
                for epochs in epochs_to_search:
                    total_val_mse = 0
                    print("--Scanning Number of Epochs:", epochs)
                    for fold, (train_index, test_index) in enumerate(kf.split(train_dataset)):
                        k_train = tc.utils.data.Subset(train_dataset, train_index)
                        k_test = tc.utils.data.Subset(train_dataset, test_index)
                        if model_type == 'res_nn':
                            ABMNet = train_res_nn(k_train, input_size=input_len, hidden_size=h_size, depth=d_len, output_size=output_len, nEpochs=epochs, use_gpu=using_GPU)
                        else: 
                            ABMNet = train_nn(k_train, input_size=input_len, hidden_size=h_size, depth=d_len, output_size=output_len, nEpochs=epochs, use_gpu=using_GPU)
                        mse, time_to_run, predictions, tested = evaluate(ABMNet, test_dataset, use_gpu=using_GPU)
                        print(repr(f"Fold {fold}, Val_MSE: {mse}"))
                        total_val_mse += mse
                        print(repr(f"{best_depth} {best_hidden_size} {best_n_epochs}"))
                        
                    # search for best combo of hyperparams
                    if total_val_mse < best_val_mse:
                        best_val_mse = total_val_mse
                        best_depth = d_len 
                        best_hidden_size = h_size
                        best_n_epochs = epochs
                        print("Found New Best Epochs:", best_n_epochs, "New Best Depth:", best_depth, " New Best Hidden Size:", best_hidden_size, " with mse:", best_val_mse)
            
        # print(len(tc.utils.data.Subset(train_dataset, train_index)))
        
    # exit(0)
    # ABMNet = None
    
    # in case user specifies the number of epochs, and they feel like it.
    if n_epochs > 0:
        best_n_epochs = n_epochs
    
    if hidden_size > 0:
        best_hidden_size = hidden_size
    
    if depth > 0:
        best_depth = depth
    
    print("----- Hyperparameters used for final training -----")
    print("Depth of NN:", best_depth)
    print("Hidden Neurons:", best_hidden_size)
    print("# Epochs Used:", best_n_epochs)
    # now train using whatever cross-validated or user-specified 
    if model_type == 'res_nn':
        ABMNet = train_res_nn(train_dataset, input_size=input_len, hidden_size=best_hidden_size, depth=best_depth, output_size=output_len, nEpochs=best_n_epochs, use_gpu=using_GPU)
    else: 
        ABMNet = train_nn(train_dataset, input_size=input_len, hidden_size=best_hidden_size, depth=best_depth, output_size=output_len, nEpochs=best_n_epochs, use_gpu=using_GPU)
        
    if saving_model:
        tc.save(ABMNet, 'model/' + output_name + '.pt')
   
    
    
    
    
    
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
        plot_histograms(unnormalized_actual, unnormalized_predictions,output='graphs/histograms/' + output_name + '_og')
        plot_scatter(unnormalized_actual, unnormalized_predictions, output='graphs/scatter/' + output_name +'_og')
        np.savetxt('data/nn_output/' + output_name + '_predicted_og.csv', unnormalized_predictions, delimiter=',')
        np.savetxt('data/nn_output/' + output_name + '_test_og.csv', unnormalized_actual, delimiter=',')
        
    np.savetxt('data/nn_output/' + output_name + '_predicted.csv', predictions, delimiter=',')
    np.savetxt('data/nn_output/' + output_name + '_test.csv', tested, delimiter=',')
    plot_histograms(tested, predictions, output='graphs/histograms/' + output_name, transform=is_transform)
    plot_scatter(tested, predictions, output='graphs/scatter/' + output_name)