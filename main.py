import torch as tc 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
from modules.utils.cli import *
from modules.data.mixed import *
from modules.models.simple import *
from modules.utils.graph import *
from modules.utils.evaluate import *
from modules.utils.train import *
from sklearn.model_selection import KFold
 
if __name__ == '__main__':
    
    # run parameters - will convert layer into a class.
    using_GPU = use_gpu()
    n_epochs = get_epochs()
    hidden_size = get_hidden()
    csv_file = get_data()
    saving_model = save_model()
    output_name = get_output()
    depth = get_depth()
    # is_transform = transform_data()
    normalize = normalize_input()
    model_type = get_nn_type()
    normalize_out = normalize_output()
    cross = cross_validate()
    # data
    abm_dataset = ABMDataset(csv_file, root_dir="data/", standardize=normalize, norm_out=normalize_out)
    train_size = int(0.85 * len(abm_dataset))
    test_size = len(abm_dataset) - train_size
    
    # split dataset, into training set and test
    train_dataset, test_dataset = tc.utils.data.random_split(abm_dataset, [train_size, test_size])

    sample = train_dataset[0]
    input_len = sample[0].detach().numpy().shape[0]
    output_len = sample[1].detach().numpy().shape[0]
    
    print("Outputting:", output_name)
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
    batch_size = get_batch_size()
    # we do small cross validation to reduce time for nonlinear 6 protein system.
    if cross:
        kf = KFold(n_splits=3, shuffle=True, random_state=42) # seed it, shuffle it again, and n splits it.
        print(kf)
        depths_to_search = [2,4,6]
        hidden_sizes_to_search = [32,64,128] # just go up to some reasonable number I guess.
        epochs_to_search = [50, 100, 150] # number of epochs to search and train for
        batch_sizes = [None] 
        best_val_mse = np.Inf
        for batch in batch_sizes:
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
                                ABMNet = train_nn(k_train, input_size=input_len, hidden_size=h_size, depth=d_len, output_size=output_len, nEpochs=epochs, use_gpu=using_GPU, batch_size=batch)
                            mse, time_to_run, predictions, tested = evaluate(ABMNet, k_test, use_gpu=using_GPU, batch_size=batch)
                            print(repr(f"Fold {fold}, Val_MSE: {mse}"))
                            total_val_mse += mse
                            print(repr(f"{d_len} {h_size} {epochs}"))
                            
                        # search for best combo of hyperparams
                        if total_val_mse < best_val_mse:
                            batch_size = batch
                            best_val_mse = total_val_mse
                            best_depth = d_len 
                            best_hidden_size = h_size
                            best_n_epochs = epochs
                            print("Found New Best Epochs:", best_n_epochs, "New Best Depth:", best_depth, " New Best Hidden Size:", best_hidden_size, " with mse:", best_val_mse)
                
        # print(len(tc.utils.data.Subset(train_dataset, train_index)))
        
    
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
    # batch_size = 500
    # now train using whatever cross-validated or user-specified 
    if model_type == 'res_nn':
        ABMNet = train_res_nn(train_dataset, input_size=input_len, hidden_size=best_hidden_size, depth=best_depth, output_size=output_len, nEpochs=best_n_epochs, use_gpu=using_GPU, batch_size=batch_size)
    else: 
        ABMNet = train_nn(train_dataset, input_size=input_len, hidden_size=best_hidden_size, depth=best_depth, output_size=output_len, nEpochs=best_n_epochs, use_gpu=using_GPU, batch_size=batch_size)
        
    if saving_model:
        tc.save(ABMNet, 'model/' + output_name + '.pt')
   

    # Validate On Test,
    mse, time_to_run, predictions, tested = evaluate(ABMNet, test_dataset, use_gpu=using_GPU, batch_size=batch_size)
    print('Final MSE On Test Dataset:', mse, ', Time For Inference:', time_to_run)
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
        # print("Final Average Percent Error:", avg_percent_error(unnormalized_predictions, unnormalized_actual)) This computation is incorrect and doesn't make any sense at the moment.
        plot_histograms(unnormalized_actual, unnormalized_predictions,output='graphs/histograms/' + output_name + '_og')
        plot_scatter(unnormalized_actual, unnormalized_predictions, output='graphs/scatter/' + output_name +'_og')
        np.savetxt('data/nn_output/' + output_name + '_predicted_og.csv', unnormalized_predictions, delimiter=',')
        np.savetxt('data/nn_output/' + output_name + '_test_og.csv', unnormalized_actual, delimiter=',')
        
    np.savetxt('data/nn_output/' + output_name + '_predicted.csv', predictions, delimiter=',')
    np.savetxt('data/nn_output/' + output_name + '_test.csv', tested, delimiter=',')
    plot_histograms_subplots(tested, predictions, output='graphs/histograms/' + output_name)
    
    nSpecies = get_n_species()
    if nSpecies < 1:
        nSpecies = None
    else:
        nSpecies = int(nSpecies)    
    plot_scatter(tested, predictions, output='graphs/scatter/' + output_name, nSpecies=nSpecies)

# full train on entire dataset and evaluate for "maximal" performance on actual parameter estimation task
ABMNet =  train_nn(abm_dataset, input_size=input_len, hidden_size=best_hidden_size, depth=best_depth, output_size=output_len, nEpochs=best_n_epochs, use_gpu=using_GPU, batch_size=batch_size)
mse, time_to_run, predictions, tested = evaluate(ABMNet, abm_dataset, use_gpu=True, batch_size=batch_size)
print('Final Average MSE On Whole Dataset:', mse, ', Time For Inference:', time_to_run)
plot_scatter(tested, predictions, output='graphs/scatter/' + output_name +"_full", nSpecies=nSpecies)

if saving_model:
    tc.save(ABMNet, 'model/' + output_name + '_full.pt')
