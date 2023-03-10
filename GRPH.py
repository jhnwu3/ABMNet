import matplotlib.pyplot as plt
import numpy as np
import scipy 

def convert_dataset_output_to_numpy(dataset):
    npy = []
    for ex in range(len(dataset)):
        sample = dataset[ex]
        npy.append(sample['moments'].detach().numpy())
    return np.array(npy)


# errors assume matrices.
def numpy_mse(x,y):
    return np.sum((np.square(x - y)).mean(axis=0)) / x.shape[0] # get average mse

def avg_percent_error(x,y):
    diff = x-y
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j] != 0:
                diff[i,j] /= y[i,j]
    return np.sum(np.abs(diff)) / (y.shape[0] * y.shape[1])

def plot_histograms(test_dataset, predictions, output='data/graphs/out', transform=False):
    binwidth = 2    
    true = test_dataset
    # column wise graphing
    for i in range(true.shape[1]):
        binwidth = (np.max(true[:,i]) - np.min(true[:,i]) )/ 20.0

        if binwidth == 0:
            binwidth = 1
        fig = plt.figure(figsize=(8.0, 8.0))
        ax = fig.add_subplot(111)
        # print(true.shape) range(int(min(true[:,i])), int(max(true[:,i])) + 2*binwidth, binwidth)
        ax.hist(true[:,i],bins=bins_list(true[:,i].min(), true[:,i].max(), binwidth), label='Ground Truth')
        ax.hist(predictions[:,i],bins=bins_list(true[:,i].min(), true[:,i].max(), binwidth), label='Surrogate Model')
        ax.legend(loc='upper right')
        ax.set_xlabel("Value of Model Output")
        ax.set_ylabel("Number of Parameter Sets")
        plt.savefig(output + str(i) +'.png')
    

    # for i in range(true.shape[1]):
    #     binwidth = (np.max(true[:,i]) - np.min(true[:,i]) )/ 10.0

    #     if binwidth == 0:
    #         binwidth = 1
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     # print(true.shape) range(int(min(true[:,i])), int(max(true[:,i])) + 2*binwidth, binwidth)
    #     ax.hist(true[:,i],bins=bins_list(true[:,i].min(), true[:,i].max(), binwidth), label='Ground Truth')
    #     ax.legend(loc='upper right')
    #     plt.savefig(output +'_tru_' + str(i) +'.png')
        
    # for i in range(true.shape[1]):
    #     binwidth = (np.max(predictions[:,i]) - np.min(predictions[:,i]) )/ 10.0

    #     if binwidth == 0:
    #         binwidth = 1
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     # print(true.shape) range(int(min(true[:,i])), int(max(true[:,i])) + 2*binwidth, binwidth)
    #     ax.hist(predictions[:,i],bins=bins_list(predictions[:,i].min(), predictions[:,i].max(), binwidth), label='Surrogate')
    #     ax.legend(loc='upper right')
    #     plt.savefig(output +'_sg_' + str(i) +'.png')
    

def bins_list(min, max, step_size):
    binss= []
    current = min
    binss.append(current)
    while current < max:
        current += step_size
        binss.append(current)
    return binss



def plot_scatter(true, predictions, output='data/graphs/out', nSpecies=None):
    plt.figure()
    fig, axes = plt.subplots(figsize=(8, 8))
    x123 = np.arange(np.min(true), np.max(true))
    
    if x123[0] == 0:
        x123 = np.append(x123,[1])
    y123 = x123
    optimal = axes.plot(np.unique(x123), np.poly1d(np.polyfit(x123, y123, 1))(np.unique(x123)),'--', c='k', label='Perfect Prediction')
    axes.set_xlabel("Original Model Value")
    axes.set_ylabel("Surrogate Model Prediction")
    
    if nSpecies is not None:
        for c in range(true.shape[1]):
            if c < nSpecies: # means
                axes.scatter(true[:,c], predictions[:,c],c='r', label='Means')
            elif c < 2*nSpecies: # variances
                axes.scatter(true[:,c], predictions[:,c],c='g', label='Variances')
            else: # covariances
                axes.scatter(true[:,c], predictions[:,c],c='b', label='Covariances')
    else:
        axes.scatter(true.flatten(), predictions.flatten(), c='c')
            
        # axes.legend(optimal, 'Perfect Prediction')
    axes.legend(loc='upper right')
    plt.savefig(output + '_scatter.png')  

        