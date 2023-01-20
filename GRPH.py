import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(test_dataset, predictions, output='data/graphs/out'):
    binwidth = 2
    true = []
    for ex in range(len(test_dataset)):
        sample = test_dataset[ex]
        true.append(sample['moments'].detach().numpy())
    
    true = np.array(true)
    # column wise graphing
    for i in range(true.shape[1]):
        plt.figure()
        plt.hist(true[:,i],bins=range(int(min(true[:,i])), int(max(true[:,i])) + binwidth , binwidth), label='Ground Truth')
        plt.hist(predictions[:,i],bins=range(int(min(predictions[:,i])), int(max(predictions[:,i])) + binwidth , binwidth), label='Surrogate Model')
        plt.legend(loc='upper right')
        plt.savefig(output + str(i) +'.png')