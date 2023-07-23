import matplotlib.pyplot as plt
import numpy as np
import scipy 
import networkx as nx
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import to_networkx
from modules.data.spatial import SpatialObj
def convert_dataset_output_to_numpy(dataset):
    npy = []
    for ex in range(len(dataset)):
        sample = dataset[ex]
        npy.append(sample['moments'].detach().numpy())
    return np.array(npy)

def r_squared(y_true, y_pred):
    # Calculate the mean of the true values
    mean_true = np.mean(y_true)

    # Calculate the total sum of squares (SS_tot)
    ss_tot = np.sum((y_true - mean_true)**2)

    # Calculate the residual sum of squares (SS_res)
    ss_res = np.sum((y_true - y_pred)**2)

    # Calculate R-squared (coefficient of determination)
    r2 = 1 - (ss_res / ss_tot)

    return r2

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

def plot_histograms_subplots(test_dataset, predictions, output='data/graphs/out', transform=False):
    binwidth = 2
    true = test_dataset
    num_plots = true.shape[1]
    plots_per_row = min(num_plots, 4)
    num_rows = int(np.ceil(num_plots / plots_per_row))
    fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(5*plots_per_row, 5*num_rows))

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_plots):
        row_idx = i // plots_per_row
        col_idx = i % plots_per_row
        ax = axes[row_idx, col_idx]

        binwidth = (np.max(true[:, i]) - np.min(true[:, i])) / 20.0
        if binwidth == 0:
            binwidth = 1

        ax.hist(true[:, i], bins=bins_list(true[:, i].min(), true[:, i].max(), binwidth),histtype='step',label='Ground Truth')
        ax.hist(predictions[:, i], bins=bins_list(true[:, i].min(), true[:, i].max(), binwidth), alpha=0.5, histtype='step', label='Surrogate Model')
        ax.legend(loc='upper right')
        ax.set_xlabel("Value of Model Output")
        ax.set_ylabel("Number of Parameter Sets")

    # Hide empty subplots
    for i in range(num_plots, num_rows * plots_per_row):
        row_idx = i // plots_per_row
        col_idx = i % plots_per_row
        fig.delaxes(axes[row_idx, col_idx])

    plt.tight_layout()
    plt.savefig(output + '.png')

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

#Primary: #1f77b4 (Blue)
# Secondary: #ff7f0e (Orange)
# Neutral: #808080 (Gray)


def plot_scatter(true, predictions, output='data/graphs/out', nSpecies=None, show=False):
    plt.figure()
    fig, axes = plt.subplots(figsize=(8, 8))
    x123 = np.arange(np.min(true), np.max(true))
    # print(np.min(true))
    # print(np.max(true))
    # # print(true)
    # print(x123)
    size = 2
    if x123.shape[0] < 2:
        x123 = np.append(x123,[0])
        x123 = np.append(x123,[1])
    elif x123[0] == 0:
        x123 = np.append(x123,[1])
        
    y123 = x123
    
    # Sort x123 and y123 in increasing order if necessary
    sort_indices = np.argsort(x123)
    x123 = x123[sort_indices]
    y123 = y123[sort_indices]
    print("Perfect Line Plotting With X and Y respectively:")
    print(x123)
    print(y123)
    categorical = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    optimal = axes.plot(np.unique(x123), np.poly1d(np.polyfit(x123, y123, 1))(np.unique(x123)),'--', c='k', label='Perfect Prediction')
    axes.set_xlabel("Original Model Value")
    axes.set_ylabel("Surrogate Model Prediction")
    r_sq = r_squared(true.flatten(), predictions.flatten())
    if nSpecies is not None:
        if true.shape[1] > 2*nSpecies:
            axes.scatter(true[:,:nSpecies], predictions[:,:nSpecies], c=categorical[0], label='Means', s=size)
            axes.scatter(true[:,nSpecies:2*nSpecies], predictions[:,nSpecies:2*nSpecies], c=categorical[1], label='Variances', s=size)
            axes.scatter(true[:,2*nSpecies:], predictions[:,2*nSpecies:], c=categorical[2], label='Covariances', s=size)
        elif true.shape[1] > nSpecies and true.shape[1] < 2*nSpecies + 1:
            axes.scatter(true[:,:nSpecies], predictions[:,:nSpecies], c=categorical[0], label='Means', s=size)
            axes.scatter(true[:,nSpecies:2*nSpecies], predictions[:,nSpecies:2*nSpecies], c=categorical[1], label='Variances', s=size)
        else: 
            axes.scatter(true[:,:nSpecies], predictions[:,:nSpecies], c=categorical[0], label='Means', s=size)
    else:
        axes.scatter(true.flatten(), predictions.flatten(), c='c', s=size)
            
        # axes.legend(optimal, 'Perfect Prediction')
    axes.legend(loc='upper right')
    axes.set_title("R^2=" + str(r_sq))
    axes.legend()
    plt.savefig(output + '_scatter.png')  
    if show: 
        plt.show()
    plt.close()

def visualize_graph(G, color, path="data/spatial/graph.png"):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                    node_color=color, cmap="Set2")
    plt.savefig(path)
    # plt.show()

# graph is a tensor array of where each row are nodes
# edges
#Pixel, Cytotoxic CD8+ T Cells, Cancer, Exhausted CD8+ T Cells, Dead Cancer Cells, Ignore, Ignore, TAMs, Ignore
def plot_giuseppe_graphs(graph, edges, path=""):
    data = Data(x=graph, edge_index=edges)
    networkG = to_networkx(data, to_undirected=True) 
    print(data.x.size())
    for i in range(data.x.size()[1]):
        visualize_graph(networkG, color = data.x[:,i], path=path + str(i) + ".png") # get the cancer cells.  
    
    

def plot_graph_to_img(graph, path =""):
    data = graph.numpy()
    img = SpatialObj.translate_to_img(data, width=int(np.sqrt(data.shape[0])), features=int((data.shape[1])) , offset=0)
    plt.figure()
    
    nCols = 4
    nOut = img.shape[2]
    nRows = int(nOut / nCols)
    if nOut % nCols != 0:
        nRows+=1

    f, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(20,20))
    channel = 0
    
    for r in range(nRows):
        for c in range(nCols):
            if channel < img.shape[2]:
                axs[r,c].imshow(img[:,:,channel])
                axs[r,c].set_title(str(channel))
            channel+=1
            
    plt.savefig(path)        


def plot_time_series_errors(truth, predicted, times, path="graphs/temporal/errors.png"):
    # Calculate element-wise square differences
    print("plotting time errors")
    differences = np.zeros(np.squeeze(truth)[0].shape)
    # print(truth.shape)
    # print(differences.shape)
    # print(predicted.shape)
    diffs = []
    for tru, pred in zip(truth, predicted):
        differences += np.square(np.squeeze(tru) - np.squeeze(pred))
        diffs.append(np.square(np.squeeze(tru) - np.squeeze(pred)))
    # print("hm")
    diffs = np.array(diffs)
    # Compute the mean of squared differences across all arrays
    mean_differences = differences / len(truth)
    variances = np.var(diffs,axis=0)
    categorical = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    fig = plt.figure()
    plt.plot(mean_differences[:,1], c=categorical[0], label="Mean Square Error")
    plt.plot(variances[:,1], c=categorical[1], label="Variances of Square Error")
    plt.xlabel('Time')
    plt.ylabel('Square Error')
    plt.title('Average Error Through Time Across All Parameter Sets')
    plt.legend()
    plt.savefig(path)
    plt.close(fig)

def plot_time_series(truth, predicted, times, path="graphs/temporal/trajectory.png"):
    fig = plt.figure()
    categorical = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    plt.plot(times, truth,c=categorical[0], label="Truth")
    plt.plot(times, predicted,c=categorical[1], label="Predicted")
    plt.xlabel('Time')
    plt.grid(True)
    plt.ylabel('Abundance')
    plt.legend()
    plt.savefig(path)
    plt.close(fig)