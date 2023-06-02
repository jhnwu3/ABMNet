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
    r_sq = r_squared(true.flatten(), predictions.flatten())
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
    axes.legend(["R^2=" + str(r_sq)])
    plt.savefig(output + '_scatter.png')  

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
    differences = np.zeros(np.squeeze(truth[0]).shape)
    print(truth[0].shape)
    for tru, pred in zip(truth, predicted):
        differences += np.square(np.squeeze(tru) - np.squeeze(pred))
    print("hm")
    # Compute the mean of squared differences across all arrays
    mean_differences = differences / len(truth)
    plt.plot(times, mean_differences)
    plt.xlabel('Time')
    plt.ylabel('Average Mean Square Error')
    plt.title('Error Through Time')
    plt.savefig(path)