import matplotlib.pyplot as plt
import numpy as np


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
    return np.sum(diff) / (y.shape[0] * y.shape[1])

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
    fig, axes = plt.subplots(figsize=(6.5, 6.0))
    x123 = np.arange(np.min(true), np.max(true))
    y123 = x123
    optimal = axes.plot(np.unique(x123), np.poly1d(np.polyfit(x123, y123, 1))(np.unique(x123)),'--')
    axes.set_xlabel("Original Model Value")
    axes.set_ylabel("Surrogate Model Prediction")
    for c in range(true.shape[1]):
        if nSpecies is not None:
            if c < nSpecies: # means
                axes.scatter(true[:,c], predictions[:,c],c='r', label='Means')
            elif c < 2*nSpecies: # variances
                axes.scatter(true[:,c], predictions[:,c],c='g', label='Variances')
            else: # covariances
                axes.scatter(true[:,c], predictions[:,c],c='b', label='Covariances')
        else:
            axes.scatter(true[:,c], predictions[:,c],c='b', label='Model Outputs')
            
        axes.legend(optimal, 'Perfect Prediction of Surrogate')
    
    plt.savefig(output + '_scatter.png')  
# assume data format n moments X 2 columns (for X and Y) 
# def plotMoments(file, title="", nSpecies=""): # get list of X, Y 
#     df = pd.read_csv(file)
#     moments = df.to_numpy()
#     if title == "":
#         title = file

#     fig, axes = plt.subplots(figsize=(6.5, 6.0))
#     axes.set_title(title, wrap=True,loc='center', fontdict = {'fontsize' : 20})    
#     plt.xlabel("Predicted Moment", fontdict = {'fontsize' : 12})
#     plt.ylabel("Observed Moment", fontdict = {'fontsize' : 12})
#     axes.spines.right.set_visible(False)
#     axes.spines.top.set_visible(False)

#     x123 = np.arange(np.min(moments[:]), np.max(moments[:]))
#     y123 = x123
#     optimalLine, = axes.plot(np.unique(x123), np.poly1d(np.polyfit(x123, y123, 1))(np.unique(x123)),'--')
#     a, b = np.polyfit(moments[:,0], moments[:,1], 1)
#     bestFit, = axes.plot(np.unique(x123), np.poly1d(np.polyfit(moments[:,0], moments[:,1], 1))(np.unique(x123)), ':')
    
#     totMoments = moments.shape[0]
#     means = ""
#     variances = ""
#     covariances = ""
#     if(nSpecies != ""):
#         nSpecies = int(nSpecies)
#         if(nSpecies*(nSpecies + 3) / 2 == totMoments):
#             means = axes.scatter(moments[:nSpecies,0],moments[:nSpecies,1], s=80)
#             variances = axes.scatter(moments[nSpecies:2*nSpecies,0], moments[nSpecies:2*nSpecies,1], s=80)
#             covariances = axes.scatter(moments[2*nSpecies:totMoments,0], moments[2*nSpecies:totMoments,1], s=80)
#             axes.legend([optimalLine, bestFit, means, variances, covariances], ["Perfect Fit",  "Best Fit of Observed vs. Predicted Line:" + " y= " + "{:.2f}".format(a) + "x + " + "{:.2f}".format(b), "Means", "Variances", "Covariances"])
#         elif(2*nSpecies == totMoments):
#             means = axes.scatter(moments[:nSpecies,0],moments[:nSpecies,1], s=80)
#             variances = axes.scatter(moments[nSpecies:2*nSpecies,0], moments[nSpecies:2*nSpecies,1], s=80)
#             axes.legend([optimalLine, bestFit, means, variances], ["Perfect Fit",  "Best Fit of Observed vs. Predicted Line:" + " y= " + "{:.2f}".format(a) + "x + " + "{:.2f}".format(b), "Means", "Variances"])      
#         else:
#             means = axes.scatter(moments[:,0],moments[:,1], s=80) 
#             axes.legend([optimalLine, bestFit, means], ["Perfect Fit",  "Best Fit of Observed vs. Predicted Line:" + " y= " + "{:.2f}".format(a) + "x + " + "{:.2f}".format(b), "Means"])      
#     else:
#         axes.scatter(moments[:,0],moments[:,1] , s=80)   
#         axes.legend([optimalLine, bestFit], ["Perfect Fit",  "Best Fit of Observed vs. Predicted Line:" + " y= " + "{:.2f}".format(a) + "x + " + "{:.2f}".format(b)])    
    
#     plt.savefig(file[:-4] + '.png')
    
    
# def plot_histograms(test_dataset, predictions, output='data/graphs/out', transform=False):
#     true = []
#     for ex in range(len(test_dataset)):
#         sample = test_dataset[ex]
#         true.append(sample['moments'].detach().numpy())
        
#     true = np.array(true)
#     fig, axes = plt.subplots(figsize=(6.5, 6.0))
#     axes.set_title("Observed vs. Predicted", wrap=True,loc='center', fontdict = {'fontsize' : 20})    
#     plt.xlabel("Observed Moment", fontdict = {'fontsize' : 12})
#     plt.ylabel("Predicted Moment", fontdict = {'fontsize' : 12})
#     x123 = np.arange(np.min(true), np.max(true[:]))
#     y123 = x123
#     optimalLine, = axes.plot(np.unique(x123), np.poly1d(np.polyfit(x123, y123, 1))(np.unique(x123)),'--')
#     # a, b = np.polyfit(moments[:,0], moments[:,1], 1)
#     # bestFit, = axes.plot(np.unique(x123), np.poly1d(np.polyfit(moments[:,0], moments[:,1], 1))(np.unique(x123)), ':')
#     for i in range(true.shape[1]):
        