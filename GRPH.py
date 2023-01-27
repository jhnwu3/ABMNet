import matplotlib.pyplot as plt
import numpy as np


def convert_dataset_output_to_numpy(dataset):
    npy = []
    for ex in range(len(dataset)):
        sample = dataset[ex]
        npy.append(sample['moments'].detach().numpy())
    return np.array(npy)

def numpy_mse(x,y):
    return np.sum((np.square(x - y)).mean(axis=0))

def plot_histograms(test_dataset, predictions, output='data/graphs/out', transform=False):
    binwidth = 2
    true = []
    for ex in range(len(test_dataset)):
        sample = test_dataset[ex]
        true.append(sample['moments'].detach().numpy())
    
    true = np.array(true)
    # column wise graphing
    for i in range(true.shape[1]):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # print(true.shape)
        ax.hist(true[:,i],bins=range(int(min(true[:,i])), int(max(true[:,i])) + 2*binwidth, binwidth), label='Ground Truth')
        ax.hist(predictions[:,i],bins=range(int(min(predictions[:,i])), int(max(predictions[:,i])) + 2*binwidth, binwidth), label='Surrogate Model')
        ax.legend(loc='upper right')
        plt.savefig(output + str(i) +'.png')
        
        
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