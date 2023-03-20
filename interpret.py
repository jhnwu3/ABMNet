import torch as tc 
import time
from torch import nn
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from ABM import *
from GRPH import *
# 
class DumbInterpreter:
    def __init__(self, modelPath, dataset=None, normalize_out=False, standardize_in=False):
        self.modelPath = modelPath
        self.model = tc.load(modelPath)        
        self.dataset = dataset
        self.norm_out = normalize_out
        self.standardize = standardize_in
    
    def plot_with_ground_truth(self, plotPath, groundTruthPath, thetaStar, nCols):
        # load input data
        groundTruth = pd.read_csv(groundTruthPath)
        # load based on input size
        groundTruth = groundTruth.to_numpy()
        inputs = groundTruth[:,:self.model.input_size].copy()
        trueOutputs = groundTruth[:,self.model.input_size:].copy()
        if self.standardize:
            stds = self.dataset.input_stds
            means = self.dataset.means
            inputs = (inputs - means) / stds
            print("Standardized Inputs:")
            print(inputs)

        # go through all inputs and outputs
        outputs = []
        for i in range(inputs.shape[0]):
            input = tc.from_numpy(inputs[i])
            if next(self.model.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            output = self.model(input).cpu().detach().numpy()
            outputs.append(output)

        outputs = np.array(outputs)
        
        if self.norm_out:
            scale_factor = self.dataset.output_maxes - self.dataset.output_mins
            outputs = (outputs * scale_factor) + self.dataset.output_mins
            
        # plot them in conjunction
        nOut = self.model.output_size
        nRows = int(nOut / nCols)
        if nOut % nCols != 0:
            nRows+=1

        f, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(20,20))
        moment = 0

        for r in range(nRows):
            for c in range(nCols):
                if moment < nOut:
                    # print(outputs[:,moment])
                    axs[r,c].plot(inputs[:,thetaStar], outputs[:,moment], label='Surrogate')
                    axs[r,c].plot(inputs[:,thetaStar], trueOutputs[:,moment], label='Ground Truth')
                    axs[r,c].set_title('o' + str(moment))
                    axs[r,c].set_xlim([0,1])
                    axs[r,c].set_ylim([0,1.5*np.max(trueOutputs[:,moment])])
                    
                    moment+=1
                    axs[r,c].legend(loc='upper right')
        
        plt.xlabel("Theta" + str(thetaStar + 1))
        plt.ylabel("Output Value")
        # filename = "int_theta" + str(thetaStar + 1) + "_nl6"
        plt.savefig(plotPath + "_theta" + str(thetaStar + 1) + '.png')

        
    
    def plot(self, path, thetaStar, thetaFixed, nCols, nSteps):
        # load model that was saved.
        nIn = self.model.input_size
        nOut = self.model.output_size

        # create some data
        nSteps = 10
        thetas = np.zeros(nIn) + thetaFixed 
        thetas[thetaStar] = 0
        print("Interpreting Input Index:", thetaStar + 1, " from Model Path:", self.modelPath)
        # pick a rate constant to interpret
        # run model on step sizes of rate constants
        changing_inputs = []
        outputs = []
        for i in range(nSteps):
            thetas[thetaStar] += (1.0 / nSteps)
            changing_inputs.append(thetas[thetaStar])
            thetas_cp = thetas
            if self.standardize: 
                thetas_cp = (thetas - self.dataset.input_means) / self.dataset.input_stds 
                
            input = tc.from_numpy(thetas_cp)
            if next(self.model.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            output = self.model(input).cpu().detach().numpy()
            outputs.append(output)
            

        changing_inputs = np.array(changing_inputs)
        outputs = np.array(outputs)

        # plot change in outputs of mean counts (because that's easier to interpret)
        # subplotting 

        nRows = int(nOut / nCols)
        if nOut % nCols != 0:
            nRows+=1

        f, axs = plt.subplots(nrows=nRows, ncols=nCols, figsize=(20,20))
        moment = 0

        for r in range(nRows):
            for c in range(nCols):
                if moment < nOut:
                    axs[r,c].plot(changing_inputs, outputs[:,moment], label='Surrogate')
                    axs[r,c].set_title('o' + str(moment))
                    axs[r,c].set_xlim([0,1])
                    axs[r,c].set_ylim([0,1.5*np.max(outputs[:,moment])])
                    moment+=1
             

        plt.xlabel("Theta" + str(thetaStar + 1))
        plt.ylabel("Output Value")
        # filename = "int_theta" + str(thetaStar + 1) + "_nl6"
        plt.savefig(path +"_theta" + str(thetaStar + 1) + '.png')


    def plot_contour(self, path, nCols, resolution, groundTruthTheta, y, levels=20):
        pairedInput = 1
        # heldThetas = 0.5 * np.ones(self.model.input_size)
        # math for nRows 
        nCombos = self.model.input_size
        nRows = int(nCombos/nCols)
        if nCombos % nCols > 0: 
            nRows = int(nCombos / nCols) + 1
              
        fig, ax = plt.subplots(nRows, nCols, constrained_layout=True, figsize=(20,20))
        
        plotRow = 0
        plotCol = 0
        for inputIdx in range(self.model.input_size):
            theta = groundTruthTheta.copy()
            theta[inputIdx] = 0
            xCoords = np.zeros((resolution, resolution))
            yCoords = np.zeros((resolution, resolution))
            heatMap = np.zeros((resolution, resolution))
            for i in range(resolution):
                theta[inputIdx] += 1.0 / resolution
                theta[pairedInput] = 0
                for j in range(resolution):
                    theta[pairedInput] += 1.0 / resolution 
                    xCoords[i,j] = theta[inputIdx] 
                    yCoords[i,j] = theta[pairedInput]
                    thetaCopy = theta
                    if self.standardize:
                        thetaCopy = (theta - self.dataset.input_means) / self.dataset.input_stds
                        # print(theta_copy)
                    input = tc.from_numpy(thetaCopy)
                    if next(self.model.parameters()).is_cuda:
                        input = input.to(tc.device("cuda"))
                    output = self.model(input).cpu().detach().numpy()
                    if self.norm_out:
                        scale_factor = self.dataset.output_maxes - self.dataset.output_mins
                        output = (output * scale_factor) + self.dataset.output_mins
                    heatMap[i,j] = (numpy_mse(output, y)) # mse cost
            
            # print(xCoords,ax)  
            print(plotRow,",", plotCol)
            print("pairedInput:", pairedInput)
            cont = ax[plotRow, plotCol].contourf(xCoords, yCoords, heatMap, cmap="plasma", levels=levels)
            ax[plotRow, plotCol].set_xlabel("Theta " + str(inputIdx + 1))
            ax[plotRow, plotCol].set_ylabel("Theta " + str(pairedInput + 1))
            ax[plotRow, plotCol].scatter(groundTruthTheta[inputIdx], groundTruthTheta[pairedInput], s=100, c='g',marker="x", label="True Theta")
            plt.colorbar(cont, ax=ax[plotRow, plotCol])     
            # plt.imshow(heatMap)
           
            # exit(0) 
            if plotCol < nCols - 1: # iterate columnwise first, then rowWise
                plotCol+=1
            elif plotRow < nRows: 
                plotCol = 0
                plotRow+=1 
                
            if pairedInput < self.model.input_size - 1:
                pairedInput+=1
            else: 
                pairedInput =0
        
        plt.savefig(path) 
        
    def plot_gmm_contour(self, path, nCols, resolution, groundTruthTheta, y, wt, levels=20):
        pairedInput = 1
        # heldThetas = 0.5 * np.ones(self.model.input_size)
        # math for nRows 
        nCombos = self.model.input_size
        nRows = int(nCombos/nCols)
        if nCombos % nCols > 0: 
            nRows = int(nCombos / nCols) + 1
              
        fig, ax = plt.subplots(nRows, nCols, constrained_layout=True, figsize=(20,20))
        
        plotRow = 0
        plotCol = 0
        for inputIdx in range(self.model.input_size):
            theta = groundTruthTheta.copy()
            theta[inputIdx] = 0
            xCoords = np.zeros((resolution, resolution))
            yCoords = np.zeros((resolution, resolution))
            heatMap = np.zeros((resolution, resolution))
            for i in range(resolution):
                theta[inputIdx] += 1.0 / resolution
                theta[pairedInput] = 0
                for j in range(resolution):
                    theta[pairedInput] += 1.0 / resolution 
                    xCoords[i,j] = theta[inputIdx] 
                    yCoords[i,j] = theta[pairedInput]
                    thetaCopy = theta
                    if self.standardize:
                        thetaCopy = (theta - self.dataset.input_means) / self.dataset.input_stds
                   
                    input = tc.from_numpy(thetaCopy)
                    if next(self.model.parameters()).is_cuda:
                        input = input.to(tc.device("cuda"))
                    output = self.model(input).cpu().detach().numpy()
                    if self.norm_out:
                        scale_factor = self.dataset.output_maxes - self.dataset.output_mins
                        output = (output * scale_factor) + self.dataset.output_mins
                    heatMap[i,j] = np.matmul(output-y, np.matmul((output - y).transpose(), wt)) # mse cost
            
            # print(xCoords,ax)  
            print(plotRow,",", plotCol)
            print("pairedInput:", pairedInput)
            cont = ax[plotRow, plotCol].contourf(xCoords, yCoords, heatMap, cmap="plasma", levels=levels)
            ax[plotRow, plotCol].set_xlabel("Theta " + str(inputIdx + 1))
            ax[plotRow, plotCol].set_ylabel("Theta " + str(pairedInput + 1))
            ax[plotRow, plotCol].scatter(groundTruthTheta[inputIdx], groundTruthTheta[pairedInput], s=100, c='g',marker="x", label="True Theta")
            plt.colorbar(cont, ax=ax[plotRow, plotCol])     
            # plt.imshow(heatMap)
           
            # exit(0) 
            if plotCol < nCols - 1: # iterate columnwise first, then rowWise
                plotCol+=1
            elif plotRow < nRows: 
                plotCol = 0
                plotRow+=1 
                
            if pairedInput < self.model.input_size - 1:
                pairedInput+=1
            else: 
                pairedInput =0
        
        plt.savefig(path)
        
        
    # return MSE metric of moments
    def evaluate(self, dataset, use_gpu = False):
        
        criterion = nn.MSELoss()
        if tc.cuda.is_available() and use_gpu:
            device = tc.device("cuda")
            self.model = self.model.cuda()
            criterion = criterion.cuda()
            using_gpu = True
        else:
            device = tc.device("cpu")
            using_gpu = False

        print(f"Using GPU: {using_gpu}")
        self.model.eval()
        loss = 0
        start_time = time.time()
        predicted = []
        tested = []
        for ex in range(len(dataset)):
            sample = dataset[ex]
            input = sample[0]
            output = sample[1]
            prediction = self.model.forward(input.to(device))
            loss += criterion(prediction.squeeze(), output.squeeze().to(device))
            tested.append(output.cpu().detach().numpy())
            predicted.append(prediction.cpu().detach().numpy())
            
        return loss.cpu().detach().numpy() / len(dataset), time.time() - start_time, np.array(predicted), np.array(tested)
    
    def plot_scatter(self, true, predictions, output='data/graphs/out', nSpecies=None):
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
        
        
        
        
class MultiInterpreter:
    def __init__(self, models, dataset=None, normalize_out=False, standard_in = False):
        self.models = models      
        self.dataset = dataset
        self.norm_out = normalize_out
        self.standardize_in = standard_in
        
    def multi_gmm_cost(self, x, y, wts):
        costs = []
        cost = 0
        w = 0
        for surrogate in self.models:
            wt = wts[w]
            input = tc.from_numpy(x)
            if next(surrogate.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            output = surrogate(input).cpu().detach().numpy()
            cost += np.matmul(output-y[w], np.matmul((output - y[w]).transpose(), wt))
            w+=1
        costs = cost
        
        # print(costs)
        # print("output:",output.shape)
        # print("wt:", wt.shape)
        # print("y:", y.shape)
        return costs
    
    def plot_mgmm_contour(self, path, nCols, resolution, groundTruthTheta, y, wts, levels=20):
        pairedInput = 1
        # heldThetas = 0.5 * np.ones(self.model.input_size)
        # math for nRows 
        nCombos = self.models[0].input_size
        nRows = int(nCombos/nCols)
        if nCombos % nCols > 0: 
            nRows = int(nCombos / nCols) + 1
              
        fig, ax = plt.subplots(nRows, nCols, constrained_layout=True, figsize=(20,20))
        
        plotRow = 0
        plotCol = 0
        for inputIdx in range(self.models[0].input_size):
            theta = groundTruthTheta.copy()
            theta[inputIdx] = 0
            xCoords = np.zeros((resolution, resolution))
            yCoords = np.zeros((resolution, resolution))
            heatMap = np.zeros((resolution, resolution))
            for i in range(resolution):
                theta[inputIdx] += 1.0 / resolution
                theta[pairedInput] = 0
                for j in range(resolution):
                    theta[pairedInput] += 1.0 / resolution 
                    thetaCp = theta
                    if self.standardize_in:
                        thetaCp = (theta - self.dataset.input_means) / self.dataset.input_stds
                    xCoords[i,j] = theta[inputIdx] 
                    yCoords[i,j] = theta[pairedInput]
                    # input = tc.from_numpy(theta)
                    # if next(self.models[0].parameters()).is_cuda:
                    #     input = input.to(tc.device("cuda"))
                    # if self.norm_out:
                    #     scale_factor = self.dataset.output_maxes - self.dataset.output_mins
                    #     output = (output * scale_factor) + self.dataset.output_mins
                    heatMap[i,j] = self.multi_gmm_cost(thetaCp,y=y,wts=wts) # gmm cost
            
            # print(xCoords,ax)  
            print(plotRow,",", plotCol)
            print("pairedInput:", pairedInput)
            cont = ax[plotRow, plotCol].contourf(xCoords, yCoords, heatMap, cmap="plasma", levels=levels)
            ax[plotRow, plotCol].set_xlabel("Theta " + str(inputIdx + 1))
            ax[plotRow, plotCol].set_ylabel("Theta " + str(pairedInput + 1))
            ax[plotRow, plotCol].scatter(groundTruthTheta[inputIdx], groundTruthTheta[pairedInput], s=100, c='g',marker="x", label="True Theta")
            plt.colorbar(cont, ax=ax[plotRow, plotCol])     
            # plt.imshow(heatMap)
           
            # exit(0) 
            if plotCol < nCols - 1: # iterate columnwise first, then rowWise
                plotCol+=1
            elif plotRow < nRows: 
                plotCol = 0
                plotRow+=1 
                
            if pairedInput < self.models[0].input_size - 1:
                pairedInput+=1
            else: 
                pairedInput =0
        
        plt.savefig(path)


if __name__ == "__main__":
    # nl6dataset = ABMDataset("data/static/NL6P_t05.csv", root_dir="data/", standardize=False, norm_out=True)
    # nl6Int= DumbInterpreter(modelPath="model/nl6_poster_default_inputs.pt", dataset=nl6dataset, normalize_out=True) 
    # wt = np.loadtxt("pso/gmm_weight/nl6p_t05.txt")
    # nl6Int.plot_contour(path="graphs/contour/nl6.png",nCols=3, groundTruthTheta = np.array([0.1, 0.1, 0.95, 0.17, 0.05, 0.18]), resolution=50, y=np.array([1.40012,1757.38,209.96,14.0588,121.369,90.9622,0.728361,273328,6695.53,201.283,6369.12,859.989,-258.233,53.35,7.06668,-18.5846,17.0746,-4524.28,-958.283,5467.64,-992.478,790.754,-1962.44,2254.3,-690.745,162.181,-220.746]))
    # nl6Int.plot_gmm_contour(path="graphs/contour/nl6gmm.png",nCols=3, groundTruthTheta = np.array([0.1, 0.1, 0.95, 0.17, 0.05, 0.18]), resolution=50, y=np.array([1.40012,1757.38,209.96,14.0588,121.369,90.9622,0.728361,273328,6695.53,201.283,6369.12,859.989,-258.233,53.35,7.06668,-18.5846,17.0746,-4524.28,-958.283,5467.64,-992.478,790.754,-1962.44,2254.3,-690.745,162.181,-220.746]), wt=wt)
    # nl6Int.plot_with_ground_truth(plotPath="graphs/interpretability/nl6_default_in", groundTruthPath="data/NL6_1k.csv",thetaStar=0, nCols=6)
    # nl6Int.plot(path="graphs/interpretability/nl6", thetaStar=0, thetaFixed=0.2, nCols=6, nSteps=10)
    
    baseName = "l3p_t"
    # models = []
    # wts = []
    # y = np.loadtxt("pso/truth/l3p_t123_mom.txt")
    # # magic number 3 for 3 tpts
    # for i in range(3):
    #     wts.append(np.loadtxt("pso/gmm_weight/" + baseName + str(i + 1) + ".txt"))
    #     models.append(tc.load("model/" + baseName + str(i+1) + ".pt"))

    # l3IntMulti = MultiInterpreter(models=models)
    # l3IntMulti.plot_mgmm_contour("graphs/contour/l3mgmm.png", nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=y, wts=wts, levels=40)
    
    
    # l3NormDataset = ABMDataset("data/static/l3p_10k_t3_5kss.csv", root_dir="data/", standardize=True, norm_out=True)
    # l3Int = DumbInterpreter(modelPath="model/l3p_10k_small_res_t3.pt", dataset=l3NormDataset, normalize_out=True, standardize_in=True)
    # wt = np.loadtxt("pso/gmm_weight/l3p_t3.txt")
    # l3Int.plot_contour(path="graphs/contour/l3_norm_t3.png",nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049]), levels=40)
    # l3Int.plot_gmm_contour(path="graphs/contour/l3_norm_gmm_t3.png",nCols=3, groundTruthTheta = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]), resolution=100, y=np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049]), wt=wt, levels = 40)
    
    
    l3tTestDataset = ABMDataset("data/time_series/l3p_unseen_data.csv", root_dir="data/time_series/")
    l3tTrainDataset = ABMDataset("data/time_series/l3pt_i.csv", root_dir="data/")
    l3T = DumbInterpreter(modelPath="model/l3p_i.pt", dataset=l3tTrainDataset)
    mse, t, predicted, tested = l3T.evaluate(l3tTestDataset, use_gpu=True)
    l3T.plot_scatter(tested, predicted, output='graphs/scatter/l3_t_test')
    
    # l3Int.plot_with_ground_truth(plotPath="graphs/interpretability/l3p_default_in", groundTruthPath="data/l3p_k1.csv",thetaStar=0, nCols=3)
    # l3Int.plot(path="graphs/interpretability/l3", thetaStar=0, thetaFixed=0.2, nCols=3, nSteps=10)
    # l3Int.plot(path="graphs/interpretability/l3", thetaStar=1, thetaFixed=0.2, nCols=3, nSteps=10)
    # l3Int.plot(path="graphs/interpretability/l3", thetaStar=2, thetaFixed=0.2, nCols=3, nSteps=10)
    # gdagdataset = ABMDataset("data/gdag1300sss_covs.csv", root_dir="data/", transform=False, standardize=False, norm_out=True)
    # gdag = DumbInterpreter(modelPath="model/gdag_default_input.pt")
    
    # gdag.plot(path="graphs/interpretability/gdag1300ss_default_in", thetaStar=0, thetaFixed=0.1, nCols=3, nSteps=20)
    # for i in range(gdag.model.input_size):
    #     gdag.plot(path="graphs/interpretability/gdag1300ss_default_in", thetaStar=i, thetaFixed=0.1, nCols=3, nSteps=10)















# # load model that was saved.
# model = tc.load("model/nl6_poster.pt")
# model.eval()
# n_inputs = model.input_size
# n_outputs = model.output_size

# # create some data
# theta_star = 0 # what index we care about
# n_steps = 10
# thetas = (2*np.ones(n_inputs)) / 10.0 # convert to a set of 0.2's
# thetas[theta_star] = 0
# print("Interpreting Input:", theta_star + 1)
# # pick a rate constant to interpret
# # run model on step sizes of rate constants
# changing_inputs = []
# outputs = []
# for i in range(n_steps):
#     changing_inputs.append(thetas[theta_star])
#     thetas[theta_star] += (1.0 / n_steps)

#     input = tc.from_numpy(thetas)
#     if next(model.parameters()).is_cuda:
#         input = input.to(tc.device("cuda"))
#     output = model(input).cpu().detach().numpy()
#     outputs.append(output)

# changing_inputs = np.array(changing_inputs)
# outputs = np.array(outputs)

# # plot change in outputs of mean counts (because that's easier to interpret)
# # subplotting 
# n_cols = 6 # Use Four Columns For Width purposes
# n_rows = int(n_outputs / n_cols)
# if n_outputs % n_cols != 0:
#     n_rows+=1


# f, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20,20))

# moment = 0

# for r in range(n_rows):
#     for c in range(n_cols):
#         if moment < n_outputs:
#             axs[r,c].plot(changing_inputs, outputs[:,moment], label='o' + 'moment')
#             moment+=1

# plt.xlabel("Theta" + str(theta_star + 1))
# plt.ylabel("Output Value")
# filename = "int_theta" + str(theta_star + 1) + "_nl6"
# plt.savefig(filename + '.png')
