import torch as tc
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from ABM import *
# 
class DumbInterpreter:
    def __init__(self, modelPath, dataset=None, normalize_out=False):
        self.modelPath = modelPath
        self.model = tc.load(modelPath)
        self.dataset = dataset
        self.norm_out = normalize_out
    
    def plot_with_ground_truth(self, plotPath, groundTruthPath, thetaStar, nCols):
        # load input data
        groundTruth = pd.read_csv(groundTruthPath)
        # load based on input size
        groundTruth = groundTruth.to_numpy()
        inputs = groundTruth[:,:self.model.input_size].copy()
        trueOutputs = groundTruth[:,self.model.input_size:].copy()
        
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
            changing_inputs.append(thetas[thetaStar])
            thetas[thetaStar] += (1.0 / nSteps)

            input = tc.from_numpy(thetas)
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







if __name__ == "__main__":
    
    nl6dataset = ABMDataset("data/NL6P.csv", root_dir="data/", transform=False, standardize=False, norm_out=True)
    nl6Int= DumbInterpreter(modelPath="model/nl6_poster_default_inputs.pt", dataset=nl6dataset, normalize_out=True) 
    nl6Int.plot_with_ground_truth(plotPath="graphs/interpretability/nl6_default_in", groundTruthPath="data/NL6_1k.csv",thetaStar=0, nCols=6)
    nl6Int.plot(path="graphs/interpretability/nl6", thetaStar=0, thetaFixed=0.2, nCols=6, nSteps=10)
    
    l3Int = DumbInterpreter(modelPath="model/l3p_poster.pt") 
    l3Int.plot_with_ground_truth(plotPath="graphs/interpretability/l3p_default_in", groundTruthPath="data/l3p_k1.csv",thetaStar=0, nCols=3)
    l3Int.plot(path="graphs/interpretability/l3", thetaStar=0, thetaFixed=0.2, nCols=3, nSteps=10)
    l3Int.plot(path="graphs/interpretability/l3", thetaStar=1, thetaFixed=0.2, nCols=3, nSteps=10)
    l3Int.plot(path="graphs/interpretability/l3", thetaStar=2, thetaFixed=0.2, nCols=3, nSteps=10)
    # gdagdataset = ABMDataset("data/gdag1300sss_covs.csv", root_dir="data/", transform=False, standardize=False, norm_out=True)
    gdag = DumbInterpreter(modelPath="model/gdag_default_input.pt")
    
    # gdag.plot(path="graphs/interpretability/gdag1300ss_default_in", thetaStar=0, thetaFixed=0.1, nCols=3, nSteps=20)
    for i in range(gdag.model.input_size):
        gdag.plot(path="graphs/interpretability/gdag1300ss_default_in", thetaStar=i, thetaFixed=0.1, nCols=3, nSteps=10)















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
