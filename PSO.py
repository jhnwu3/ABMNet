import numpy as np
import pandas as pd
import torch as tc 
import pyswarms as ps
import random
from pyswarms.single.global_best import GlobalBestPSO
from modules.utils.graph import *
from modules.data.mixed import *
from modules.utils.pso import *
if __name__ == "__main__":
    baseName = "l3p_t"
    
    
    
    
    # # multi surrogates
    # models = []
    # wts = []
    # y = np.loadtxt("pso/truth/l3p_t123_mom.txt")
    # # magic number 3 for 3 tpts
    # for i in range(3):
    #     wts.append(np.loadtxt("pso/gmm_weight/" + baseName + str(i + 1) + ".txt"))
    #     models.append(tc.load("model/" + baseName + str(i+1) + ".pt"))
        
    # bounds = (np.zeros(models[0].input_size), np.ones(models[0].input_size))
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # optimizer = GlobalBestPSO(n_particles=1, dimensions=models[0].input_size, options=options, bounds=bounds)
    
    # cost, pos = optimizer.optimize(multi_gmm_cost, iters=30, surrogates=models, y = y, wts = wts)
    # print("GMM:", cost)
    
    # mse_estimate = 0
    # gmm_truth = 0
    # mse_truth = 0
    # for i in range(len(models)):
    #     # print(wts[i].shape)
    #     mse_estimate+= numpy_mse(y[i], models[i](tc.from_numpy(np.array(pos)).to(tc.device("cuda"))).cpu().detach().numpy())
        
    #     output = models[i](tc.from_numpy(np.array(np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]))).to(tc.device("cuda"))).cpu().detach().numpy()
    #     gmm_truth +=np.matmul(output-y[i], np.matmul((output - y[i]).transpose(), wts[i]))
     
    #     mse_truth += numpy_mse(y[i], output)
        
    #     # print(y[i])

    # print("MSE of estimate:",mse_estimate)

    # # outputs = sgModel(tc.from_numpy(np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]))).cpu().detach().numpy()
    
    # print("GMM Cost of ground truth l3p:", gmm_truth)
    # print("MSE of ground truth l3p:", mse_truth)
    # print(gmm_cost(x, sgModel, y, wt ))
    # # print(sgModel.parameters)
    # gTruth = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502])

    # bounds = (np.zeros(sgModel.input_size), np.ones(sgModel.input_size))
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # optimizer = GlobalBestPSO(n_particles=1500, dimensions=sgModel.input_size, options=options, bounds=bounds)
    # cost, pos = optimizer.optimize(gmm_cost, iters=30, surrogate=sgModel, y = y, wt = wt, dataset=l3Dataset100k, standardize = True, normalize=True)
    # print("GMM:", cost)
    # pos = tc.from_numpy(np.array(pos))
    # gTruth = tc.from_numpy(gTruth)
    # if next(sgModel.parameters()).is_cuda:
    #     pos = pos.to(tc.device("cuda"))
    #     gTruth = gTruth.to(tc.device("cuda"))
        
    # print("MSE of estimate:",numpy_mse(y, sgModel(pos).cpu().detach().numpy()) )
    # output = sgModel(gTruth).cpu().detach().numpy()
    # print("GMM Cost of ground truth l3p:", np.matmul(output - y, np.matmul((output - y).transpose(), wt)))
    # print("MSE of ground truth l3p:", numpy_mse(y, output))
    
    
    
    # # PSO for time as a feature model
    
    # model = tc.load("model/l3p_i.pt")
    # print(model.parameters)
    