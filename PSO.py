import numpy as np
import torch as tc 
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from GRPH import *

def gmm_cost(x, surrogate, y, wt):
    # print("x:",x.shape)
    # print(x)
    # print(len(x.shape))
    
    # print(input.size)
    costs = []
    if len(x.shape) < 2:
        input = tc.from_numpy(x)
        if next(surrogate.parameters()).is_cuda:
            input = input.to(tc.device("cuda"))
        output = surrogate(input).cpu().detach().numpy()
        costs.append(np.matmul(output-y, np.matmul((output - y).transpose(), wt)))
        
    else:
        for i in range(x.shape[0]):
            input = tc.from_numpy(x[i])
            if next(surrogate.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            output = surrogate(input).cpu().detach().numpy()
            costs.append(np.matmul(output-y, np.matmul((output - y).transpose(), wt)))
    
    print(costs)
    # print("output:",output.shape)
    # print("wt:", wt.shape)
    # print("y:", y.shape)
    return np.array(costs)

def multi_gmm_cost(x, surrogates, y, wts):
    costs = []
   
    if len(x.shape) < 2:
        cost = 0
        w = 0
        for surrogate in surrogates:
            wt = wts[w]
            input = tc.from_numpy(x)
            if next(surrogate.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            output = surrogate(input).cpu().detach().numpy()
            cost += np.matmul(output-y, np.matmul((output - y).transpose(), wt))
        costs.append(cost)
    else:
        for i in range(x.shape[0]):
            cost = 0
            w = 0
            for surrogate in surrogates:
                wt = wts[w]
                input = tc.from_numpy(x[i])
                if next(surrogate.parameters()).is_cuda:
                    input = input.to(tc.device("cuda"))
                output = surrogate(input).cpu().detach().numpy()
                cost+= np.matmul(output-y, np.matmul((output - y).transpose(), wt))
                
            costs.append(cost)
    
    print(costs)
    # print("output:",output.shape)
    # print("wt:", wt.shape)
    # print("y:", y.shape)
    return np.array(costs)

if __name__ == "__main__":
    baseName = "l3p_t"
    models = []
    wts = []
    y = []
    # magic number 3 for 3 tpts
    for i in range(3):
        wts.append(np.loadtxt("pso/gmm_weight/" + baseName + str(i + 1) + ".txt"))
        models.append(tc.load("model/" + baseName + str(i+1) + ".pt"))
    
    sgModel = tc.load("model/l3p_t3_10k.pt")
    wt = np.loadtxt("pso/gmm_weight/l3p_t3inv.txt")
    # wt = np.identity(sgModel.output_size)
    # print(wt)
    x = np.zeros(sgModel.input_size)
    y = np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049])
    print(gmm_cost(x, sgModel, y, wt ))
    # print(sgModel.parameters)
    
    bounds = (np.zeros(sgModel.input_size), np.ones(sgModel.input_size))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=1500, dimensions=sgModel.input_size, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(gmm_cost, iters=30, surrogate=sgModel, y = y, wt = wt)
    print("GMM:", cost)
    print("MSE of estimate:",numpy_mse(y, sgModel(tc.from_numpy(np.array(pos))).cpu().detach().numpy()) )
    
    output = sgModel(tc.from_numpy(np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]))).cpu().detach().numpy()
    print("GMM Cost of ground truth l3p:", np.matmul(output - y, np.matmul((output - y).transpose(), wt)))
    print("MSE of ground truth l3p:", numpy_mse(y, sgModel(tc.from_numpy(np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502]))).cpu().detach().numpy()))