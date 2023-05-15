import numpy as np
import torch as tc 
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from graph import *
from abm import *

def gmm_cost(x, surrogate, y, wt, dataset=None, standardize=False, normalize=False):
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
            thetaCopy = x[i]
            if standardize:
                thetaCopy = (thetaCopy - dataset.input_means) / dataset.input_stds
            input = tc.from_numpy(thetaCopy)
            if next(surrogate.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            output = surrogate(input).cpu().detach().numpy()
            if normalize: 
                scale_factor = dataset.output_maxes - dataset.output_mins
                output = (output * scale_factor) + dataset.output_mins
            # print(wt)
            costs.append(np.matmul(output-y, np.matmul((output - y).transpose(), wt)))
    
    # print(costs)
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
            cost += np.matmul(output-y[w], np.matmul((output - y[w]).transpose(), wt))
            w+=1
        costs.append(cost)
    else:
        for i in range(x.shape[0]):
            cost = 0
            w = 0
            for surrogate in surrogates:
                input = tc.from_numpy(x[i])
                if next(surrogate.parameters()).is_cuda:
                    input = input.to(tc.device("cuda"))
                output = surrogate(input).cpu().detach().numpy()
                cost+= np.matmul(output-y[w], np.matmul((output - y[w]).transpose(), wts[w]))
                w+=1
                
            costs.append(cost)
    
    # print(costs)
    # print("output:",output.shape)
    # print("wt:", wt.shape)
    # print("y:", y.shape)
    return np.array(costs)

# let t be the length of the time points
def multi_gmm_cost(x, t, surrogates, y, wts ):
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
            cost += np.matmul(output-y[w], np.matmul((output - y[w]).transpose(), wt))
            w+=1
        costs.append(cost)
    else:
        for i in range(x.shape[0]):
            cost = 0
            w = 0
            for surrogate in surrogates:
                print(x[i].shape)
                input = tc.from_numpy(x[i])
               
                if next(surrogate.parameters()).is_cuda:
                    input = input.to(tc.device("cuda"))
                output = surrogate(input).cpu().detach().numpy()
                cost+= np.matmul(output-y[w], np.matmul((output - y[w]).transpose(), wts[w]))
                w+=1
                
            costs.append(cost)
    
    # print(costs)
    # print("output:",output.shape)
    # print("wt:", wt.shape)
    # print("y:", y.shape)
    return np.array(costs)

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
    
    # # pso for hard trained model
    l3Dataset100k = ABMDataset("data/static/l3p_100k.csv", root_dir="data/", standardize=True, norm_out=True)
    sgModel = tc.load("model/l3p_100k_small_t3.pt")
    wt = np.loadtxt("pso/gmm_weight/l3p_t3.txt")
    # # wt = np.identity(sgModel.output_size)
    # # print(wt)
    x = np.zeros(sgModel.input_size)
    y = np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049])
    print(gmm_cost(x, sgModel, y, wt ))
    # print(sgModel.parameters)
    gTruth = np.array([0.27678200,0.83708059,0.44321700,0.04244124, 0.30464502])

    bounds = (np.zeros(sgModel.input_size), np.ones(sgModel.input_size))
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=1500, dimensions=sgModel.input_size, options=options, bounds=bounds)
    cost, pos = optimizer.optimize(gmm_cost, iters=30, surrogate=sgModel, y = y, wt = wt, dataset=l3Dataset100k, standardize = True, normalize=True)
    print("GMM:", cost)
    pos = tc.from_numpy(np.array(pos))
    gTruth = tc.from_numpy(gTruth)
    if next(sgModel.parameters()).is_cuda:
        pos = pos.to(tc.device("cuda"))
        gTruth = gTruth.to(tc.device("cuda"))
        
    print("MSE of estimate:",numpy_mse(y, sgModel(pos).cpu().detach().numpy()) )
    output = sgModel(gTruth).cpu().detach().numpy()
    print("GMM Cost of ground truth l3p:", np.matmul(output - y, np.matmul((output - y).transpose(), wt)))
    print("MSE of ground truth l3p:", numpy_mse(y, output))
    
    
    
    # # PSO for time as a feature model
    
    # model = tc.load("model/l3p_i.pt")
    # print(model.parameters)
    