import numpy as np
import pandas as pd
import torch as tc 
import pyswarms as ps
import random
from pyswarms.single.global_best import GlobalBestPSO
from modules.utils.graph import *
from modules.data.mixed import *

def rpoint(og_pos, seed=3, epsi=0.02, nan=0.02, hone =28):
    
    min_length = 0
    max_length = og_pos.shape[0]
    length = np.random.randint(min_length, max_length + 1)
    indices = random.sample(range(max_length), length)
    new_pos = og_pos
    for i in range(len(indices)):
        if new_pos[indices[i]] > 1.0 -nan:
            new_pos[indices[i]] -= epsi 
        elif (new_pos[indices[i]]) < nan: 
            new_pos[indices[i]] += nan 
        alpha = hone * new_pos[indices[i]]
        beta = hone - alpha
        new_pos[indices[i]] = np.random.beta(alpha,beta)

    return new_pos

# ported version of Dr. Stewart's PSO
def StewartPSO(model, y, wt, n_steps, n_particles, dataset, standardize=False, normalize_out=True, pBestW_init = 3, gBestW_init = 1, pInertiaW_init = 6, batch=True):
    pBestW_init = 3 
    gBestW_init = 1 
    pInertiaW_init = 6

    pBestW_curr = pBestW_init 
    gBestW_curr = gBestW_init
    pInertiaW_curr = pInertiaW_init
   
    posmat = np.random.rand(n_particles, model.input_size)
    pbmat = posmat # the extra 1 is for the cost saving to check. b.c we need to use both pbmat and gbest for updates.
    max_float = np.finfo(np.float64).max  # Maximum value for float64 data type
    max_array = np.full(n_particles, max_float)

    # print(max_array)
    model.eval()
    pbmat = np.concatenate((posmat, max_array[:, np.newaxis]), axis=1)
    gbest = np.random.rand(1, model.input_size)
    # print(gbest)
    gcost = gmm_cost(gbest, model, y, wt, dataset=dataset, standardize=standardize, normalize=normalize_out, batch=batch)
    print("initialized gcost:", gcost)
    print("with gbest:", gbest)
    for step in range(n_steps):
        w1 = pInertiaW_curr * np.random.uniform(0,1) 
        w2 = pBestW_curr * np.random.uniform(0,1)
        w3 = gBestW_curr * np.random.uniform(0,1)
        sumW = w1 + w2 + w3 
        w1 /= sumW 
        w2 /= sumW 
        w3 /= sumW 

        for p in range(n_particles):
            # print(dataset)
            cost = 0
            # exit(0)
            cost = gmm_cost(posmat[p], model, y, wt, dataset=dataset, standardize=standardize, normalize=normalize_out, batch=batch)
            # print(cost)
            # print(posmat[p])
            if float(cost) < pbmat[p,-1]:
                pbmat[p, -1] = cost 
                pbmat[p,:-1] = posmat[p]
            if float(cost) < float(gcost): 
                gcost = cost 
                gbest = posmat[p]
                print("New Global Best:", gbest, " with cost:", gcost)

            posmat[p] = w1 * rpoint(posmat[p]) + w2 * pbmat[p,:posmat.shape[1]]  + w3 * gbest
        
        pInertiaW_curr = pInertiaW_curr - (pInertiaW_init - gBestW_init) / n_steps 
        gBestW_curr = gBestW_curr + (pInertiaW_init - gBestW_init) / n_steps 

    return gcost, gbest


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
    # l3Dataset100k = ABMDataset("data/static/l3p_100k.csv", root_dir="data/", standardize=True, norm_out=True)
    # sgModel = tc.load("model/l3p_100k_large_batch_normed.pt")
    # wt = np.loadtxt("pso/gmm_weight/l3p_t3.txt")
    # # wt = np.identity(sgModel.output_size)
    # # # print(wt)
    # x = np.zeros(sgModel.input_size)
    # y = np.array([12.4509,  6.9795, 9.06247, 93.9796, 31.9489, 84.5102, 53.8117, 72.7715, 47.3049])
    
    
    # estimates = []
    # for i in range(10):   
    #     gcost, gbest = StewartPSO(sgModel, y, wt, n_steps=50, n_particles=500, dataset=l3Dataset100k, standardize=False, normalize_out=True, batch=True)
    #     estimates.append(gbest)

    # estimates = np.array(estimates)
    # print("Surrogate Estimates:")
    # print(estimates)
    # print("Means:",np.mean(estimates,axis=0))
    # print("Var:",np.var(estimates, axis=0))
    # # from BioNetGMMFit:
    bngmm_est = "data/BNGMMFit/model_estimates.csv"
    print("BioNetGMMFit Estimates 200 Particles, 40 steps:")
    data = np.genfromtxt(bngmm_est, delimiter=",", skip_header=True)
    np.set_printoptions(suppress=True)
    print("Means:", np.mean(data, axis=0))
    print("Var:", np.var(data, axis=0))
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
    