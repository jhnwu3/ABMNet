import numpy as np
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

def indrani_cost(x, surrogate, y, dataset, standardize=True, normalize = True, batch=True):
    costs = []
    if len(x.shape) < 2:
        thetaCopy = x
        if standardize:
            thetaCopy = (thetaCopy - dataset.input_means) / dataset.input_stds
        input = tc.from_numpy(x)
        
        if next(surrogate.parameters()).is_cuda:
            input = input.to(tc.device("cuda"))
            
        if batch: 
            input = input.unsqueeze(dim=0)

        output = surrogate(input).squeeze().cpu().detach().numpy()

        if normalize: 
            scale_factor = dataset.output_maxes - dataset.output_mins
            output = (output * scale_factor) + dataset.output_mins
            
        costs.append(np.matmul(output-y, np.matmul((output - y).transpose(), wt)))
    else:
        print(x.shape)
        for i in range(x.shape[0]):
            thetaCopy = x[i]
            if standardize:
                thetaCopy = (thetaCopy - dataset.input_means) / dataset.input_stds
            input = tc.from_numpy(thetaCopy)
            if next(surrogate.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            if batch:
                input = input.squeeze() # this is messy I agree...
                input = input.unsqueeze(dim=0)
            output = 0 
            with tc.no_grad():
                output = surrogate(input).squeeze().cpu().detach().numpy()
            if normalize: 
                scale_factor = dataset.output_maxes - dataset.output_mins
                output = (output * scale_factor) + dataset.output_mins

            costs.append(np.matmul(output-y, np.matmul((output - y).transpose(), wt)))

    return np.array(costs)

def gmm_cost(x, surrogate, y, wt, dataset=None, standardize=False, normalize=False, batch=True):
    costs = []
    if len(x.shape) < 2:
        thetaCopy = x
        if standardize:
            thetaCopy = (thetaCopy - dataset.input_means) / dataset.input_stds
        input = tc.from_numpy(x)
        
        if next(surrogate.parameters()).is_cuda:
            input = input.to(tc.device("cuda"))
            
        if batch: 
            input = input.unsqueeze(dim=0)

        output = surrogate(input).squeeze().cpu().detach().numpy()

        if normalize: 
            scale_factor = dataset.output_maxes - dataset.output_mins
            output = (output * scale_factor) + dataset.output_mins
            
        costs.append(np.matmul(output-y, np.matmul((output - y).transpose(), wt)))
    else:
        print(x.shape)
        for i in range(x.shape[0]):
            thetaCopy = x[i]
            if standardize:
                thetaCopy = (thetaCopy - dataset.input_means) / dataset.input_stds
            input = tc.from_numpy(thetaCopy)
            if next(surrogate.parameters()).is_cuda:
                input = input.to(tc.device("cuda"))
            if batch:
                input = input.squeeze() # this is messy I agree...
                input = input.unsqueeze(dim=0)
            output = 0 
            with tc.no_grad():
                output = surrogate(input).squeeze().cpu().detach().numpy()
            if normalize: 
                scale_factor = dataset.output_maxes - dataset.output_mins
                output = (output * scale_factor) + dataset.output_mins

            costs.append(np.matmul(output-y, np.matmul((output - y).transpose(), wt)))

    return np.array(costs)

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

    model.eval()
    pbmat = np.concatenate((posmat, max_array[:, np.newaxis]), axis=1)
    gbest = np.random.rand(1, model.input_size)

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
            cost = 0
            cost = gmm_cost(posmat[p], model, y, wt, dataset=dataset, standardize=standardize, normalize=normalize_out, batch=batch)
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
    
    return np.array(costs)