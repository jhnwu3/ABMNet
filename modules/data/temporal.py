import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import roadrunner
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
import pickle

# Cytotoxic CD8+ T Cells, Cancer, Exhausted CD8+ T Cells, Dead Cancer Cells, Ignore, Ignore, TAMs, Ignore, Ignore
class TemporalDataset(Dataset):
    # path to a pickle file that contains a dictionary of the following variables shown below
    # [] is a list of indices of features to keep in the input and output graphs
    def __init__(self, path, min_max_scale = True):
        # Initialize your dataset here
        # Store the necessary data or file paths
        data = pickle.load(open(path, "rb"))
        self.outputs = data["outputs"] # N x L tensors
        self.rates = data["rates"]
        self.n_rates = self.rates[0].size()[0]
        self.input_size = 1
        if len(self.outputs[0].size()) > 1:
            print(self.outputs[0].size())
            self.input_size = self.outputs[0].size()[1]
            
        if min_max_scale:
            # mins and maxes in the most roundabout way possible, haha ;(
            # convert back to numpy to get them mins and maxes 
            arr = []
            for output in self.outputs:
                arr.append(output.numpy())
            arr = np.array(arr)
            # now to do the ugly min maxing, Don't DO THIS KIDS
            for i in range(len(self.outputs)):
                self.outputs[i] = self.outputs[i].squeeze() - arr.min()
                self.outputs[i] /= (arr.max() - arr.min())
            
            
    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.rates) # len(rates) == len(output_graphs)
    
    def __getitem__(self, index):
        # Retrieve a single item from the dataset based on the given index
        # Return a tuple (input, target) or dictionary {'input': input, 'target': target}
        # The input and target can be tensors, NumPy arrays, or other data types
        # returns a 1D tensor of rates, one input sequence, one output sequence
        # need to convert to sets of sequences
        return self.rates[index], self.outputs[index][:-1], self.outputs[index][1:]



if __name__ == "__main__":
    time_points = 500
    num_parameters = 1000
    num_init_cond = 5000

    rr = roadrunner.RoadRunner("3pro_sbml.xml")
    initial_conditions_arr = np.loadtxt("3linX0.csv", delimiter=",", dtype=float)
    rr.k1 = 0.276782
    rr.k2 = 0.837081
    rr.k3 = 0.443217
    rr.k4 = 0.0424412
    rr.k5 = 0.304645

    # EXAMPLE CODE:
    # x_mat = np.zeros((5000, 3))
    # # for x in range(num_init_cond):
    # rr.model.setFloatingSpeciesInitConcentrations(initial_conditions_arr[0])
    # rr.setIntegrator("gillespie")
    # result = rr.simulate(0, 100, 100)
    # print(result)

    times = np.linspace(0.0, 5.0, num=time_points+1)

    mean_arr = np.zeros((time_points, 3))
    var_arr = np.zeros((time_points, 3))
    cov_arr = np.zeros((time_points, 3))

    mean_arr[0] = np.mean(initial_conditions_arr, axis=0)
    var_arr[0] = np.var(initial_conditions_arr, axis=0)
    c = np.cov(initial_conditions_arr, rowvar=False)
    n = 0
    for i in range(c.shape[0] - 1):
        for j in range(1, c.shape[1]):
            if(i != j):
                cov_arr[0][n] = c[i,j]
                n += 1

    for t in range(1, time_points):
        x_mat = np.zeros((5000, 3))
        for x in range(num_init_cond):
            rr.model.setFloatingSpeciesInitConcentrations(initial_conditions_arr[x])
            rr.setIntegrator("gillespie")
            result = rr.simulate(0, times[t], 2)
            x_mat[x][0] = result[1][1]
            x_mat[x][1] = result[1][2]
            x_mat[x][2] = result[1][3]
        mean_arr[t] = np.mean(x_mat, axis=0)
        var_arr[t] = np.var(x_mat, axis=0)
        c = np.cov(x_mat, rowvar=False)
        n = 0
        for i in range(c.shape[0] - 1):
            for j in range(1, c.shape[1]):
                if(i != j):
                    cov_arr[t][n] = c[i,j]
                    n += 1

    times = times[:time_points].reshape((time_points, 1))

    total_mat = np.hstack((mean_arr, var_arr, cov_arr, times))
    np.savetxt("3linyt2.csv", total_mat, delimiter=",")