import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import roadrunner
import torch
from torch.utils.data import DataLoader, IterableDataset, Dataset
from modules.data.mixed import *
import pickle

def chunk_sequence(sequence, chunk_size):
    num_chunks = len(sequence) // chunk_size
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = sequence[start:end]

        if chunk.size()[0] > 1:
            chunks.append(chunk)

    # Handle the remaining elements
    remainder = len(sequence) % chunk_size
    if remainder != 0:
        last_chunk = sequence[-remainder:]
        if last_chunk.size()[0] > 1:
            chunks.append(last_chunk)

    return chunks



# Cytotoxic CD8+ T Cells, Cancer, Exhausted CD8+ T Cells, Dead Cancer Cells, Ignore, Ignore, TAMs, Ignore, Ignore
class TemporalDataset(Dataset):
    # path to a pickle file that contains a dictionary of the following variables shown below
    # [] is a list of indices of features to keep in the input and output graphs
    def __init__(self, path, min_max_scale = True, standardize_inputs = True, steps=1):
        # Initialize your dataset here
        # Store the necessary data or file paths
        data = pickle.load(open(path, "rb"))
        self.outputs = data["outputs"] # N x L tensors
        self.rates = data["rates"]
        self.times = data["time_points"]
        self.n_rates = self.rates[0].size()[0]
        self.input_size = 1
        self.steps_into_future = steps
        self.min = None 
        self.max = None
        self.input_mean = None
        self.input_std = None
        if len(self.outputs[0].size()) > 1:
            print(self.outputs[0].size())
            self.input_size = self.outputs[0].size()[1]
            
        if min_max_scale:
            print("Min Maxed Applied")
            # mins and maxes in the most roundabout way possible, haha ;(
            # convert back to numpy to get them mins and maxes 
            arr = []
            for output in self.outputs:
                arr.append(output.numpy())
            arr = np.array(arr)
            self.min = arr.min(axis=0).min(axis=0)
            print("Found Minimums:", self.min)
            self.max = arr.max(axis=0).max(axis=0)
            print("Found Max:", self.max)
            # now to do the ugly min maxing, Don't DO THIS KIDS
            for i in range(len(self.outputs)):
                self.outputs[i] = self.outputs[i].squeeze() - arr.min(axis=0).min(axis=0)
                self.outputs[i] /= (arr.max(axis=0).max(axis=0) - arr.min(axis=0).min(axis=0))
                
        # standard
        if standardize_inputs:
            print("Standardization to Input Parameters Applied")
            arr = []
            for rate in self.rates:
                arr.append(rate.numpy())
            arr = np.array(arr)
            self.input_mean = arr.mean(axis=0)
            self.input_std = arr.std(axis=0)
            print("Found Mean Rates:", self.input_mean)
            print("With Std:", self.input_std)
            for i in range(len(self.rates)):
                self.rates[i] = (self.rates[i] - arr.mean(axis=0)) / arr.std(axis=0)  
            
    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.rates) # len(rates) == len(output_graphs)
    
    def __getitem__(self, index):
        # Retrieve a single item from the dataset based on the given index
        # Return a tuple (input, target) or dictionary {'input': input, 'target': target}
        # The input and target can be tensors, NumPy arrays, or other data types
        # returns a 1D tensor of rates, one input sequence, one output sequence
        # need to convert to sets of sequences
        return self.rates[index], self.outputs[index][:-self.steps_into_future], self.outputs[index][self.steps_into_future:]



class TemporalChunkedDataset(Dataset):
    def __init__(self, path, min_max_scale = True, standardize_inputs = True, time_chunk_size=5, batch_size=None, steps=5):
            # Initialize your dataset here
        # Store the necessary data or file paths
        data = pickle.load(open(path, "rb"))
        self.outputs = data["outputs"] # N x L tensors
        self.rates = data["rates"]
        self.times = data["time_points"]
        self.n_rates = self.rates[0].size()[0]
        self.input_size = 1
        self.batch_size = batch_size
        self.steps_into_future = steps
        if len(self.outputs[0].size()) > 1:
            # print(self.outputs[0].size())
            self.input_size = self.outputs[0].size()[1]
        self.min = None 
        self.max = None
        if min_max_scale:
            print("Min Maxed Applied")
            # mins and maxes in the most roundabout way possible, haha ;(
            # convert back to numpy to get them mins and maxes 
            arr = []
            for output in self.outputs:
                arr.append(output.numpy())
            arr = np.array(arr)
            self.min = arr.min(axis=0).min(axis=0)
            print("Found Minimums:", self.min)
            self.max = arr.max(axis=0).max(axis=0)
            print("Found Max:", self.max)
            # now to do the ugly min maxing, Don't DO THIS KIDS
            for i in range(len(self.outputs)):
                self.outputs[i] = self.outputs[i].squeeze() - arr.min(axis=0).min(axis=0)
                self.outputs[i] /= (arr.max(axis=0).max(axis=0) - arr.min(axis=0).min(axis=0))
                
        # standard
        if standardize_inputs:
            print("Standardization to Input Parameters Applied")
            arr = []
            for rate in self.rates:
                arr.append(rate.numpy())
            arr = np.array(arr)
            self.input_mean = arr.mean(axis=0)
            self.input_std = arr.std(axis=0)
            print("Found Mean Rates:", self.input_mean)
            print("With Std:", self.input_std)
            for i in range(len(self.rates)):
                self.rates[i] = (self.rates[i] - arr.mean(axis=0)) / arr.std(axis=0)  
            
        # we need to then chunk all of them into little rate x time_chunk_size pairs.
        # and put them back into the self.outputs and self.rates
        chunked_rates = []
        chunked_outputs = []
        for i in range(len(self.outputs)):
            chunked = chunk_sequence(self.outputs[i], time_chunk_size)
            
            chunked_outputs = chunked_outputs + chunked 
            # print(chunked_outputs)
            # exit(0)
            for j in range(len(chunked)):
                chunked_rates.append(self.rates[i])
        self.rates = chunked_rates
        self.outputs = chunked_outputs




    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.rates) # len(rates) == len(output_graphs)
    
    def __getitem__(self, index):
        # Retrieve a single item from the dataset based on the given index
        # Return a tuple (input, target) or dictionary {'input': input, 'target': target}
        # The input and target can be tensors, NumPy arrays, or other data types
        # returns a 1D tensor of rates, one input sequence, one output sequence
        # need to convert to sets of sequences
        if self.batch_size is None:
            return self.rates[index], self.outputs[index][:-1], self.outputs[index][1:]
        else: 
            return self.rates[index], self.outputs[index][:-self.steps_into_future].unsqueeze(dim=1), self.outputs[index][self.steps_into_future:].unsqueeze(dim=1)


def generate_static_dataset(dataset : TemporalDataset, t):
    rates = dataset.rates 
    output = []
    for i in range(len(dataset)):
        output.append(dataset.outputs[i][t])

    return StaticDataset(rates, output)


def combine_temporal_pickles(file1, file2, save=True, path=""):
    data1 = None
    data2 = None
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        data1 = pickle.load(f1)
        data2 = pickle.load(f2)

    combined_data = {}

    for key in data1:
        if key is not "time_points":
            combined_data[key] = data1[key] + data2[key]
        else:
            combined_data[key] = data1[key]
            
    if save:
        with open(path, 'wb') as f:
            pickle.dump(combined_data, f)
    
    return combined_data


if __name__ == "__main__":
    data = TemporalDataset("data/time_series/indrani_gamma_no_zeroes.pickle")
    print(len(data))
    data_chunked = TemporalChunkedDataset("data/time_series/indrani_gamma_no_zeroes.pickle")
    print(len(data_chunked))
    for i in range(len(data_chunked)):
        rates, input, output = data_chunked[i]
        if input.size()[0] < 1 or output.size()[0] < 1:
            print("error we found some 0 sequences")
            print(i)
            print(input.size())
            print(output.size())
            exit(0)
    