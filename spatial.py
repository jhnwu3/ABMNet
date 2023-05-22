# problem with the code is that the spatial data can only be run on the cluster, because it's so massive, which means we can't run anything locally
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from textwrap import wrap
import sys
import os
import pickle
import torch 
import torch.nn.functional as F
import torch.nn as nn
import resource
from torch_geometric.nn import GCNConv
from scipy import spatial
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.spatial import *

# loaded_data = pickle.load(open("../gdag_data/gdag_graph_data.pickle", "rb"))
# GiuseppeSurrogateGraphData.chunk_pkl(loaded_data, "../gdag_data/gdag_chunked")
# @profile
def train_profiled(input_graph, output_graphs_chunk, rates_chunk, nEpochs=2):
    # for manual testing, load everything at once, and train
    model = GCNComplex(n_features=input_graph.size()[1], n_classes= output_graphs_chunk[0].size()[1], n_rates=rates_chunk[0].size()[0],hidden_channels=32)
    model.train()
    model = model.double()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    
    # device = ""
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     model = model.cuda()
    #     criterion = criterion.cuda()
    #     using_gpu = True
    # else:
    #     device = torch.device("cpu")
    #     using_gpu = False
        
    for epoch in range(nEpochs):
        loss_per_epoch = 0
       
        for graph in range(len(output_graphs_chunk)):
            optimizer.zero_grad()
            out = model(input_graph.to, edges, rates_chunk[graph])
            loss = criterion(out, output_graphs_chunk[graph])
            loss.backward()
            loss_per_epoch+=loss
            optimizer.step()
            del loss
            del out
            print('Memory usage: %s (kb)', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if epoch % 1 == 0:
            print("Epoch:", epoch, " Loss:", loss_per_epoch)   


data_directory = os.path.join("../gdag_data", "gdag_chunked")

nEpochs = 2
single_init_cond = True 
input_graph = os.path.join(data_directory, "input_graphs", "graph.pt")
output_graphs_chunk = os.path.join(data_directory, "output_graphs", "graph0.pickle")
edges = os.path.join(data_directory, "edges.pt")
rates_chunk = os.path.join(data_directory, "rates", "rates0.pickle")

input_graph = torch.load(input_graph)
output_graphs_chunk = pickle.load(open(output_graphs_chunk, "rb"))
edges = torch.load(edges)
rates_chunk = pickle.load(open(rates_chunk, "rb"))
print(type(output_graphs_chunk))
print(type(rates_chunk))
print(len(rates_chunk))
print(edges)
plot_graph_to_img(input_graph, path="test.png")
plot_graph_to_img(output_graphs_chunk[0], path="test_first_output.png")


# for i in range(len(output_graphs_chunk)):
#     print(output_graphs_chunk[i].size())
train_profiled(input_graph, output_graphs_chunk, rates_chunk)




# model = train_giuseppe_surrogate_pkl(loaded_data, nEpochs=20)




# loadedDict = pickle.load(open("../gdag_test_full.pickle","rb"))
# testDataConverted = GiuseppeSurrogateGraphData()
# testDataConverted.delaunay_edges_and_data(loadedDict)
# testDataConverted.save("../gdag_graph_data.pickle")
# plot_giuseppe_graph(testDataConverted.input_graphs, testDataConverted.edges)

# model = train_gnn(testDataConverted)
# model1 = train_giuseppe_surrogate(testDataConverted, nEpochs=50)
#
# plot_giuseppe_graph(testDataConverted.input_graphs, testDataConverted.edges, path="data/spatial/input_graph.png")
# TODO: Look at the actual graph plot of the input data you've read in
# Sanity Check #1: Write code to look at the data representation, and cross check with Giuseppe if they look right.
# Sanity Check #2: look at graph
# Sanity Check #3: Look at neural network implementation and seriously think about what happens with sparse representations of feature vectors.
# i.e look at Soham's implementation.


# loadedDict = pickle.load(open("data/spatial/test.pickle","rb"))
# for keys in loadedDict.keys():
#     print(np.frombuffer(keys))