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
from torch_geometric.nn import GCNConv
from scipy import spatial
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.spatial import *

loaded_data = pickle.load(open("../gdag_data/gdag_graph_data.pickle", "rb"))





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