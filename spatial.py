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

parent_data_dir = "../../share/Giuseppe_John/training_data_with_dump_files_ParamSweep_11-03-2023_30_samples/"
parent_data_dir = "data/spatial/"

# test = GiuseppeSpatialDataProcessor(parent_data_dir)
# # print(test.spatialData[0].rates)
# # print(test.spatialData[1].rates)
# # print(test.spatialData[0].data[0][0])
# test.convert_to_pickle("../gdag_test_full.pickle")
# test.print_data_statistics()
loadedDict = pickle.load(open("../gdag_test.pickle","rb"))
testDataConverted = GiuseppeSurrogateGraphData()
testDataConverted.delaunay_edges_and_data(loadedDict)
plot_giuseppe_graph(testDataConverted.input_graphs, testDataConverted.edges)

# model = train_gnn(testDataConverted)
# model1 = train_giuseppe_surrogate(testDataConverted, nEpochs=50)




# loadedDict = pickle.load(open("data/spatial/test.pickle","rb"))
# for keys in loadedDict.keys():
#     print(np.frombuffer(keys))