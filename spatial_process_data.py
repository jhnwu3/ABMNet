import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import torch 
import gc
from torch.cuda.amp import autocast
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.spatial import *


# testing on own pc


# cluster
path = "../gdag_data/gdag_test_full.pickle"
print("Reading From:", path)
lattice_data = pickle.load(open(path, "rb"))
data_processed = GiuseppeSurrogateGraphData()



# print("Computing Delaunay Moments")
# data_processed.delaunay_moments(lattice_data, channels=[0,1,2,3,6])
# print(data_processed.output_graphs[0].size())
# print(len(data_processed.input_graphs))
# print(len(data_processed.output_graphs))
# print(len(data_processed.rates))
# data_processed.save("../gdag_data/gdag_spatial_moments.pickle")

print("Computing Delaunay AutoCorrelation")
data_processed.delaunay_autocorrelation(lattice_data, channels=[0,1,2,3,6])
print(data_processed.output_graphs[0].size())
print(len(data_processed.input_graphs))
print(len(data_processed.output_graphs))
print(len(data_processed.rates))
data_processed.save("../gdag_data/gdag_autocorr.pickle")