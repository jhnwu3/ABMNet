# problem with the code is that the spatial data can only be run on the cluster, because it's so massive, which means we can't run anything locally
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from textwrap import wrap
import sys
import os




# each spatial object has 3 specific components
# 1 set of rate constants that generated the data
# a initial set of conditions (sample)
# a final set of conditions (sample)
# input: directory name
class SpatialObj():
    # dirname = parameters_x_y_z_f
    def get_rates(dirname : str):
        prefix = "parameters"
        separator = "_"
        path_separated = dirname.split('/')
        rates_string = ""
        for word in path_separated:
            if prefix in word:
                rates_string = word[word.rindex(prefix):]
        
        rates = rates_string.split(separator)
        return np.array([float(i) for i in rates[1:]])
    
    # list where each entity is a tuple of 2 time pts (for now and simplicity).
    def get_time_data(dirname : str):
        sample_dirs = [os.path.join(dirname, x) for x in os.listdir(dirname)]
        data = []
        for dir in sample_dirs:
            if dirname != dir:
                file_i = os.path.join(dir, "dump.2D_model.0")
                file_t = os.path.join(dir, "dump.2D_model.1")
                # dump 
                dump_i = SpatialObj.process_dump_file(file_i)
                dump_t = SpatialObj.process_dump_file(file_t)
                # translate
                data_i = SpatialObj.translate_to_x_y(dump_i)
                data_t = SpatialObj.translate_to_x_y(dump_t)
                # append
                data.append((data_i, data_t))
            
        # return
        return data
            
    def process_dump_file(file):
        # eliminate all 11 lines until the matrix
        f = open(file, "r")
        lines = f.readlines()
        lines = lines[11:]
        # load remaining matrix into numpy
        data = np.array([np.fromstring(i, sep=" ") for i in lines])
        f.close()
        # return
        return data
        
    #Pixel, Cytotoxic CD8+ T Cells, Cancer, Exhausted CD8+ T Cells, Dead Cancer Cells, Ignore, Ignore, TAMs, Ignore
    #  x,y,class
    def translate_to_x_y(matrix, width=100, features=9):
        # convert indices in far left column to 
        # get some tensor feature
        organized = np.zeros((width, width, features))
        for i in range(matrix.shape[0]):
            column = (i+1) % width
            row = int((i+1) / width)
            organized[row,column] = matrix[i,1:]
        
        return organized
         
        
    def __init__(self, dirname : str):
        self.dir = dirname
        self.rates = SpatialObj.get_rates(dirname)
        self.data = SpatialObj.get_time_data(os.path.join(dirname,"06RD_responder/")) 
        self.n_samples = len(self.data)
        

class GiuseppeSpatialDataLoader():
    def __init__(self, path, single_initial_cond = True):
        # each spatial object contains a set of parameters 
        self.path = path
        self.spatialData = GiuseppeSpatialDataLoader.get_spatial_objs(path)
        
    def get_spatial_objs(path : str, single_initial_cond = True):
        parameter_dirs = [os.path.join(path,x) for x in os.listdir(path)]
        spatialObjs = []
        for dir in parameter_dirs:
            print(dir)
            if dir != path and "parameter" in dir:
                obj = SpatialObj(dir)
                spatialObjs.append(obj)
        return spatialObjs    
        


parent_data_dir = "../../share/Giuseppe_John/training_data_with_dump_files_ParamSweep_11-03-2023_30_samples/"


test = GiuseppeSpatialDataLoader(parent_data_dir)
print(test.spatialData[0].rates)
print(test.spatialData[1].rates)

print(test.spatialData[0].data[0][0])

# Currently data looks something a little like this, 
# parent_data_dir
#    parameters_xyz_
#       06rd thing
#           -t=0 matrix file (with need to calculate coordinates and convert into a separate matrix)
#           -t=1 matrix file
#           


# need to figure out how to convert it all into one ginormous data .csv file for sake of being used to the format (??)
# 