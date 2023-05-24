import os
import pickle
import torch 
import numpy as np
from torch.utils.data import DataLoader, IterableDataset, Dataset
from scipy import spatial
import math

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
            file_i = os.path.join(dir, "dump.2D_model.0")
            file_t = os.path.join(dir, "dump.2D_model.1")
            if dirname != dir and os.path.isfile(file_t) and os.path.isfile(file_i):
                # dump 
                dump_i = SpatialObj.process_dump_file(file_i)
                dump_t = SpatialObj.process_dump_file(file_t)
                # translate
                data_i = SpatialObj.translate_to_img(dump_i)
                data_t = SpatialObj.translate_to_img(dump_t)
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
    #  matrix[x,y,feature]
    def translate_to_img(matrix, width=100, features=9, offset=1):
        # convert indices in far left column to 
        # get some tensor feature
        organized = np.zeros((width, width, features))
        for i in range(matrix.shape[0]):
            column = (i+1) % width
            row = int((i) / width)
            organized[row,column] = matrix[i,offset:]
        
        return organized
        
    def __init__(self, dirname : str):
        self.dir = dirname
        self.rates = SpatialObj.get_rates(dirname)
        self.data = SpatialObj.get_time_data(os.path.join(dirname,"06RD_responder/")) 
        self.n_samples = len(self.data)
        

class GiuseppeSpatialDataProcessor():
    def __init__(self, path, single_initial_cond = True):
        # each spatial object contains a set of parameters 
        self.path = path
        self.spatialData = GiuseppeSpatialDataProcessor.get_spatial_objs(path)
        
    def get_spatial_objs(path : str, single_initial_cond = True):
        parameter_dirs = [os.path.join(path,x) for x in os.listdir(path)]
        spatialObjs = []
        for dir in parameter_dirs:
            print(dir)
            if dir != path and "parameter" in dir:
                obj = SpatialObj(dir)
                spatialObjs.append(obj)
        return spatialObjs    
    
    # it will convert all of the objects in the data structure into a dictionary, and then save as a pickle file once done. 
    # Probably should've been done this way originally we can keep things clean.
    def convert_to_pickle(self, path="file.pickle"):
        dictionary = {} # every key will be the set of rate constants and every value is a list of 30 tuples of (in, out)
        for obj in self.spatialData:
            dictionary[obj.rates.tobytes()] = obj.data
        with open(path, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def print_data_statistics(self):
        for obj in self.spatialData:
            print("-----------------------------------")
            print("Rates:", obj.rates)
            print("Number of Samples:", len(obj.data))

# come back to this once we can create a pickle file/ dictionary for Pytorch Geometric
class GiuseppeSurrogateGraphData():
    def __init__(self, single_initial_cond = True):
        self.input_graphs = [] # list of torch array of all features row by row, each row a node
        self.output_graphs = [] # list of torch array with each row a node as well.
        self.edges = [] # torch array of all edges 
        self.rates = []# list of rates we care about in torch
        self.single_init = single_initial_cond
        self.n_features = 0 # feature dimension for node
        self.n_output = 0 # feature dimension for ouput node 
        self.length = 0
        self.n_rates = 0
    # lattice is a WxHxF tensor where F is the feature dimension size
    # return a python list of nodes
    def convert_lattice_to_node(lattice):
        nodes = [] # torch.zeros((lattice.shape[0] * lattice.shape[1], lattice.shape[2]))
        i = 0
        for r in range(lattice.shape[0]):
            for c in range(lattice.shape[1]):
                nodes.append(lattice[r,c])
        # bad code inbound
        nodes = np.array(nodes)
        return torch.from_numpy(nodes)
    
    def delaunay_edges_and_data(self, dictionary):
        for key in dictionary.keys():
            rates = np.frombuffer(key)
            lattice_shape = dictionary[key][0][0].shape
            
            coords = np.zeros((lattice_shape[0] * lattice_shape[1], 2))
            i = 0
            for r in range(lattice_shape[0]):
                for c in range(lattice_shape[1]):
                    # create array of 2D coordinates for triangulation
                    coords[i,0] = r
                    coords[i,1] = c
                    i+=1
            
            # now get the edges from delaunay triangulation that will be reused across all things
            tri = spatial.Delaunay(coords)
            edges = []
            for triangle in range(tri.simplices.shape[0]):
                # basically now need to go through each of simplices rows and convert them into 
                # edges are two-way, so have to re-iterate twice through.
                for coordinateIndex in range (tri.simplices.shape[1]):
                    for secondCoordinateIndex in range(coordinateIndex + 1, tri.simplices.shape[1]):
                        if coordinateIndex != secondCoordinateIndex:
                            edges.append([tri.simplices[triangle, coordinateIndex], tri.simplices[triangle, secondCoordinateIndex]]) # use same set of edges each time
            
            self.edges = torch.Tensor(edges)
            self.edges = self.edges.long()   
            self.edges = self.edges.transpose(0,1)   
            # now append all of the graphs in order with respect to the input and output data
            for sample in dictionary[key]: 
                initial_lattice = sample[0]
                final_lattice = sample[1]
                self.rates.append(torch.from_numpy(rates.copy())) # yes there will be duplicate rates, but we need to stay consistent with indexing.
                self.input_graphs.append(GiuseppeSurrogateGraphData.convert_lattice_to_node(initial_lattice))
                self.output_graphs.append(GiuseppeSurrogateGraphData.convert_lattice_to_node(final_lattice))
            self.n_features = self.input_graphs[0].size()[1]
            self.n_output = self.output_graphs[0].size()[1]
            self.length = len(self.output_graphs) 
            self.n_rates = self.rates[0].size()[0]
            
        if self.single_init: # what if we only need one of the initial conditions
            self.input_graphs = self.input_graphs[0]

    def delaunay_edges_and_spatial_correlation(self, dictionary):
        
        
    # create a pickle data structure for all the Y stuff
    # since we are not memory constrained just yet, we can simply load it on the cluster no need 
    # to do fileIO for now. 
    def save(self, path):
        print("Saving Graph Data to PKL")
        dictionary = {} # every key will be the set of rate constants and every value is a list of 30 tuples of (in, out)
        dictionary["rates"] = self.rates
        dictionary["input_graphs"] = self.input_graphs # for now every graph looks the same
        dictionary["output_graphs"] = self.output_graphs
        dictionary["edges"] = self.edges
        dictionary["n_features"] = self.n_features
        dictionary["n_outputs"] = self.n_output
        dictionary["n_rates"] = self.n_rates
        dictionary["n"] = self.length
        with open(path, 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def create_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)        
            
    def chunk_pkl(pkl_dict, parent_directory):
        # create directory structure 
        GiuseppeSurrogateGraphData.create_directory(parent_directory)
        
        # create subdirectories that categorize input, outputs, and rates, etc.
        rates_dir = os.path.join(parent_directory, "rates")
        input_graphs_dir = os.path.join(parent_directory, "input_graphs")
        output_graphs_dir = os.path.join(parent_directory, "output_graphs")
        edges_file = os.path.join(parent_directory, "edges.pt")
        metadata_file = os.path.join(parent_directory, "metadata.pickle")
        
        GiuseppeSurrogateGraphData.create_directory(rates_dir)
        GiuseppeSurrogateGraphData.create_directory(input_graphs_dir)
        GiuseppeSurrogateGraphData.create_directory(output_graphs_dir)
        torch.save(pkl_dict["edges"], edges_file)
        
        metadata = {}
        metadata["n_features"] = pkl_dict["n_features"]
        metadata["n_outputs"] = pkl_dict["n_outputs"]
        metadata["n_rates"] = pkl_dict["n_rates"]
        metadata["n"] = pkl_dict["n"]
        
        with open(metadata_file, 'wb') as handle:
            pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # now put each rate and graph with its respective name into its respective directory location, all indexed 
        # chunk by 1/3rds, so every directory has three explicit chunks
        chunk_indices = [i for i in range(0, metadata["n"], int(metadata["n"]/3))]
        for i in range(len(chunk_indices)):
            file = os.path.join(rates_dir, "rates" + str(chunk_indices[i]) + ".pickle")
            with open(file, 'wb') as handle:
                if i < len(chunk_indices) - 1:
                    pickle.dump(pkl_dict["rates"][chunk_indices[i]:chunk_indices[i+1]], handle, protocol=pickle.HIGHEST_PROTOCOL)
                else: 
                    pickle.dump(pkl_dict["rates"][chunk_indices[i]:], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # check if just one graph in there or many to create the big chonkers
        if isinstance(pkl_dict["input_graphs"], list):
            for i in range(len(chunk_indices)):
                file = os.path.join(input_graphs_dir, "graph" + str(chunk_indices[i]) + ".pickle")
                with open(file, 'wb') as handle:
                    if i < len(chunk_indices) - 1:
                        pickle.dump(pkl_dict["input_graphs"][chunk_indices[i]:chunk_indices[i+1]], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else: 
                        pickle.dump(pkl_dict["input_graphs"][chunk_indices[i]:], handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            file = os.path.join(input_graphs_dir, "graph.pt")
            torch.save(pkl_dict["input_graphs"], file)
            
        for i in range(len(chunk_indices)):
            file = os.path.join(output_graphs_dir, "graph" + str(chunk_indices[i]) + ".pickle")
            with open(file, 'wb') as handle:
                if i < len(chunk_indices) - 1:
                    pickle.dump(pkl_dict["output_graphs"][chunk_indices[i]:chunk_indices[i+1]], handle, protocol=pickle.HIGHEST_PROTOCOL)
                else: 
                    pickle.dump(pkl_dict["output_graphs"][chunk_indices[i]:], handle, protocol=pickle.HIGHEST_PROTOCOL)
        


# Cytotoxic CD8+ T Cells, Cancer, Exhausted CD8+ T Cells, Dead Cancer Cells, Ignore, Ignore, TAMs, Ignore, Ignore
class SingleInitialConditionDataset(Dataset):
    # path to a pickle file that contains a dictionary of the following variables shown below
    # [] is a list of indices of features to keep in the input and output graphs
    def __init__(self, path, channels = []):
        # Initialize your dataset here
        # Store the necessary data or file paths
        data = pickle.load(open(path, "rb"))
        self.output_graphs = data["output_graphs"]
        self.initial_graph = data["input_graphs"]
        self.edges = data["edges"]
        self.rates = data["rates"]
        self.n_outputs = data["n_outputs"] 
        self.n_inputs = data["n_features"]
        self.n_rates = data["n_rates"]
        if len(channels) > 0:
            self.n_outputs = len(channels)
            self.n_inputs = len(channels)
            self.initial_graph = self.initial_graph[:,channels]
            for i in range(len(self.output_graphs)):
                self.output_graphs[i] = self.output_graphs[i][:,channels] # this should work I believe lol.
            
        
    def __len__(self):
        # Return the total number of samples in your dataset
        return len(self.rates) # len(rates) == len(output_graphs)
    
    def __getitem__(self, index):
        # Retrieve a single item from the dataset based on the given index
        # Return a tuple (input, target) or dictionary {'input': input, 'target': target}
        # The input and target can be tensors, NumPy arrays, or other data types
        return self.rates[index], self.output_graphs[index]


# come back to modify this if we find out it takes too long to train and we need to leverage more power
class MyIterableDataset(IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(range(iter_start, iter_end))