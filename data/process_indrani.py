import pandas as pd 
import numpy as np
import os 
import torch
import pickle
# from modules.data.spatial import *
def write_array_to_csv(array, column_labels, file_path):
    # Create a DataFrame from the array and column labels
    df = pd.DataFrame(data=array, columns=column_labels)

    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)



parent_dir = "data/John_Indrani_data/zeta_Ca_signal/training_kon_koff"
output_path = "data/time_series/indrani_zeta_ca_no_zeroes.pickle"
# parent_dir = "data/John_Indrani_data/zeta/training_kon_koff"
parameter_dirs = [os.path.join(parent_dir,x) for x in os.listdir(parent_dir)]
print(parameter_dirs)

rates = []
output = []
# keep one vector of the time steps too just in case.
times = []
for dir in parameter_dirs:
    if dir != parent_dir and ".txt" in dir and "train" in dir:
        data = np.loadtxt(dir)
        # print(data.shape)
        rates.append(data[0])
        output.append(data[1:,1])
        
        if len(times) < 1:
            times = data[1:,0] # keep for plotting


# save into tensors and a dictionary
tosave = {}
rates_tensors = []
output_tensors  = []
for i in range(len(rates)):
    if np.sum(output[i]) > 0:
        rates_tensors.append(torch.from_numpy(rates[i]))
        output_tensors.append(torch.from_numpy(output[i]))    
        
tosave["rates"] = rates_tensors
tosave["outputs"] = output_tensors
tosave["time_points"] = times
with open(output_path, 'wb') as handle:
    pickle.dump(tosave, handle, protocol=pickle.HIGHEST_PROTOCOL)


rates = np.array(rates)
output = np.array(output)
# now let us create a dictionary for the dataset where each entity in the list is some time-series evolution 
print(rates.shape)
print(output.shape)

# # for simplicity, we'll just do counts for now, and save into a specific dataset
# output_file = "data/static/indrani_t400.csv"
# time_point = 400
# # matrix with inputs and outputs
# labels = []
# for i in range(1, rates.shape[1] + 1):
#     labels.append("k" + str(i))
# labels.append("o1")

# output_data = np.zeros((rates.shape[0], rates.shape[1] + 1))
# output_data[:,:2] = rates
# output_data[:,2] = output[:,time_point]   


# write_array_to_csv(output_data, labels, output_file) 