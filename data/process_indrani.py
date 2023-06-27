import pandas as pd 
import numpy as np
import os 
import torch
import pickle
import shutil
# from modules.data.spatial import *
def write_array_to_csv(array, column_labels, file_path):
    # Create a DataFrame from the array and column labels
    df = pd.DataFrame(data=array, columns=column_labels)

    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

def get_data_inhomogenous(filename):
    # filename = 'your_file.txt'
    delimiter = '\t'  # Adjust this according to your file's format
    placeholder_value = np.nan  # Placeholder for missing values

    # Step 1: Read the file as a list of strings
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Step 2: Determine the maximum number of columns
    max_columns = max(len(line.split(delimiter)) for line in lines)

    # Step 4: Create a numpy array filled with the placeholder value
    data = np.empty((len(lines), max_columns))
    data.fill(placeholder_value)

    # Step 5: Iterate over the lines and fill the numpy array
    for i, line in enumerate(lines):
        values = line.strip().split(delimiter)
        data[i, :len(values)] = values

    # Print the resulting numpy array
    print(data.shape) 
    return data   

parent_dir = "data/John_Indrani_data/zeta_Ca_signal/training_kon_koff"
output_path = "data/time_series/indrani_zeta_ca_no_zeroes.pickle"
# parent_dir = "data/John_Indrani_data/zeta/training_kon_koff"
parameter_dirs = [os.path.join(parent_dir,x) for x in os.listdir(parent_dir)]
print(parameter_dirs)

rates = []
output = []
# keep one vector of the time steps too just in case.
times = []
destination_dir = "data/zeta_Ca_signal/training_data_1000"
for dir in parameter_dirs:
    if dir != parent_dir and ".txt" in dir and "train" in dir:
        data = get_data_inhomogenous(dir)
        # print(data.shape)
        rates.append(data[0])
        output.append(data[1:,1:3])
        
        if len(times) < 1:
            times = data[1:,0] # keep for plotting
            
        shutil.move(dir, destination_dir)


# save into tensors and a dictionary
tosave = {}
rates_tensors = []
output_tensors  = []
for i in range(len(rates)):
    if np.sum(output[i]) > 0:
        print("rates:",rates[i].shape)
        rates_tensors.append(torch.from_numpy(rates[i]))
        print("output:", output[i].shape)
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