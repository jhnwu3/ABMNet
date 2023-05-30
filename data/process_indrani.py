import pandas as pd 
import numpy as np
import os 
# from modules.data.spatial import *
def write_array_to_csv(array, column_labels, file_path):
    # Create a DataFrame from the array and column labels
    df = pd.DataFrame(data=array, columns=column_labels)

    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

parent_dir = "data/John_Indrani_data/data"
parameter_dirs = [os.path.join(parent_dir,x) for x in os.listdir(parent_dir)]
print(parameter_dirs)

rates = []
output = []
for dir in parameter_dirs:
    if dir != parent_dir:
        data = np.loadtxt(dir)
        print(data.shape)
        
        rates.append(data[0])
        output.append(data[1:,1])
# exit(0)
rates = np.array(rates)
output = np.array(output)

print(rates.shape)
print(output.shape)

# for simplicity, we'll just do counts for now, and save into a specific dataset
output_file = "data/static/indrani.csv"
time_point = 300
# matrix with inputs and outputs
labels = []
for i in range(1, rates.shape[1] + 1):
    labels.append("k" + str(i))
labels.append("o1")

output_data = np.zeros((rates.shape[0], rates.shape[1] + 1))
output_data[:,:2] = rates
output_data[:,2] = output[:,time_point]   


write_array_to_csv(output_data, labels, output_file) 