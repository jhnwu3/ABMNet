import pandas as pd 
import numpy as np
import os 
# from modules.data.spatial import *


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