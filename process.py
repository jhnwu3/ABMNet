import pandas as pd 
import numpy as np


file = "data/gdag_more_more.csv"
n=30

# process every 30 rows to produce a new file for NN training.
df = pd.read_csv(file)
data = df.to_numpy()
new_mat = []
for i in range(data.shape[0]):
    if i % n == 0:
        new_mat.append(np.mean(data[i:i+n], axis=0))

new_mat = np.array(new_mat)


new_df = pd.DataFrame(new_mat, columns=['k1','k2','k3','k4','o1','o2','o3'])

new_df.to_csv('data/gdag_1300ss.csv')