import pandas as pd 
import numpy as np


file = 'data/NL6P.csv'
nParams = 6
nOutputs = 27
df = pd.read_csv(file)
data = df.to_numpy()
indices_changing = np.array([0,2])
const_val = 0.4
initial_moments = np.loadtxt('data/time_series/initial_6pro_moments.txt',delimiter=',') 
print(initial_moments)
# exit(0)
to_save = []



# For when you want only means.
# nIn = nParams
# nOutputs = 6
# to_save = data[:,:nIn+nOutputs]


# For when you want input == output 33 to 27
# nIn = nParams + nOutputs
# for r in range(data.shape[0]):
#     vec = np.zeros(nIn+nOutputs)
#     vec[:nParams] = data[r,:nParams]
#     vec[nParams:nIn] = initial_moments
#     vec[nIn:nIn+nOutputs] = data[r,nParams:]
#     to_save.append(vec)

# For when you wanna limit the data sizes.
nIn = nParams
for row in range(data.shape[0]):
    save_row = True
    for c in range(nParams):
        if c not in indices_changing:
            # print(c)
            save_row = save_row and data[row,c] == const_val 
    if save_row:
        to_save.append(data[row,:])

to_save = np.array(to_save)


cols = []
for i in range(nIn):
    cols.append('k' + str(i + 1))

for o in range(nOutputs):
    cols.append('o' + str(o+1))
print(to_save.shape)
new_df = pd.DataFrame(to_save, columns=cols)
new_df.to_csv('data/NL6_2k.csv')