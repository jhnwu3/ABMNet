import pandas as pd 
import numpy as np


file = "data/gdag_more_more.csv"
n=30
nCovariances = 3
nParams = 4
nOutputs = 3 + nCovariances # nSpecies + nCovariances you want

# process every 30 rows to produce a new file for NN training.
df = pd.read_csv(file)
data = df.to_numpy()
new_mat = []
for i in range(data.shape[0]):
    if i % n == 0:
        new_vec = np.zeros((data.shape[1] + nCovariances))
        new_vec[:data.shape[1]] = np.mean(data[i:i+n], axis=0)
        # get covariances
        cov_mat = np.cov(data[i:i+n,4:].transpose())
        covs = np.triu_indices(cov_mat.shape[0], 1) 
        # for r in range(covs.shape[0] - 1):
        #     for c in range(1,covs.shape[1]):
        #         if r != c:
        # print(cov_mat[covs])      
        # exit(0)
        new_vec[data.shape[1]:] = cov_mat[covs]
        new_mat.append(new_vec)

new_mat = np.array(new_mat)

# once data is collected, compute covariances
# print(new_mat.shape)
# # print(np.cov(new_mat.transpose()))
# print(np.cov(data[:30,4:].transpose()))

# technically k5 will change to some other type of inputs for another type of dataset
cols = []
for i in range(nParams):
    cols.append('k' + str(i + 1))

for o in range(nOutputs):
    cols.append('o' + str(o+1))

new_df = pd.DataFrame(new_mat, columns=cols)

new_df.to_csv('data/gdag_1300ss_covs.csv')