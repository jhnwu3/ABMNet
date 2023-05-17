import numpy as np
import torch as tc 
# takes data and outputs a mahalonobis matrix 1/variance(i) where i is the vector of data of a specific column
def mahalonobis_matrix_torch(data):
    variances = tc.var(data, dim=0)
    inversed = 1 / variances 
    return tc.diag(inversed)

def mahalonobis_matrix_numpy(data):
    variances = np.var(data, axis=0)
    inversed = 1 /variances 
    return np.diag(inversed)

