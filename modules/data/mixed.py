import torch as tc
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from modules.utils.gmm import *


class ABMDataset(Dataset):
    def __init__(self, csv_file, root_dir=None, standardize=False, norm_out = False):
        self.dframe = pd.read_csv(csv_file)
        self.root = root_dir 
        columns = self.dframe.columns 
        self.final_input_idx = 0 
        # initialize cut off point between inputs and outputs.
        for column in columns: 
            if 'k' in column: 
                self.final_input_idx += 1
        self.transform_mat = None
        self.untransformed_outputs = None # if transform 
        self.output_mins = []
        self.output_maxes = []
        self.input_means = []
        self.input_stds = []
        self.input_dim = self.final_input_idx 
        # just to see what happens in gdags data by normalizing parameters
        allData = self.dframe.to_numpy()
        self.out_dim = allData.shape[1] - self.final_input_idx 
        if standardize:
            inputs = allData[:, :self.final_input_idx].copy()
            self.input_means = inputs.mean(axis=0)
            self.input_stds = inputs.std(axis=0)
            ret_inputs = inputs - inputs.mean(axis=0)
            ret_inputs = ret_inputs / inputs.std(axis=0)
            allData[:, :self.final_input_idx] = ret_inputs
           
            print("Standardization to Input Parameters Applied")
            print("Original Mean Inputs:", self.input_means)
            print("Original Stds:", self.input_stds)
            print('New Average Input Value:', allData[:,:self.final_input_idx].mean(axis=0))
            print('New Std Input Value:', allData[:,:self.final_input_idx].std(axis=0))
            print('max:', np.max(inputs))
        
        if norm_out:
            outputs = allData[:, self.final_input_idx:].copy()
            for c in range(outputs.shape[1]):
                self.output_mins.append(outputs[:,c].min())
                self.output_maxes.append(outputs[:,c].max())
                outputs[:,c] = (outputs[:,c] - outputs[:,c].min()) / (outputs[:,c].max() - outputs[:,c].min())
                
            allData[:, self.final_input_idx:] = outputs 
            print("Normalization of Outputs")
            print("New Max:", np.max(outputs))
            print("New Min:", np.min(outputs))
            self.output_mins = np.array(self.output_mins)
            self.output_maxes = np.array(self.output_maxes)
            
        self.dframe = pd.DataFrame(allData, columns=columns)
        
    def __len__(self):
        return len(self.dframe)
    
    def __getitem__(self, idx):
        if tc.is_tensor(idx):
            idx = idx.tolist()
        
        dfRow = self.dframe.iloc[idx].to_numpy()
        #sample = {'params': tc.tensor(dfRow[:self.final_input_idx]).double(), 'moments': tc.tensor(dfRow[self.final_input_idx:]).double()}
        
        return tc.tensor(dfRow[:self.final_input_idx]).double(), tc.tensor(dfRow[self.final_input_idx:]).double()


class StaticDataset(Dataset):
    def __init__(self, rates, outputs):
        self.rates = rates
        self.outputs = outputs

        
    def __len__(self):
        return len(self.rates)
    
    def __getitem__(self, idx):
        return self.rates[idx], self.outputs[idx]
    
    def write_to_csv(self, file_path):
        rates = []
        output = []
        for rate in self.rates:
            rates.append(rate.numpy())
        for out in self.outputs:
            output.append(out.numpy())
            
        rates = np.array(rates)
        output = np.array(output)
        labels = []
        for i in range(1, rates.shape[1] + 1):
            labels.append("k" + str(i))

        for o in range(1, output.shape[1] + 1):
            labels.append("o" + str(o))
        array = np.zeros((rates.shape[0], rates.shape[1] + output.shape[1]))
        array[:,:rates.shape[1]] = rates
        array[:,rates.shape[1]:] = output   
        # Create a DataFrame from the array and column labels
        df = pd.DataFrame(data=array, columns=labels)

        # Write the DataFrame to a CSV file
        df.to_csv(file_path, index=False)
        

class TimeDataset(Dataset):
    def __init__(self, csv_file, root_dir, standardize=False, norm_out = False):
        self.dframe = pd.read_csv(csv_file)
        self.root = root_dir 
     
        self.final_input_idx = 0 
        # initialize cut off point between inputs and outputs.
        columns = self.dframe.columns 
        for column in columns: 
            if 'k' in column: # we use k for inputs
                self.final_input_idx += 1
                
        # t is the last column always
        self.output_mins = []
        self.output_maxes = []
        self.input_means = []
        self.input_stds = []
        
        # just to see what happens in gdags data by normalizing parameters
        allData = self.dframe.to_numpy()
        self.times = np.unique(allData[:,-1])
        
        nSamples = int(allData.shape[0] / self.times.shape[0])
        self.dmat = np.zeros((nSamples, len(self.times), allData.shape[1] - 1))
        
        current = 0
        # samples x times x moments
        for t in range(len(self.times)):
            for s in range(nSamples):
                self.dmat[s,t,:] = allData[current,:-1]                
                current+=1
        
        if standardize:
            # means and stds ~ time x moment
            for t in range(len(self.times)):
                inputs = self.dmat[:, t, :self.final_input_idx].copy()
                self.input_means.append(inputs.mean(axis=0))
                self.input_stds.append(inputs.std(axis=0))
                ret_inputs = inputs - inputs.mean(axis=0)
                ret_inputs = ret_inputs / inputs.std(axis=0)
                self.dmat[:,t, :self.final_input_idx] = ret_inputs
            
                print("Standardization to Input Parameters Applied For Time ",self.times[t])
                print('New Average Input Value:',inputs.mean(axis=0))
                print('New Std Input Value:',inputs.std(axis=0))
                print('max:', np.max(inputs))
            self.input_means = np.array(self.input_means)
            self.input_stds = np.array(self.input_stds)
            
            # normalized values ~ time x moment
        if norm_out:
            for t in range(len(self.times)):
                outputs = self.dmat[:,t, self.final_input_idx:].copy()
                minsPerTime = []
                maxesPerTime = []
                for c in range(outputs.shape[1]):
                    minsPerTime.append(outputs[:,c].min())
                    maxesPerTime.append(outputs[:,c].max())
                    outputs[:,c] = (outputs[:,c] - outputs[:,c].min()) / (outputs[:,c].max() - outputs[:,c].min())
                    
                self.dmat[:,t, self.final_input_idx:] = outputs 
                print("Normalization of Outputs at time", self.times[t])
                print("New Max:", np.max(outputs))
                print("New Min:", np.min(outputs))
                
                self.output_mins.append(np.array(minsPerTime))
                self.output_maxes.append(np.array(maxesPerTime))
            
            self.output_mins = np.array(self.output_mins)
            self.output_maxes = np.array(self.output_maxes)

    def __len__(self):
        return self.dmat.shape[0]

    # returns a time series of a sample
    def __getitem__(self, idx):
        if tc.is_tensor(idx):
            idx = idx.tolist()
        
        dfRow = self.dmat[idx,:,:]
        #sample = {'params': tc.tensor(dfRow[:self.final_input_idx]).double(), 'moments': tc.tensor(dfRow[self.final_input_idx:]).double()}
        
        return tc.tensor(dfRow[:,:self.final_input_idx]).double(), tc.tensor(dfRow[:,self.final_input_idx:]).double()




# for testing purposes
if __name__ == "__main__":
    tdata = TimeDataset(csv_file="data/time_series/l3p.csv",root_dir="data/time_series" ,standardize=False, norm_out=True)
    
    input, output = tdata[0]
    print(input.size())
    print(output.size())
    print(len(tdata))
    print(input)



