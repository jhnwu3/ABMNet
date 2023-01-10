import torch as tc 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
import time
import mesa
from ABM import *
from NN import *

if __name__ == '__main__':
    
    # Get ABM Data from ABM File
    ABM_Model = MoneyModel(100)
    for i in range(10):
        ABM_Model.step()
    agent_wealth = np.array([a.wealth for a in ABM_Model.schedule.agents])
    print(agent_wealth)
    
    
    # Train Neural Network.
    
    
    # Validate