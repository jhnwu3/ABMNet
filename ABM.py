import mesa
import torch as tc
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from GMM import *


class ABMDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=False):
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
        if transform:
            allData = self.dframe.to_numpy()
            outputs = allData[:, self.final_input_idx:].copy()
            self.transform_mat = mahalonobis_matrix_numpy(outputs)
            transformed = np.matmul(outputs, self.transform_mat)
            self.untransformed_outputs = outputs
            allData[:, self.final_input_idx:] = transformed 
            self.dframe = pd.DataFrame(allData, columns=columns)
            print("Transformation Matrix Applied:")
            print(self.transform_mat)
        
    def __len__(self):
        return len(self.dframe)
    
    def __getitem__(self, idx):
        if tc.is_tensor(idx):
            idx = idx.tolist()
        
        dfRow = self.dframe.iloc[idx].to_numpy()
        sample = {'params': tc.tensor(dfRow[:self.final_input_idx]).double(), 'moments': tc.tensor(dfRow[self.final_input_idx:]).double()}
        
        return sample   
            

def compute_gini(model):
    agent_wealths = [agent.wealth for agent in model.schedule.agents]
    x = sorted(agent_wealths)
    N = model.num_agents
    B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * sum(x))
    return 1 + (1 / N) - 2 * B

def compute_NCC(model):
    agent_pos = [a.pos for a in model.schedule.agents]
    
class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1
        
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        
    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1
    
    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()
        
class MoneyModel(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)

        # Create agents
        for i in range(self.num_agents):
            a = MoneyAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))
            
        self.datacollector = mesa.DataCollector(
            model_reporters={"Gini": compute_gini}, agent_reporters={"Wealth": "wealth"}
        )
        
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()