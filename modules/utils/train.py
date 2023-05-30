import os, psutil
import torch 
import torch.nn.functional as F
import torch.nn as nn
import gc
from torch_geometric.nn import GCNConv, global_mean_pool
from scipy import spatial
from modules.models.spatial import *
from torch.cuda.amp import autocast


class SpatialModel():
    def train_moments(data, nEpochs, n_inputs, n_outputs, n_rates, path = ""):
        model = GCNComplexMoments(n_inputs=n_inputs, n_rates=n_rates, hidden_channels=32, n_outputs=n_outputs)
        model.train()
        model = model.double()
        optimizer = torch.optim.AdamW(model.parameters())
        criterion = torch.nn.MSELoss()

        device = ""
        scaler = ""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            using_gpu = True
            scaler = torch.cuda.amp.GradScaler()
        else:
            device = torch.device("cpu")
            using_gpu = False
            
        model = model.to(device)
        criterion = criterion.to(device)

        dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=True) 
        for epoch in range(nEpochs):
            loss_per_epoch = 0
            for rates, output_graph in dataloader:
                optimizer.zero_grad()
                
                if using_gpu:
                    loss = 0
                    with autocast():
                        out = model(data.initial_graph.to(device), data.edges.to(device), rates.to(device))
                        loss = criterion(out, output_graph.to(device))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else: 
                    loss.backward()
                    loss_per_epoch += loss.item()
                    optimizer.step()
                    
                loss_per_epoch += loss.cpu().item()
                if using_gpu:
                    loss = None 
                    
            if using_gpu and epoch  % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            if epoch % 1 == 0:
                print("mem:", torch.cuda.memory_allocated(), " cached:", torch.cuda.memory_cached())
                print("Epoch:", epoch, " Loss:", loss_per_epoch)   
        if len(path) > 0:
            torch.save(model.state_dict(), path)
            
        return model


    