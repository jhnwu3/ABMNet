import os, psutil
import torch 
import torch.nn.functional as F
import torch.nn as nn
import gc
from torch_geometric.nn import GCNConv, global_mean_pool
from scipy import spatial
from modules.models.spatial import *
from torch.cuda.amp import autocast
from modules.models.temporal import *
from modules.data.temporal import *
import time 
class SpatialModel():
    def train_gcn(data, nEpochs, n_inputs, n_outputs, n_rates, initial_graph, edges, hidden_channels, path = ""):
        model = GCNComplexMoments(n_inputs=n_inputs, n_rates=n_rates, hidden_channels=hidden_channels,
                                  n_outputs=n_outputs)
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
                        out = model(initial_graph.to(device), edges.to(device), rates.to(device))
                        loss = criterion(out.squeeze(), output_graph.squeeze().to(device))
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
            #     print("mem:", torch.cuda.memory_allocated(), " cached:", torch.cuda.memory_reserved())
                print("Epoch:", epoch, " Loss:", loss_per_epoch)   
        if len(path) > 0:
            torch.save(model.state_dict(), path)
            
        # return both a model and the device used to train it.
        return model, device
    
    def train_gat(data, nEpochs, n_inputs, n_outputs, n_rates, initial_graph, edges, hidden_channels, path = ""):
        model = GATComplex(input_dim=n_inputs, n_rates=n_rates, hidden_dim=hidden_channels,
                                  num_classes=n_outputs)
        model.train()
        model = model.double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
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
                        out = model(initial_graph.to(device), edges.to(device), rates.to(device))
                        loss = criterion(out.squeeze(), output_graph.squeeze().to(device))
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
            #     print("mem:", torch.cuda.memory_allocated(), " cached:", torch.cuda.memory_reserved())
                print("Epoch:", epoch, " Loss:", loss_per_epoch)   
        if len(path) > 0:
            torch.save(model.state_dict(), path)
            
        # return both a model and the device used to train it.
        return model, device


def train_temporal_model(data : TemporalDataset, hidden_size=256, lr=0.001, n_epochs=20, n_layers=5, path=""):
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
            
    model = TemporalComplexModel(input_size=data.input_size, hidden_dim=hidden_size, n_layers=n_layers, n_rates = data.n_rates)
    model = model.to(device).float()
    
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_size = int(0.1 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=None, shuffle=True) 
    epoch_start = time.time()
    for epoch in range(n_epochs):
        loss_per_epoch = 0
        hidden = (torch.zeros(n_layers, hidden_size).detach(), torch.zeros(n_layers, hidden_size).detach())
        for rates, input, output in dataloader:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            out, hidden = model(input.to(device).float(), (hidden[0].detach().to(device), hidden[1].detach().to(device)), rates.to(device).float())
            loss = criterion(out.squeeze().float(), output.to(device).float())
            loss_per_epoch += loss
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
        if(epoch % 10 == 0):
            print("Epoch:", epoch, " loss:", loss_per_epoch, " in Time:", time.time() - epoch_start)
            epoch_start = time.time()
            
    if len(path) > 0:
        torch.save(model.state_dict(), path)
        
    return model, device, test_data



    