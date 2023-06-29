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
            torch.save(model, path)
            
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
            torch.save(model, path)
            
        # return both a model and the device used to train it.
        return model, device


def train_temporal_model(data, input_size : int, n_rates : int, hidden_size : int, lr : float, n_epochs : int, n_layers : int, path="", batch_size = None, early_stopping = True):
    device = ""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
            
    model = TemporalComplexModel(input_size=input_size, hidden_dim=hidden_size, n_layers=n_layers, n_rates = n_rates)
    model = model.to(device).float()
    model.train()
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # train_size = int(0.1 * len(data))
    # test_size = len(data) - train_size
    # train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size])
    best_training_loss = 30
    counter = 0
    if batch_size is None:
        dataloader = torch.utils.data.DataLoader(data, batch_size=None, shuffle=True) 
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
            if(epoch % 5 == 0):
                print("Epoch:", epoch, " loss:", loss_per_epoch, " in Time:", time.time() - epoch_start)
                epoch_start = time.time()
    else: 
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True) 
        epoch_start = time.time()
        for epoch in range(n_epochs):
            loss_per_epoch = 0
            h_0 = torch.zeros(n_layers, batch_size, hidden_size)
            c_0 = torch.zeros(n_layers, batch_size, hidden_size)
            
            # Initialize the LSTM hidden state
            lstm_hidden = (h_0.to(device), c_0.to(device))
            # hidden = (torch.zeros(n_layers, hidden_size).detach(), torch.zeros(n_layers, hidden_size).detach())
            for rates, input, output in dataloader:
                optimizer.zero_grad() # Clears existing gradients from previous epoch
                if len(input.size()) > 3:
                    input = input.squeeze()
                    output = output.squeeze()
                
                out, lstm_hidden = model(input.to(device).float(), (lstm_hidden[0][:,:input.size()[0],:].detach().contiguous(), lstm_hidden[1][:,:input.size()[0],:].detach().contiguous()), rates.to(device).float())
                loss = criterion(out.float(), output.to(device).float())
                loss_per_epoch += loss
                loss.backward() # Does backpropagation and calculates gradients
                optimizer.step() # Updates the weights accordingly
            if(epoch % 5 == 0):
                print("Epoch:", epoch, " loss:", loss_per_epoch, " in Time:", time.time() - epoch_start)
                epoch_start = time.time()
                counter += 1
            if loss_per_epoch < best_training_loss:
                best_training_loss = loss_per_epoch
                counter = 0
                
            if counter > 5: 
                break
                
                
                
    if len(path) > 0:
        torch.save(model, path)
        
    return model, device

# train stuff
def train_nn(dataset, input_size, hidden_size, depth, output_size, nEpochs, use_gpu = False, batch_size = 512):
    
    model = NeuralNetwork(input_size, hidden_size, depth, output_size).double()
    optimizer = optim.AdamW(model.parameters(),lr=0.0001)
    criterion = nn.MSELoss()
    
    if tc.cuda.is_available() and use_gpu:
        device = tc.device("cuda")
        model = model.cuda()
        criterion = criterion.cuda()
        using_gpu = True
    else:
        device = tc.device("cpu")
        using_gpu = False

    print(f"Using GPU: {using_gpu}")
    print(f"Batch Size: {batch_size}")
    model.train()
    epoch_start = time.time()
    
    # enable shuffling so we can 
    loader = tc.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  
    for epoch in range(nEpochs):
        loss_this_epoch = 0
           
        for input, output in loader:
            optimizer.zero_grad()
            loss = 0
            prediction = model.forward(input.to(device))
            loss += criterion(prediction.squeeze(), output.squeeze().to(device))
            loss_this_epoch += loss.item() 
            loss.backward()
            optimizer.step()
            # print('Memory usage: %s (kb)', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if epoch % 10 == 0:
            print(repr(f"Finished epoch {epoch} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
            epoch_start = time.time()
            
    return model  


def train_res_nn(dataset : ABMDataset, input_size, hidden_size, depth, output_size, nEpochs, use_gpu = False):
    
    model = ResidualNN(input_size, hidden_size, depth, output_size).double()
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.MSELoss()
    
    if tc.cuda.is_available() and use_gpu:
        device = tc.device("cuda")
        model = model.cuda()
        criterion = criterion.cuda()
        using_gpu = True
    else:
        device = tc.device("cpu")
        using_gpu = False

    print(f"Using GPU: {using_gpu}")
    model.train()
    epoch_start = time.time()
    loader = tc.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)
    for epoch in range(nEpochs):
        
        loss_this_epoch = 0
        for input, output in loader:
            optimizer.zero_grad()
            loss = 0
            # sample = dataset[ex]
            # input = sample[0] # no more keys, justt values input is first index, output is second
            # output = sample[1] 
            prediction = model.forward(input.to(device))
            loss += criterion(prediction.squeeze(), output.squeeze().to(device))
            loss_this_epoch += loss.item() 
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print(repr(f"Finished epoch {epoch} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
            epoch_start = time.time()
            
#     return model  


# def train_rnn(dataset : ABMDataset, input_size, hidden_size, depth, output_size, nEpochs, use_gpu = False):
#     model = RecurrentNN(input_size, hidden_size, depth, output_size).double()
#     optimizer = optim.AdamW(model.parameters())
#     criterion = nn.MSELoss()
    
#     if tc.cuda.is_available() and use_gpu:
#         device = tc.device("cuda")
#         model = model.cuda()
#         criterion = criterion.cuda()
#         using_gpu = True
#     else:
#         device = tc.device("cpu")
#         using_gpu = False

#     print(f"Using GPU: {using_gpu}")
#     model.train()
#     epoch_start = time.time()
#     for epoch in range(nEpochs):
        
#         loss_this_epoch = 0
#         for s in range(len(dataset)):
#             optimizer.zero_grad()
#             loss = 0
            
#             # start with first sample and then iterate through each time step. 
            
#             # sample = dataset[ex]
#             # input = sample[0] # no more keys, justt values input is first index, output is second
#             # output = sample[1] 
#             prediction = model.forward(input.to(device))
#             loss += criterion(prediction.squeeze(), output.squeeze().to(device))
#             loss_this_epoch += loss.item() 
#             loss.backward()
#             optimizer.step()
            
#         if epoch % 10 == 0:
#             print(repr(f"Finished epoch {epoch} with loss {loss_this_epoch} in time {time.time() - epoch_start}"))
#             epoch_start = time.time()
            
    # return model  


# def train_giuseppe_surrogate(data_obj : GiuseppeSurrogateGraphData, nEpochs = 30, single_init_cond = True):
#     model = GCNComplex(n_features=data_obj.n_features, n_classes=data_obj.n_output, hidden_channels=32, n_rates=data_obj.n_rates)
#     model.train()
#     model = model.double()
#     optimizer = torch.optim.AdamW(model.parameters())
#     criterion = torch.nn.MSELoss()
#     for epoch in range(nEpochs):
#         loss_per_epoch = 0
#         for graph in range(data_obj.length):
#             optimizer.zero_grad()
#             input_graph = data_obj.input_graphs
#             if not single_init_cond:
#                 input_graph = data_obj.input_graphs[graph]
#             out = model(input_graph, data_obj.edges, data_obj.rates[graph])
#             loss = criterion(out, data_obj.output_graphs[graph])
#             loss.backward()
#             loss_per_epoch+= float(loss)
#             optimizer.step()
            
#         if epoch % 10 == 0:
#             print("Epoch:", epoch, " Loss:", loss_per_epoch)   
#     return model     

# def train_gnn(data_obj : GiuseppeSurrogateGraphData, nEpochs = 30, single_init_cond = True):
#     model = GCN(n_features=data_obj.n_features, n_classes=data_obj.n_output, hidden_channels=32)
#     model.train()
#     model = model.double()
#     optimizer = torch.optim.AdamW(model.parameters())
#     criterion = torch.nn.MSELoss()
#     for epoch in range(nEpochs):
#         loss_per_epoch = 0
#         for graph in range(data_obj.length):
#             optimizer.zero_grad()
#             input_graph = data_obj.input_graphs
#             if not single_init_cond:
#                 input_graph = data_obj.input_graphs[graph]
#             out = model(input_graph, data_obj.edges)
#             loss = criterion(out, data_obj.output_graphs[graph])
#             loss.backward()
#             loss_per_epoch+=loss
#             optimizer.step()
            
#         if epoch % 10 == 0:
#             print("Epoch:", epoch, " Loss:", loss_per_epoch)   
            
#     return model     


# def train_giuseppe_surrogate_pkl(data : dict, nEpochs = 30, single_init_cond = True):
#     model = GCNComplex(n_features=data["n_features"], n_classes=data["n_outputs"], n_rates=data["n_rates"],hidden_channels=32)
#     model.train()
#     model = model.double()
#     optimizer = torch.optim.AdamW(model.parameters())
#     criterion = torch.nn.MSELoss()
#     for epoch in range(nEpochs):
#         loss_per_epoch = 0
#         for graph in range(data["n"]):
#             optimizer.zero_grad()
#             input_graph = data["input_graphs"]
#             if not single_init_cond:
#                 input_graph = data["input_graphs"][graph]
#             out = model(input_graph, data["edges"], data["rates"][graph])
#             loss = criterion(out, data["output_graphs"][graph])
#             loss.backward()
#             loss_per_epoch+=loss
#             optimizer.step()
            
#         if epoch % 1 == 0:
#             print("Epoch:", epoch, " Loss:", loss_per_epoch)   
#     return model     