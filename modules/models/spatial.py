import os, psutil
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from scipy import spatial
from modules.utils.graph import *
from modules.data.spatial import *
class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(n_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, n_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    
class EncoderLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class GCNComplex(torch.nn.Module):
    # default embedding size maybe, 64?
    def __init__(self, n_features, n_classes, hidden_channels, n_rates, embedding_size=64):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(n_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels + hidden_channels, n_classes)
        self.rates_encoder = EncoderLayer(n_rates, embedding_size, hidden_channels)

    def forward(self, graph, edge_index, rates):
        graph = self.conv1(graph, edge_index)
        graph = graph.relu()
        graph = F.dropout(graph, p=0.5, training=self.training)
        rates_rep = self.rates_encoder(rates)
        # concatenate both the graph and rates_representation and produce next
        rates_rep = rates_rep.repeat(graph.size()[0]).reshape((graph.size()[0], rates_rep.size()[0]))
        x = torch.cat((graph, rates_rep), dim=1)
        
        # convolve again and get the output u care about
        x = self.conv2(x, edge_index)
        return x

def train_giuseppe_surrogate(data_obj : GiuseppeSurrogateGraphData, nEpochs = 30, single_init_cond = True):
    model = GCNComplex(n_features=data_obj.n_features, n_classes=data_obj.n_output, hidden_channels=32, n_rates=data_obj.n_rates)
    model.train()
    model = model.double()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(nEpochs):
        loss_per_epoch = 0
        for graph in range(data_obj.length):
            optimizer.zero_grad()
            input_graph = data_obj.input_graphs
            if not single_init_cond:
                input_graph = data_obj.input_graphs[graph]
            out = model(input_graph, data_obj.edges, data_obj.rates[graph])
            loss = criterion(out, data_obj.output_graphs[graph])
            loss.backward()
            loss_per_epoch+=loss
            optimizer.step()
            
        if epoch % 10 == 0:
            print("Epoch:", epoch, " Loss:", loss_per_epoch)   
    return model     

def train_gnn(data_obj : GiuseppeSurrogateGraphData, nEpochs = 30, single_init_cond = True):
    model = GCN(n_features=data_obj.n_features, n_classes=data_obj.n_output, hidden_channels=32)
    model.train()
    model = model.double()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(nEpochs):
        loss_per_epoch = 0
        for graph in range(data_obj.length):
            optimizer.zero_grad()
            input_graph = data_obj.input_graphs
            if not single_init_cond:
                input_graph = data_obj.input_graphs[graph]
            out = model(input_graph, data_obj.edges)
            loss = criterion(out, data_obj.output_graphs[graph])
            loss.backward()
            loss_per_epoch+=loss
            optimizer.step()
            
        if epoch % 10 == 0:
            print("Epoch:", epoch, " Loss:", loss_per_epoch)   
            
    return model     


def train_giuseppe_surrogate_pkl(data : dict, nEpochs = 30, single_init_cond = True):
    model = GCNComplex(n_features=data["n_features"], n_classes=data["n_outputs"], n_rates=data["n_rates"],hidden_channels=32)
    model.train()
    model = model.double()
    optimizer = torch.optim.AdamW(model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(nEpochs):
        loss_per_epoch = 0
        for graph in range(data["n"]):
            optimizer.zero_grad()
            input_graph = data["input_graphs"]
            if not single_init_cond:
                input_graph = data["input_graphs"][graph]
            out = model(input_graph, data["edges"], data["rates"][graph])
            loss = criterion(out, data["output_graphs"][graph])
            loss.backward()
            loss_per_epoch+=loss
            optimizer.step()
            
        if epoch % 1 == 0:
            print("Epoch:", epoch, " Loss:", loss_per_epoch)   
    return model     