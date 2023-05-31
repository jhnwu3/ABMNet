import os, psutil
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from scipy import spatial
from modules.utils.graph import *
from modules.data.spatial import *
from torch_geometric.nn import GATConv

class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes, hidden_channels=8):
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

class GCNComplexMoments(nn.Module):
    def __init__(self, n_inputs, hidden_channels, n_rates, embedding_size=8, n_outputs=2):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(n_inputs, hidden_channels)
        self.rates_encoder = EncoderLayer(n_rates, embedding_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels + hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.hidden = nn.Linear(hidden_channels, hidden_channels*2)
        self.hidden2 = nn.Linear(hidden_channels*2, hidden_channels)
        self.final = nn.Linear(hidden_channels, n_outputs)
        
        
    def forward(self, graph, edge_index, rates):
        graph = self.conv1(graph, edge_index)
        graph = graph.relu()
        graph = F.dropout(graph, p=0.5, training=self.training)
        rates_rep = self.rates_encoder(rates)
        # concatenate both the graph and rates_representation and produce next
        graph = torch.cat((graph, self.rates_encoder(rates).repeat(graph.size()[0]).reshape((graph.size()[0], rates_rep.size()[0]))), dim=1)
        graph = self.conv2(graph, edge_index)
        graph = self.conv3(graph, edge_index)
        # get the average for final prediction.
        graph = global_mean_pool(graph, batch=None)
        # convolve again and get the output u care about
        graph = self.hidden(graph)
        graph = F.relu(graph)
        graph = self.hidden2(graph)
        graph = F.relu(graph)
        graph = self.final(graph)
        return graph


# Define the Graph Attention Network (GAT) model
class GATComplex(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_rates, embedding_size=8, num_heads=4):
        super(GATComplex, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.rates_encoder = EncoderLayer(n_rates, embedding_size, hidden_dim)
        self.conv2 = GATConv(hidden_dim * num_heads + hidden_dim, hidden_dim, heads=num_heads)
        self.fc = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.final_out = nn.Linear(hidden_dim, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, rates):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        print(x.size())
        x = torch.cat((x, self.rates_encoder(rates).repeat(x.size()[0], 1).reshape((x.size()[0], self.hidden_dim))), dim=1)
        print(x.size())
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        print("before error:",x.size())
        x = torch.cat([x[:, head_idx] for head_idx in range(x.size(1))], dim=2)
        x = self.fc(x)
        x = x.relu()
        x = global_mean_pool(x, batch=None)
        self.final_out(x)
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
        graph = torch.cat((graph, self.rates_encoder(rates).repeat(graph.size()[0]).reshape((graph.size()[0], rates_rep.size()[0]))), dim=1)
        
        # convolve again and get the output u care about
        graph = self.conv2(graph, edge_index)
        return graph

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