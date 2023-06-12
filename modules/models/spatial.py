import os, psutil
import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
from scipy import spatial
from modules.utils.graph import *
from modules.data.spatial import *
from modules.models.simple import *
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
        self.final_out = NeuralNetwork(hidden_dim, hidden_dim*2, 3, num_classes)
        self.hidden_dim = hidden_dim

    def forward(self, x, edge_index, rates):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = torch.cat((x, self.rates_encoder(rates).repeat(x.size()[0], 1).reshape((x.size()[0], self.hidden_dim))), dim=1)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        # x = torch.cat([x[:, head_idx] for head_idx in range(x.size(1))], dim=0)
        x = self.fc(x)
        x = x.relu()
        x = global_mean_pool(x, batch=None)
        # let's do an MLP
        x = self.final_out(x)
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


#MLP heads for ViTs
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# Define the Vision Transformer model
class ViT(nn.Module):
    # Implementation of ViT goes here...
    def __init__(self, image_size, patch_size, num_classes, embed_dim, depth, num_heads, mlp_dim):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels=1, embed_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.patch_embedding.num_patches, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim)
            for _ in range(depth)
        ])
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mlp_head = MLP(embed_dim, mlp_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x + self.positional_encoding
        for transformer in self.transformer_blocks:
            x = transformer(x)
        x = x.mean(1)  # Global average pooling
        x = self.mlp_head(x)
        return x

