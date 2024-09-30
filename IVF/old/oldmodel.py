import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, heads=4, num_layers=6):
        super(GNNModel, self).__init__()

        self.layers = nn.ModuleList()

        self.layers.append(GCNConv(in_channels, hidden_channels))
        self.layers.append(nn.ReLU())
        
        for i in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * (2 ** (2*i)), hidden_channels * (2 ** (2*i + 1)), heads=heads, concat=False))
            self.layers.append(nn.ReLU())
            self.layers.append(GCNConv(hidden_channels * (2 ** (2*i + 1)), hidden_channels * (2 ** (2*i + 2))))
            self.layers.append(nn.ReLU())

        self.layers.append(GATConv(hidden_channels * (2 ** (2*(num_layers - 2))), hidden_channels *  (2 ** (2*(num_layers - 1))), heads=heads, concat=False))
        self.layers.append(nn.ReLU())

        outchannels = hidden_channels * (2 ** (2*(num_layers - 1)))


        # FC layer to predict edge probabilities
        self.edge_pred = nn.Sequential(
                            nn.Linear(2*outchannels, 4*outchannels),
                            nn.ReLU(),
                            nn.Linear(4*outchannels, 1),
                            nn.Sigmoid()
                         )

    def forward(self, x, edge_index):

        for layer in self.layers:
            if isinstance(layer, MessagePassing):
                x = layer(x, edge_index)  # Apply GCN or GAT layer
            else:
                x = layer(x)

        edge_start = x[edge_index[0]]
        edge_end   = x[edge_index[1]]
        edge_feats = torch.cat([edge_start, edge_end], dim=1)

        edge_probs = self.edge_pred(edge_feats)
        return edge_probs

