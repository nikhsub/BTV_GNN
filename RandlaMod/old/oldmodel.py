import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, GCNConv
from torch.nn import Linear
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, num_features):
        super(GNN, self).__init__()
        # Max vs mean
        self.conv1 = GCNConv(num_features, 128, aggr='mean')
        self.bnorm1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, 256, aggr='mean')
        self.bnorm2 = nn.BatchNorm1d(256)
        self.conv3 = GCNConv(256, 512, aggr='mean')
        self.bnorm3 = nn.BatchNorm1d(512)
        # Increase dropout rate with size of network
        self.dropout1 = nn.Dropout(p = 0.20)
        self.dropout2 = nn.Dropout(p = 0.25)
        # self.fcn = Linear(256*2, 1)
        self.edge_predictor = torch.nn.Sequential(
            torch.nn.Linear(512 * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x, edge_index):
        x = F.leaky_relu(self.bnorm1(self.conv1(x, edge_index)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bnorm2(self.conv2(x, edge_index)))
        x = self.dropout2(x)
        x = F.leaky_relu(self.bnorm3(self.conv3(x, edge_index)))
        pred = torch.sigmoid(self.edge_predictor(torch.cat([x[edge_index[0]],
                                                            x[edge_index[1]]],
                                                            dim=1)))
        return x, pred
