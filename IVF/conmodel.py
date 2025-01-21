import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, heads=5, dropout=0.25):
        super(GNNModel, self).__init__()

        self.nn1 = nn.Sequential(
                            nn.Linear(indim, 2*indim),
                            nn.Softplus(),
                            nn.Linear(2*indim, outdim),
                            nn.Softplus(),
                            nn.Linear(outdim, outdim*2)
                         )

        self.bn0 = nn.BatchNorm1d(outdim*2)

        self.gcn1 = GCNConv(outdim*2, outdim*2)
        self.gat1 = GATConv(outdim*2, outdim*2, heads=heads, concat=False)

        self.gcn2 = GCNConv(outdim*2, outdim)
        self.gat2 = GATConv(outdim, outdim, heads=heads, concat=False)


        self.bn1 = nn.BatchNorm1d(outdim*2)
        self.bn2 = nn.BatchNorm1d(outdim)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

        catout = outdim + outdim*2

        self.node_pred = nn.Sequential(
            nn.Linear(catout, catout//2),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(catout//2, catout//4),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(catout//4, catout//8),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(catout//8, 1),
            nn.Sigmoid()
        )

    def forward(self, x_in, edge_index):


        x = self.nn1(x_in)
        x = self.bn0(x)
        x = F.leaky_relu(x)

        x1 = self.gcn1(x, edge_index)
        x1 = self.gat1(x1, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        skip1 = x + x1

        x2 = self.gcn2(skip1, edge_index)
        x2 = self.gat2(x2, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.drop1(x2)

        xf = torch.cat([skip1, x2], dim=1)

        node_probs = self.node_pred(xf)

        return xf, node_probs
