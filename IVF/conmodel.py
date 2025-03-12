import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv

class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, heads=5, dropout=0.25):
        super(GNNModel, self).__init__()

        self.bn0 = nn.BatchNorm1d(indim)

        self.proj_skip = nn.Linear(indim, outdim * 2)

        self.gcn1 = GCNConv(indim, outdim)
        self.gat1 = GATConv(outdim, outdim*2, heads=heads, concat=False)

        self.gcn2 = GCNConv(outdim*2, outdim)
        self.gat2 = GATConv(outdim, outdim, heads=heads, concat=False)


        self.bn1 = nn.BatchNorm1d(outdim*2)
        self.bn2 = nn.BatchNorm1d(outdim)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

        self.skip_weight = nn.Parameter(torch.tensor(0.2))

        #catout = outdim + outdim*2
        catout = outdim*3

        self.node_pred = nn.Sequential(
            nn.Linear(catout, catout//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(catout//2, catout//4),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(catout//4, catout//8),
            nn.LeakyReLU(),
            nn.Dropout(0.25),
            nn.Linear(catout//8, 1)
            #nn.Sigmoid()
        )

    def forward(self, x_in, edge_index):


        x = self.bn0(x_in)

        x1 = self.gcn1(x, edge_index)
        x1 = self.gat1(x1, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        x_proj = self.proj_skip(x)
        skip1 = self.skip_weight * x_proj + x1

        x2 = self.gcn2(skip1, edge_index)
        x2 = self.gat2(x2, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.drop2(x2)

        xf = torch.cat([x2, x1], dim=1)

        node_probs = self.node_pred(xf)

        return xf, node_probs
