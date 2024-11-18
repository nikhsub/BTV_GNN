import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph
import faiss

class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, heads=5):
        super(GNNModel, self).__init__()

        self.nn1 = nn.Sequential(
                            nn.Linear(indim, 2*indim),
                            nn.Softplus(),
                            nn.Linear(2*indim, outdim//4),
                            nn.Softplus(),
                            nn.Linear(outdim//4, outdim//8)
                         )

        self.bn0 = nn.BatchNorm1d(outdim//8)

        self.gcn1 = GCNConv(outdim//8, outdim//4)
        self.gat1 = GATConv(outdim//4, outdim//4, heads=heads, concat=False)

        self.gcn2 = GCNConv(3*outdim//8, outdim//2)
        self.gat2 = GATConv(outdim//2, 7*outdim//8, heads=heads, concat=False)


        self.bn1 = nn.BatchNorm1d(outdim//4)
        self.bn2 = nn.BatchNorm1d(7*outdim//8)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.25)

        catout = outdim+outdim//4

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

    def forward(self, data, edge_index, device):


        x = self.nn1(data.x)
        x = self.bn0(x)
        x = F.leaky_relu(x)

        x1 = self.gcn1(x, edge_index)
        x1 = self.gat1(x1, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        skip1 = torch.cat([x, x1], dim=1)

        x2 = self.gcn2(skip1, edge_index)
        x2 = self.gat2(x2, edge_index)
        x3 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.drop1(x2)

        xf = torch.cat([x, x1, x2], dim=1)

        edge_probs = self.node_pred(xf)

        return xf, edge_probs
