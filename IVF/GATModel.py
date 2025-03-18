import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim, heads=5, dropout=0.25):
        super(GNNModel, self).__init__()

        self.bn0 = nn.BatchNorm1d(indim)

        self.proj_skip = nn.Linear(indim, indim*2)

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, outdim),
            nn.LeakyReLU(),
            nn.Linear(outdim, outdim // 2),
            nn.LeakyReLU()
        )

        self.gat1 = GATv2Conv(indim, 2*indim, edge_dim=outdim // 2, heads=heads, concat=False)
        self.gat2 = GATv2Conv(2*indim, outdim, edge_dim=outdim // 2, heads=heads, concat=False, negative_slope=0.2)
        self.gat3 = GATv2Conv(outdim, outdim*2, edge_dim=outdim // 2, heads=heads, concat=False, negative_slope=0.2)

        self.bn1 = nn.BatchNorm1d(2*indim)
        self.bn2 = nn.BatchNorm1d(outdim)
        self.bn3 = nn.BatchNorm1d(outdim*2)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)

        self.skip_weight = nn.Parameter(torch.tensor(0.2), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad=True)  # Learnable mix parameter
        
        self.final_proj = nn.Linear(indim*2, outdim*2)

        catout = outdim*2

        self.node_pred = nn.Sequential(
            nn.Linear(catout, catout//2),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(catout//2, catout//4),
            nn.Softplus(),
            nn.Dropout(0.2),
            nn.Linear(catout//4, 1)
        )

    def forward(self, x_in, edge_index, edge_attr):
        x = self.bn0(x_in)

        edge_attr_enc = self.edge_encoder(edge_attr)

        x1, attn1 = self.gat1(x, edge_index, edge_attr=edge_attr_enc, return_attention_weights=True)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        x_proj = self.proj_skip(x)
        skip1 = self.skip_weight * x_proj + x1

        x2, attn2 = self.gat2(skip1, edge_index, edge_attr=edge_attr_enc, return_attention_weights=True)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.drop2(x2)

        x3, attn3 = self.gat3(x2, edge_index, edge_attr=edge_attr_enc, return_attention_weights=True)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3)
        x3 = self.drop3(x3)
        
        final_x1 = self.final_proj(x1)

        xf = torch.sigmoid(self.alpha) * final_x1 + (1 - torch.sigmoid(self.alpha)) * x3

        node_probs = self.node_pred(xf)

        return xf, node_probs, attn1, attn2, attn3

