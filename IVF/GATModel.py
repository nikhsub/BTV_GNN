import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim, heads=4, dropout=0.25):
        super(GNNModel, self).__init__()

        self.bn0 = nn.BatchNorm1d(indim)

        self.proj_skip = nn.Linear(indim, (2*indim//heads)*heads)

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, outdim),
            nn.LeakyReLU(),
            nn.Linear(outdim, outdim // 2),
            nn.LeakyReLU()
        )

        self.gat1 = GATv2Conv(indim, 2*indim//heads, edge_dim=outdim // 2, heads=heads, concat=True, dropout=0.2)
        self.gat2 = GATv2Conv((2*indim//heads)*heads, outdim//heads, edge_dim=outdim // 2, heads=heads, concat=True, dropout=0.25, negative_slope=0.2)
        self.gat3 = GATv2Conv((outdim//heads)*heads, (outdim*2)//heads, edge_dim=outdim // 2, heads=heads, concat=True, dropout=0.3, negative_slope=0.2)

        self.bn1 = nn.LayerNorm((2*indim//heads)*heads)
        self.bn2 = nn.LayerNorm((outdim//heads)*heads)
        self.bn3 = nn.LayerNorm((2*outdim//heads)*heads)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=0.1)

        self.skip_weight = nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad=True)  # Learnable mix parameter
        
        self.final_proj = nn.Linear((2*indim//heads)*heads, (2*outdim//heads)*heads)

        catout = (2*outdim//heads)*heads

        self.node_pred = nn.Sequential(
            nn.Linear(catout, catout//2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(catout//2, catout//4),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(catout//4, 1)
        )

    def forward(self, x_in, edge_index, edge_attr):
        x = self.bn0(x_in)

        edge_attr_enc = self.edge_encoder(edge_attr)
        edge_attr_enc = F.normalize(edge_attr_enc, p=2, dim=-1)

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

