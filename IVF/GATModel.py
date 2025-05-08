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
        nn.LayerNorm(edge_dim),
        nn.Linear(edge_dim, outdim//2),
        nn.ReLU(),  # more stable than LeakyReLU here
        nn.Linear(outdim//2, outdim)
        ) 

        self.edge_classifier = nn.Sequential(
        nn.Linear(edge_dim, edge_dim // 2),
        nn.ReLU(),
        nn.Linear(edge_dim // 2, 1),
        nn.Sigmoid()
        )

        self.gat1 = GATv2Conv(indim, 2*indim//heads, edge_dim=outdim, heads=heads, concat=True, dropout=0.2)
        self.gat2 = GATv2Conv((2*indim//heads)*heads, outdim//heads, edge_dim=outdim, heads=heads, concat=True, dropout=0.25, negative_slope=0.2)
        self.gat3 = GATv2Conv((outdim//heads)*heads, (outdim*2)//heads, edge_dim=outdim, heads=heads, concat=True, dropout=0.3, negative_slope=0.2)

        self.bn1 = nn.LayerNorm((2*indim//heads)*heads)
        self.bn2 = nn.LayerNorm((outdim//heads)*heads)
        self.bn3 = nn.LayerNorm((2*outdim//heads)*heads)
        self.drop1 = nn.Dropout(p=0.3)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=0.1)

        self.skip_weight = nn.Parameter(torch.tensor(0.01), requires_grad=False)
        self.alpha = nn.Parameter(torch.tensor(0.3), requires_grad=True)  # Learnable mix parameter
        
        self.final_proj = nn.Linear((2*indim//heads)*heads, (outdim//heads)*heads)

        catout = (outdim//heads)*heads + outdim

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

        edge_weights = self.edge_classifier(edge_attr).view(-1, 1)
        edge_attr_enc = edge_attr_enc * edge_weights

        x1, attn1 = self.gat1(x, edge_index, edge_attr=edge_attr_enc, return_attention_weights=True)
        if torch.isnan(x1).any():
            print("NaNs after gat1")
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        x_proj = self.proj_skip(x)
        skip1 = self.skip_weight * x_proj + x1

        x2, attn2 = self.gat2(skip1, edge_index, edge_attr=edge_attr_enc, return_attention_weights=True)
        if torch.isnan(x2).any():
            print("NaNs after gat2")
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.drop2(x2)

        #x3, attn3 = self.gat3(x2, edge_index, edge_attr=edge_attr_enc, return_attention_weights=True)
        #if torch.isnan(x3).any():
        #    print("NaNs after gat3")
        #x3 = self.bn3(x3)
        #x3 = F.leaky_relu(x3)
        #x3 = self.drop3(x3)
        
        final_x1 = self.final_proj(x1)
        if torch.isnan(final_x1).any():
            print("NaNs in final_proj")

        xf = torch.sigmoid(self.alpha) * final_x1 + (1 - torch.sigmoid(self.alpha)) * x2
        if torch.isnan(xf).any():
            print("NaNs in final output xf")

        num_nodes = x.size(0)
        edge_feats_sum = torch.zeros((num_nodes, edge_attr_enc.size(1)), device=x.device)
        edge_feats_sum = edge_feats_sum.index_add(0, edge_index[1], edge_attr_enc)
        
        deg = torch.bincount(edge_index[1], minlength=num_nodes).clamp(min=1).unsqueeze(1).float()
        edge_feats_mean = edge_feats_sum / deg
        
        xf = torch.cat([xf, edge_feats_mean], dim=1)

        node_probs = self.node_pred(xf)
        
        attn3 = None

        return xf, node_probs, attn1, attn2, attn3

