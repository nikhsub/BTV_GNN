import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# ----------- Custom Edge-aware Conv Layer -------------
class EdgeMLPConv(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim, heads=4, dropout=0.25):
        super(GNNModel, self).__init__()

        self.bn0 = nn.BatchNorm1d(indim)

        self.proj_skip = nn.Linear(indim, outdim)

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

        self.edge_mlpconv1 = EdgeMLPConv(indim, outdim, outdim//4)
        self.edge_mlpconv2 = EdgeMLPConv(outdim//4, outdim, outdim//2)
        self.edge_mlpconv3 = EdgeMLPConv(outdim//2, outdim, outdim)

        self.bn1 = nn.LayerNorm(outdim//4)
        self.drop1 = nn.Dropout(p=0.3)
        
        self.bn2 = nn.LayerNorm(outdim//2)
        self.drop2 = nn.Dropout(p=0.2)

        self.bn3 = nn.LayerNorm(outdim)
        self.drop3 = nn.Dropout(p=0.1)

        self.skip_weight = nn.Parameter(torch.tensor(0.01), requires_grad=False)

        catout = outdim + outdim

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

        x1 = self.edge_mlpconv1(x, edge_index, edge_attr_enc)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)
    
        x2 = self.edge_mlpconv2(x1, edge_index, edge_attr_enc)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.drop2(x2)

        x3 = self.edge_mlpconv3(x2, edge_index, edge_attr_enc)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = self.drop3(x3)

        x_proj = self.proj_skip(x)
        xf = self.skip_weight * x_proj + x3

        num_nodes = x.size(0)
        edge_feats_sum = torch.zeros((num_nodes, edge_attr_enc.size(1)), device=x.device)
        edge_feats_sum = edge_feats_sum.index_add(0, edge_index[1], edge_attr_enc)
        
        deg = torch.bincount(edge_index[1], minlength=num_nodes).clamp(min=1).unsqueeze(1).float()
        edge_feats_mean = edge_feats_sum / deg
        
        xf = torch.cat([xf, edge_feats_mean], dim=1)

        node_probs = self.node_pred(xf)
        
        attn3 = None
        attn2 = None
        attn1 = None

        return xf, node_probs, attn1, attn2, attn3

