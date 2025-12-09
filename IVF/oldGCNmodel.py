import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

# ----------- Custom Edge-aware Conv Layer -------------
class EdgeMLPConv(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr=None)
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )

        self.edge_score = nn.Linear(out_channels, 1)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # inside message()
        msg = self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))
        w = torch.tanh(self.edge_score(msg)).squeeze(-1)   # smaller range than sigmoid
        self._alpha = w
        return msg * (1 + w).unsqueeze(-1) * 0.5           # range [0,1]


    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Softmax normalization per receiver node
        alpha = softmax(self._alpha, index)            # [E]
        return scatter_add(inputs * alpha.unsqueeze(-1),
                           index, dim=0, dim_size=dim_size)


class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim):
        super(GNNModel, self).__init__()

        self.bn0 = nn.LayerNorm(indim, eps=1e-6)

        self.proj_skip = nn.Linear(indim, outdim)

        self.dummy_token = nn.Parameter(torch.randn(indim)) 

        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_dim, eps=1e-6),
            nn.Linear(edge_dim, outdim),
            nn.ReLU(),
            nn.Linear(outdim, outdim * 2),
            nn.ReLU(),
            nn.Linear(outdim * 2, outdim)
        )

        self.edge_att = nn.Sequential(
            nn.Linear(outdim, outdim // 2), nn.ReLU(),
            nn.Linear(outdim // 2, 1)
        ) 

        self.edge_mlpconv1 = EdgeMLPConv(indim, outdim, outdim//2)
        self.edge_mlpconv2 = EdgeMLPConv(outdim//2, outdim, outdim)

        self.bn1 = nn.LayerNorm(outdim//2, eps=1e-6)
        self.drop1 = nn.Dropout(p=0.3)

        self.bn2 = nn.LayerNorm(outdim, eps=1e-6)
        self.drop2 = nn.Dropout(p=0.2)

        self.gate_layer = nn.Linear(outdim, outdim)

        catout = outdim + outdim

        self.node_pred = nn.Sequential(
            nn.Linear(catout, catout//2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(catout//2, catout//4),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(catout//4, 1)
        )

    def forward(self, x_in, edge_index, edge_attr):

        _, num_nodes, _ = x_in.shape

        edge_index = edge_index.long()


        x = x_in.squeeze(0)                          # [num_nodes, indim]
        e_idx = edge_index.squeeze(0)               # [2, num_edges
        e_attr = edge_attr.squeeze(0)               # [num_edges, edge_dim]


        invalid_mask = (x[:, 0] == -999.0)
        dummy = self.dummy_token.expand_as(x)
        x = torch.where(invalid_mask.unsqueeze(1), dummy, x)

        x = self.bn0(x)

        e_attr_enc = self.edge_encoder(e_attr)
        e_scores = self.edge_att(e_attr_enc).squeeze(-1)             # [E]
        # normalize per receiver node (dst) => soft attention
        alpha = softmax(e_scores, e_idx[1])                             # [E]
        # gate the messages; keep residual so info can't vanish
        e_attr_msg = e_attr_enc * alpha.unsqueeze(1) 

        x1 = self.edge_mlpconv1(x, e_idx, e_attr_msg)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        x2 = self.edge_mlpconv2(x1, e_idx, e_attr_msg)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.drop2(x2)

        gate = torch.sigmoid(self.gate_layer(self.proj_skip(x)))
        xf = gate * self.proj_skip(x) + (1-gate) * x2

        ones = torch.ones(e_idx.size(1), device=x.device)
        deg = scatter_add(ones, e_idx[1], dim=0, dim_size=num_nodes).unsqueeze(1).clamp(min=1)
        edge_feats_sum = scatter_add(e_attr_enc, e_idx[1], dim=0, dim_size=num_nodes)
         
        edge_feats_mean = edge_feats_sum / deg

        xf_combined = torch.cat([xf, edge_feats_mean], dim=1)
        node_probs = self.node_pred(xf_combined)

        return xf_combined, node_probs
