import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

# ----------- Custom Edge-aware Conv Layer -------------
class EdgeMLPConv(MessagePassing):
    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

class SafeLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        count = x.size(-1)
        mean = torch.sum(x, dim=-1, keepdim=True) / count
        var = torch.sum((x - mean) ** 2, dim=-1, keepdim=True) / count
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias

class ONNXScatter(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, src, index, dim_size):
        output = torch.zeros(dim_size, *src.shape[1:], device=src.device)
        return output.index_add(0, index, src)
        


class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim):
        super(GNNModel, self).__init__()

        self.bn0 = nn.LayerNorm(indim, eps=1e-6)

        self.onnx_scatter = ONNXScatter()

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

        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, 1),
            nn.Sigmoid()
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

        batch_size, num_nodes, _ = x_in.shape
        xf_all, probs_all = [], []

        edge_index = edge_index.long()

        x = x_in[0]                          # [num_nodes, indim]
        e_idx = edge_index[0]               # [2, num_edges]
        e_attr = edge_attr[0]               # [num_edges, edge_dim]


        #manual_dummy = torch.tensor(
        #    [-2.3952117, 0.54164386, 0.09683848, 0.85531396, 0.7040078,
        #     0.21099302, 1.6392194, 0.24887794, -1.8308424, -1.1111757,
        #     -1.0824525, -1.8598198],
        #    dtype=x.dtype,
        #    device=x.device
        #)
        invalid_mask = (x[:, 0] == -999.0)
        x[invalid_mask] = self.dummy_token
        #x[invalid_mask] = manual_dummy

        x = self.bn0(x)

        e_attr_enc = self.edge_encoder(e_attr)
        edge_weights = self.edge_classifier(e_attr).view(-1, 1)
        e_attr_enc = e_attr_enc * edge_weights

        x1 = self.edge_mlpconv1(x, e_idx, e_attr_enc)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        x2 = self.edge_mlpconv2(x1, e_idx, e_attr_enc)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.drop2(x2)

        gate = torch.sigmoid(self.gate_layer(self.proj_skip(x)))
        #self.last_gate = gate.detach()
        xf = gate * self.proj_skip(x) + torch.sub(1, gate) * x2

        ones = torch.ones(e_idx.size(1), device=x.device)
        deg = self.onnx_scatter(ones, e_idx[1], num_nodes).unsqueeze(1).clamp(min=1)
        edge_feats_sum = self.onnx_scatter(e_attr_enc, e_idx[1], num_nodes)
        edge_feats_mean = edge_feats_sum / deg

        xf_combined = torch.cat([xf, edge_feats_mean], dim=1)
        node_probs = self.node_pred(xf_combined)

        return xf_combined.unsqueeze(0), node_probs.unsqueeze(0)



