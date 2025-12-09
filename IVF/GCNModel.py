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
        # Message built from receiver, sender, and edge
        msg = self.mlp(torch.cat([x_i, x_j, edge_attr], dim=1))
        w = torch.tanh(self.edge_score(msg)).squeeze(-1)   # scalar per edge
        self._alpha = w
        return msg * (1 + w).unsqueeze(-1) * 0.5           # rescale to [0,1]

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Softmax normalization per receiver node
        alpha = softmax(self._alpha, index)            # [E]
        return scatter_add(inputs * alpha.unsqueeze(-1),
                           index, dim=0, dim_size=dim_size)


class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, edge_dim, num_classes=7):
        super(GNNModel, self).__init__()

        self.indim = indim
        self.outdim = outdim
        self.num_classes = num_classes

        self.bn0 = nn.LayerNorm(indim, eps=1e-6)

        self.proj_skip = nn.Linear(indim, outdim)

        # Learnable dummy token for invalid tracks
        self.dummy_token = nn.Parameter(torch.randn(indim))

        # ----- Edge encoder -----
        self.edge_encoder = nn.Sequential(
            nn.LayerNorm(edge_dim, eps=1e-6),
            nn.Linear(edge_dim, outdim),
            nn.ReLU(),
            nn.Linear(outdim, outdim * 2),
            nn.ReLU(),
            nn.Linear(outdim * 2, outdim)
        )

        # Edge-level attention scores (scalar per edge)
        self.edge_att = nn.Sequential(
            nn.Linear(outdim, outdim // 2),
            nn.ReLU(),
            nn.Linear(outdim // 2, 1)
        )

        # ----- Node message passing using edge-aware conv -----
        self.edge_mlpconv1 = EdgeMLPConv(indim, outdim, outdim // 2)
        self.edge_mlpconv2 = EdgeMLPConv(outdim // 2, outdim, outdim)

        self.bn1 = nn.LayerNorm(outdim // 2, eps=1e-6)
        self.drop1 = nn.Dropout(p=0.3)

        self.bn2 = nn.LayerNorm(outdim, eps=1e-6)
        self.drop2 = nn.Dropout(p=0.2)

        # Gating between skip and message-passed features
        self.gate_layer = nn.Linear(outdim, outdim)

        # Final concatenated node feature size
        catout = outdim + outdim  # [xf, edge_feats_mean]

        # ----- Node classifier (multiclass) -----
        self.node_pred = nn.Sequential(
            nn.Linear(catout, catout // 2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(catout // 2, catout // 4),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(catout // 4, num_classes)   # logits for 7 classes
        )

        # ----- Edge classifier -----
        # We'll use xf[src], xf[dst], and e_attr_enc (per-edge) as input
        edge_in_dim = outdim * 2 + outdim  # xf[src], xf[dst], e_attr_enc

        self.edge_pred = nn.Sequential(
            nn.Linear(edge_in_dim, outdim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(outdim, outdim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(outdim // 2, 1)      # binary edge logit
        )

    def forward(self, x_in, edge_index, edge_attr):
        """
        x_in:       [1, N, indim]
        edge_index: [1, 2, E]
        edge_attr:  [1, E, edge_dim]

        Returns:
            node_logits: [N, num_classes]
            edge_logits: [E, 1]
            node_probs:  [N, num_classes]  (softmax)
            edge_probs:  [E, 1]            (sigmoid)
        """

        # Unpack batch dim (you were using unsqueeze(0) in the caller)
        num_nodes, _ = x_in.shape

        edge_index = edge_index.long()

        x = x_in.squeeze(0)          # [N, indim]
        e_idx = edge_index.squeeze(0)  # [2, E]
        e_attr = edge_attr.squeeze(0)  # [E, edge_dim]

        # ----- Handle invalid tracks with dummy token -----
        invalid_mask = (x[:, 0] == -999.0)
        dummy = self.dummy_token.expand_as(x)
        x = torch.where(invalid_mask.unsqueeze(1), dummy, x)

        x = self.bn0(x)

        # ----- Encode edges + attention weighting -----
        e_attr_enc = self.edge_encoder(e_attr)           # [E, outdim]
        e_scores = self.edge_att(e_attr_enc).squeeze(-1) # [E]
        alpha = softmax(e_scores, e_idx[1])              # normalize per dst node
        e_attr_msg = e_attr_enc * alpha.unsqueeze(1)     # [E, outdim]

        # ----- Node message passing -----
        x1 = self.edge_mlpconv1(x, e_idx, e_attr_msg)    # [N, outdim//2]
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        x2 = self.edge_mlpconv2(x1, e_idx, e_attr_msg)   # [N, outdim]
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.drop2(x2)

        # Gated skip connection from original (encoded) x
        skip = self.proj_skip(x)                         # [N, outdim]
        gate = torch.sigmoid(self.gate_layer(skip))      # [N, outdim]
        xf = gate * skip + (1 - gate) * x2               # [N, outdim]

        # ----- Aggregate edge features per node -----
        ones = torch.ones(e_idx.size(1), device=x.device)
        deg = scatter_add(ones, e_idx[1], dim=0, dim_size=num_nodes).unsqueeze(1).clamp(min=1)
        edge_feats_sum = scatter_add(e_attr_enc, e_idx[1], dim=0, dim_size=num_nodes)
        edge_feats_mean = edge_feats_sum / deg           # [N, outdim]

        # Final node features (node + aggregated edge)
        xf_combined = torch.cat([xf, edge_feats_mean], dim=1)  # [N, 2*outdim]

        # ----- Node classification -----
        node_logits = self.node_pred(xf_combined)        # [N, num_classes]
        node_probs = torch.softmax(node_logits, dim=-1)  # [N, num_classes]

        # ----- Edge classification -----
        src, dst = e_idx
        # use node features after gating (xf), not the concatenated ones
        edge_feat_input = torch.cat(
            [xf[src], xf[dst], e_attr_enc], dim=1
        )                                                # [E, 3*outdim]

        edge_logits = self.edge_pred(edge_feat_input)    # [E, 1]
        edge_probs = torch.sigmoid(edge_logits)          # [E, 1]

        return node_logits, edge_logits, node_probs, edge_probs

