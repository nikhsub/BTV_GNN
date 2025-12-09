import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

class EdgeAttnConv(MessagePassing):
    """
    Simple single-head edge-aware attention layer.

    x:          [N, node_feat_dim]
    edge_index: [2, E]
    edge_attr:  [E, edge_dim]
    """

    def __init__(self, node_feat_dim, edge_dim, out_dim, dropout=0.1):
        super().__init__(aggr="add")  # sum messages per receiver

        self.node_feat_dim = node_feat_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim

        # Message network: depends on sender + edge
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        # Attention network: depends on receiver, sender, edge
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * node_feat_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),  # scalar score per edge
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_feat_dim]
        # edge_index: [2, E]
        # edge_attr: [E, edge_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index):
        """
        x_i: [E, node_feat_dim]  (receiver)
        x_j: [E, node_feat_dim]  (sender)
        edge_attr: [E, edge_dim]
        index: [E]          (receiver indices)
        """

        # ---- attention scores ----
        att_in = torch.cat([x_i, x_j, edge_attr], dim=-1)  # [E, 2*node_feat_dim+edge_dim]
        scores = self.attn_mlp(att_in).squeeze(-1)         # [E]
        alpha = softmax(scores, index)                     # softmax over incoming edges per node
        alpha = self.dropout(alpha)

        # ---- messages ----
        msg_in = torch.cat([x_j, edge_attr], dim=-1)       # [E, node_feat_dim+edge_dim]
        msg = self.msg_mlp(msg_in)                         # [E, out_dim]

        return msg * alpha.unsqueeze(-1)                   # [E, out_dim]


class TrackEdgeGNN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim=64,
        num_classes=7,
        dropout=0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # -------- Learnable dummy token for invalid tracks --------
        self.dummy_token = nn.Parameter(0.02 * torch.randn(node_in_dim))

        # -------- Encoders (1 hidden layer each) --------
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # -------- GNN layers (no edge updates) --------
        self.conv1 = EdgeAttnConv(
            node_feat_dim=hidden_dim,
            edge_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )
        self.conv2 = EdgeAttnConv(
            node_feat_dim=hidden_dim,
            edge_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

        # Simple residuals + norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # -------- Node head (shallow) --------
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # -------- Edge head (shallow) --------
        # src node, dst node, and encoded edge features
        edge_cls_in = hidden_dim * 2 + hidden_dim

        self.edge_head = nn.Sequential(
            nn.Linear(edge_cls_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # binary edge label
        )

    def forward(self, x_in, edge_index, edge_attr):
        """
        x_in:       [N, node_in_dim]
        edge_index: [2, E]
        edge_attr:  [E, edge_in_dim]
        """

        # 1) Replace invalid tracks with dummy
        invalid_mask = (x_in[:, 0] == -999.0)
        dummy = self.dummy_token.unsqueeze(0).expand_as(x_in)
        x_clean = torch.where(invalid_mask.unsqueeze(-1), dummy, x_in)

        # 2) Encode nodes and edges
        x = self.node_encoder(x_clean)      # [N, H]
        e = self.edge_encoder(edge_attr)    # [E, H]

        # 3) GNN layer 1
        x1 = self.conv1(x, edge_index, e)   # [N, H]
        x1 = self.ln1(x1 + x)               # residual
        x1 = self.act(x1)
        x1 = self.drop(x1)

        # 4) GNN layer 2
        x2 = self.conv2(x1, edge_index, e)  # [N, H]
        x2 = self.ln2(x2 + x1)              # residual
        x2 = self.act(x2)
        x2 = self.drop(x2)

        x_out = x2                          # [N, H]

        # 5) Node classification
        node_logits = self.node_head(x_out)          # [N, num_classes]
        node_probs  = torch.softmax(node_logits, dim=-1)

        # 6) Edge classification (use final node features + encoded edges)
        src, dst = edge_index
        edge_feat = torch.cat(
            [x_out[src], x_out[dst], e], dim=-1
        )                                             # [E, 3H]

        edge_logits = self.edge_head(edge_feat)       # [E, 1]
        edge_probs  = torch.sigmoid(edge_logits)

        return node_logits, edge_logits, node_probs, edge_probs



