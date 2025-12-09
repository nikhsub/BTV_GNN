import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing

class EdgeAttnConv(MessagePassing):
    """
    ONNX-safe multi-head edge-aware attention GNN layer.

    Input:
        x:          [N, node_dim]
        edge_index: [2, E]
        edge_attr:  [E, edge_dim]

    """

    def __init__(self, node_feat_dim, edge_dim, out_dim, heads=4, dropout=0.1):
        super().__init__(aggr="add")
        self.node_feat_dim = node_feat_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.heads = heads

        # ----------- Message network: all heads at once --------------
        self.msg_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_dim, out_dim * heads),
            nn.ReLU(),
            nn.Linear(out_dim * heads, out_dim * heads),
        )

        # ----------- Attention network: all heads at once -------------
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * node_feat_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, heads),  # 1 score per head
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index):

        # ---------------- Attention scores ----------------
        att_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        scores = self.attn_mlp(att_in)                    # [E, heads]
        alpha = softmax(scores, index)                    # [E, heads]
        alpha = self.dropout(alpha).unsqueeze(-1)         # [E, heads, 1]

        # ---------------- Messages ------------------------
        msg_in = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_in)                        # [E, heads*out_dim]
        msg = msg.view(-1, self.heads, self.out_dim)      # [E, heads, out_dim]

        # ---------------- Weighted messages ---------------
        out = msg * alpha                                 # [E, heads, out_dim]

        return out.reshape(-1, self.heads * self.out_dim)  # [E, heads*outdim]

class EdgeUpdateLayer(nn.Module):
    """
    Edge update:

        e_ij' = MLP( [x_i, x_j, e_ij] ) + e_ij  (with LayerNorm)

    x:          [N, node_dim]
    edge_index: [2, E]
    edge_attr:  [E, edge_dim]
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, dropout=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim),
        )
        self.ln = nn.LayerNorm(edge_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x_i = x[dst]
        x_j = x[src]

        upd_in = torch.cat([x_i, x_j, edge_attr], dim=-1)   # [E, 2*node_dim + edge_dim]
        delta = self.mlp(upd_in)                            # [E, edge_dim]
        e_new = edge_attr + self.dropout(delta)             # residual
        e_new = self.ln(e_new)
        return e_new

class TrackEdgeGNN(nn.Module):
    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim=64,
        heads=4,
        num_classes=7,
        dropout=0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.heads = heads
        self.out_dim = hidden_dim * heads

        # ---------- Learnable dummy token for invalid tracks ----------
        self.dummy_token = nn.Parameter(0.02 * torch.randn(node_in_dim))

        # ---------- Encoders (2-layer MLPs) ----------
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

        # ---------- Node attention blocks ----------
        self.conv1 = EdgeAttnConv(
            node_feat_dim=hidden_dim,
            edge_dim=hidden_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        self.conv2 = EdgeAttnConv(
            node_feat_dim=self.out_dim,
            edge_dim=hidden_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        self.conv3 = EdgeAttnConv(
            node_feat_dim=self.out_dim,
            edge_dim=hidden_dim,
            out_dim=hidden_dim,
            heads=heads,
            dropout=dropout,
        )

        # ---------- Edge update blocks (between node blocks) ----------
        self.edge_up1 = EdgeUpdateLayer(node_dim=self.out_dim, edge_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.edge_up2 = EdgeUpdateLayer(node_dim=self.out_dim, edge_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout)

        # ---------- Skip connections ----------
        self.skip0 = nn.Linear(hidden_dim, self.out_dim)   # x0 -> x1 proj
        self.skip_long = nn.Linear(self.out_dim, self.out_dim)  # x1 -> x_out

        # ---------- Norms / activations / dropout ----------
        self.ln1 = nn.LayerNorm(self.out_dim)
        self.ln2 = nn.LayerNorm(self.out_dim)
        self.ln3 = nn.LayerNorm(self.out_dim)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # ---------- Node classifier (3 hidden layers) ----------
        self.node_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(self.out_dim // 2, self.out_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(self.out_dim // 4, self.out_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(self.out_dim // 8, num_classes),
        )

        # ---------- Edge classifier (3 hidden layers) ----------
        edge_in_dim = self.out_dim * 2 + hidden_dim  # src node, dst node, final edge features

        self.edge_head = nn.Sequential(
            nn.Linear(edge_in_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(self.out_dim, self.out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(self.out_dim // 2, self.out_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(self.out_dim // 4, 1),  # binary edge prediction
        )

    def forward(self, x_in, edge_index, edge_attr):
        """
        x_in:       [N, node_in_dim]
        edge_index: [2, E]
        edge_attr:  [E, edge_in_dim]
        """

        # ----- 1. Replace invalid tracks with dummy token -----
        invalid_mask = (x_in[:, 0] == -999.0)
        dummy = self.dummy_token.unsqueeze(0).expand_as(x_in)
        x_clean = torch.where(invalid_mask.unsqueeze(-1), dummy, x_in)

        # ----- 2. Encode nodes and edges -----
        x0 = self.node_encoder(x_clean)      # [N, H]
        e0 = self.edge_encoder(edge_attr)    # [E, H]

        # ----- 3. Block 1: node update -----
        x1 = self.conv1(x0, edge_index, e0)  # [N, H*heads]
        x1 = x1 + self.skip0(x0)             # project skip
        x1 = self.ln1(x1)
        x1 = self.act(x1)
        x1 = self.drop(x1)

        # ----- 4. Edge update 1 -----
        e1 = self.edge_up1(x1, edge_index, e0)  # [E, H]

        # ----- 5. Block 2: node update -----
        x2 = self.conv2(x1, edge_index, e1)     # [N, H*heads]
        x2 = x2 + x1                            # residual
        x2 = self.ln2(x2)
        x2 = self.act(x2)
        x2 = self.drop(x2)

        # ----- 6. Edge update 2 -----
        e2 = self.edge_up2(x2, edge_index, e1)  # [E, H]

        # ----- 7. Block 3: node update -----
        x3 = self.conv3(x2, edge_index, e2)     # [N, H*heads]
        x3 = x3 + x2                            # residual
        x3 = self.ln3(x3)
        x3 = self.act(x3)
        x3 = self.drop(x3)

        # ----- 8. Long skip from x1 -> output -----
        x_out = x3 + self.skip_long(x1)         # [N, H*heads]

        # ----- 9. Node classification -----
        node_logits = self.node_head(x_out)     # [N, num_classes]
        node_probs = torch.softmax(node_logits, dim=-1)

        # ----- 10. Edge classification (using final edges e2) -----
        src, dst = edge_index
        edge_features = torch.cat(
            [x_out[src], x_out[dst], e2], dim=-1
        )                                       # [E, 2*out_dim + H]

        edge_logits = self.edge_head(edge_features)  # [E, 1]
        edge_probs = torch.sigmoid(edge_logits)

        return node_logits, edge_logits, node_probs, edge_probs

