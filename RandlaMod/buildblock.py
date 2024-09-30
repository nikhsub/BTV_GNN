import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from torch_cluster import knn_graph
#from sklearn.neighbors import kneighbors_graph
import numpy as np

class Locse(nn.Module):
    def __init__(self, in_features, out_features, k):
        super(Locse, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
                nn.Linear(7, out_features//2),  # Input dimension is 7
                nn.ReLU(),
                nn.Linear(out_features//2, out_features)
                )

    def forward(self, x, pos, edge_index):
        # x: node features, pos: positions

        device = pos.device

        #edge_index = knn_graph(pos, self.k)

        row, col = edge_index
        relative_pos = pos[col] - pos[row]
        distance = torch.norm(relative_pos, dim=1, keepdim=True)

        center_pos = pos[row]
        neighbor_pos = pos[col]
        concatenated_features = torch.cat([center_pos, neighbor_pos, relative_pos, distance], dim=1)

        rk = self.mlp(concatenated_features)

        #fkhat = torch.zeros((self.k*x.size(0), x.size(1) + rk.size(1)), device=device)

        #for i in range(len(row)):
        #    center_idx = row[i]
        #    neighbor_idx = col[i]
        #    augfeat = torch.cat([x[neighbor_idx], rk[i]], dim=0)
        #    fkhat[i] = augfeat
        #fkhat = fkhat.view(x.size(0), self.k, -1)

        augfeat = torch.cat([x[col], rk], dim=1)
        fkhat = augfeat.view(x.size(0), self.k, -1).to(pos.device)

        return fkhat

class AttentivePooling(nn.Module):
    def __init__(self, in_features, out_features):
        super(AttentivePooling, self).__init__()
        # Shared MLP for computing attention scores
        self.attention_mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output a single score for each feature
        )
        # Output MLP to project aggregated features to desired output dimensions
        self.output_mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU()
        )

    def forward(self, x):
        # x: Input tensor with shape [num_points, k, in_features]
        
        device = x.device

        num_points, k, in_features = x.size()

        x_reshaped = x.view(-1, in_features).to(device)  # Shape: [num_points * k, in_features]
        attention_scores = self.attention_mlp(x_reshaped)  # Shape: [num_points * k, 1]
        attention_scores = attention_scores.view(num_points, k)  # Shape: [num_points, k]

        # Apply softmax to attention scores to get normalized weights
        attention_weights = F.softmax(attention_scores, dim=1)  # Shape: [num_points, k]

        # Weighted sum of neighboring features
        weighted_features = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # Shape: [num_points, in_features]

        # Project the aggregated features to output dimensions
        ftilde = self.output_mlp(weighted_features)  # Shape: [num_points, out_features]

        return ftilde
        


class Resblock (nn.Module):
    def __init__(self, in_features, out_features, k):
        super(Resblock, self).__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(in_features, out_features//4),  # Concatenated features
            nn.ReLU(),
            nn.Linear(out_features//4, out_features//2)
        )

        self.skipmlp = nn.Sequential(
            nn.Linear(in_features, out_features),  # Concatenated features
            nn.ReLU(),
            nn.Linear(out_features, out_features*2)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(out_features, out_features*4),  # Concatenated features
            nn.ReLU(),
            nn.Linear(out_features*4, out_features*2)
        )

        self.loc1 = Locse(2, out_features,k) #2 because of [eta,phi]
        self.attn1 = AttentivePooling(out_features+out_features//2, out_features//2)
        self.loc2 = Locse(2, out_features,k)
        self.attn2 = AttentivePooling(out_features+out_features//2, out_features)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, pos, edge_index):
        mlp1_out = self.mlp1(x)

        loc1_out = self.loc1(mlp1_out, pos, edge_index)
        attn1_out = self.attn1(loc1_out)

        loc2_out = self.loc2(attn1_out, pos, edge_index)
        attn2_out = self.attn2(loc2_out)

        mlp2_out = self.mlp2(attn2_out)

        skip_out = self.skipmlp(x)

        output = self.lrelu(mlp2_out + skip_out)

        return output    
