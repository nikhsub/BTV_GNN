import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import knn_graph
import faiss

class GNNModel(torch.nn.Module):
    def __init__(self, indim, outdim, heads=5):
        super(GNNModel, self).__init__()

        #self.cnn = nn.Conv1d(in_channels=indim, out_channels=outdim//8, kernel_size=3, padding=1)

        self.nn1 = nn.Sequential(
                            nn.Linear(indim, 2*indim),
                            nn.Softplus(),
                            nn.Linear(2*indim, outdim//4),
                            nn.Softplus(),
                            nn.Linear(outdim//4, outdim//8)
                         )
#        self.fc = nn.Linear(indim, outdim//8)

        self.gcn1 = GCNConv(outdim//8, outdim//4)
        self.gat1 = GATConv(outdim//4, outdim//4, heads=heads, concat=False)

        self.gcn2 = GCNConv(outdim//4, outdim//2)
        self.gat2 = GATConv(outdim//2, outdim, heads=heads, concat=False)

        catout = outdim//4+outdim

        self.bn1 = nn.BatchNorm1d(outdim//4)
        self.bn2 = nn.BatchNorm1d(outdim)
        self.drop1 = nn.Dropout(p=0.2)
        self.drop2 = nn.Dropout(p=0.25)

        # FC layer to predict edge probabilities
        self.edge_pred = nn.Sequential(
                            nn.Linear(2*catout+1, 4*catout),
                            nn.LeakyReLU(),
                            nn.Dropout(0.25),
                            nn.Linear(4*catout, 1),
                            #nn.ReLU(),
                            #nn.Dropout(0.2),
                            #nn.Linear(2*catout, 1),
                            nn.Sigmoid()
                         )
    
    def forward(self, data, device, training, thres=0.5):

        num_seeds = data.seeds.size(0)
        
        x = self.nn1(data.x)
        x = F.leaky_relu(x)

        edge_index = self.knn_update(x, data.seeds, num_seeds, device, False, training)

        x1 = self.gcn1(x, edge_index)
        #x1 = F.leaky_relu(x1)
        x1 = self.gat1(x1, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1)
        x1 = self.drop1(x1)

        x2 = self.gcn2(x1, edge_index)
        x2 = self.gat2(x2, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2)
        x2 = self.drop1(x2)

        edge_index = self.knn_update(x1, data.seeds, num_seeds, device, False, training )

        xf = torch.cat((x1, x2), dim=1)

        edge_start = xf[edge_index[0]]
        edge_end   = xf[edge_index[1]]

        cosine_sim = F.cosine_similarity(edge_start, edge_end, dim=1).unsqueeze(1)  # Shape: [num_edges, 1]

        edge_feats = torch.cat([edge_start, edge_end, cosine_sim], dim=1)

        #edge_weights = torch.norm(edge_start - edge_end, dim=1)
        #edge_feats = torch.cat([edge_feats, edge_weights.unsqueeze(1)], dim=1)

        edge_probs = self.edge_pred(edge_feats)

        if training:
            edge_labels = self.compute_edge_labels(data.y, edge_index, device)
        else:
            edge_labels = []

        if(training): return edge_probs, edge_labels, edge_index
        else: return edge_probs, edge_index

    def knn_update(self, x, seeds, k, device, init, training):
        if(init): knn_features = x[:, :4].to(device)
        else:  knn_features = x[:].to(device)
        
        if training:
            # During training, the first half of the dataset are seeds
            num_seeds = x.size(0) // 2  # First half of the data considered seeds
            seed_tracks = torch.arange(num_seeds, device=device)
        else:
            seed_tracks = torch.tensor(seeds, device=device)
    
        # Perform KNN on the entire set of tracks
        edge_index = knn_graph(knn_features, k=k, batch=None, loop=False, cosine=True, flow="source_to_target").to(device)
    
        # Filter edge_index to retain only edges where the source node is a seed track
        if not training and not init:
            mask_source = torch.isin(edge_index[0], seed_tracks).to(device)  # Keep edges starting from seed tracks
            mask_target = torch.isin(edge_index[1], seed_tracks).to(device)
            mask = mask_source | mask_target
            edge_index = edge_index[:, mask].to(device)  # Apply mask
    
        return edge_index


    def compute_edge_labels(self, labels, edge_index, device):
        # Precompute label sums and differences for all edges at once
        label_start = labels[edge_index[0]]  # Shape: [num_edges]
        label_end = labels[edge_index[1]]    # Shape: [num_edges]

        # Initialize edge labels based on conditions
        edge_labels = torch.zeros(edge_index.size(1), dtype=torch.float, device=device)  # Shape: [num_edges]

        edge_labels[(label_start == 1) & (label_end == 1)] = 0.99  # Signal-to-signal connection
        edge_labels[(label_start == 1) & (label_end == 0)] = 0.01   # Signal-to-background connection
        edge_labels[(label_start == 0) & (label_end == 1)] = 0.01   # Background-to-signal connection
        edge_labels[(label_start == 0) & (label_end == 0)] = 0.001  # Background-to-background connection

        return edge_labels


