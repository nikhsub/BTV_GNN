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
                            #nn.ReLU(),
                            nn.Linear(2*indim, outdim//4),
                            #nn.ReLU(),
                            nn.Linear(outdim//4, outdim//8)
                         )
        #self.fc = nn.Linear(indim, outdim//8)

        self.gcn1 = GCNConv(outdim//8, outdim//4)
        self.gat1 = GATConv(outdim//4, outdim//4, heads=heads, concat=False)

        self.gcn2 = GCNConv(outdim//4, outdim//2)
        self.gat2 = GATConv(outdim//2, outdim//2, heads=heads, concat=False)

        self.gcn3 = GCNConv(outdim//2, outdim)
        self.gat3 = GATConv(outdim, outdim, heads=heads, concat=False)

        self.gcn4 = GCNConv(outdim, outdim)
        self.gat4 = GATConv(outdim, outdim, heads=heads*2, concat=False)

        catout = outdim//4+outdim//2+outdim+outdim

        self.bn1 = nn.BatchNorm1d(outdim//4)
        self.bn2 = nn.BatchNorm1d(outdim//2)
        self.bn3 = nn.BatchNorm1d(outdim)

        # FC layer to predict edge probabilities
        self.edge_pred = nn.Sequential(
                            nn.Linear(2*catout+1, 4*catout),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(4*catout, 2*catout),
                            nn.ReLU(),
                            nn.Dropout(0.3),
                            nn.Linear(2*catout, 1),
                            nn.Sigmoid()
                         )
    
    def forward(self, data, device):


        edge_index = self.knn_update(data.x, data.seeds, 20, device, True)

        x = self.nn1(data.x)
        x = F.relu(x)

        x1 = self.gcn1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.gat1(x1, edge_index)
        x1 = F.relu(x1)
#        x1 = self.bn1(x1)

        #edge_index = self.knn_update(x1, data.seeds, 20, device)

        x2 = self.gcn2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = self.gat2(x2, edge_index)
        x2 = F.relu(x2)
 #       x2 = self.bn2(x2)

        edge_index = self.knn_update(x2, data.seeds, 10, device, False)

        x3 = self.gcn3(x2, edge_index)
        x3 = F.relu(x3)
        x3 = self.gat3(x3, edge_index)
        x3 = F.relu(x3)
  #      x3 = self.bn3(x3)

        #edge_index = self.knn_update(x3, data.seeds, 10, device)

        x4 = self.gcn4(x3, edge_index)
        x4 = F.relu(x4)
        x4 = self.gat4(x4, edge_index)
        x4 = F.relu(x4)

        xf = torch.cat((x1, x2, x3, x4), dim=1)

        edge_start = xf[edge_index[0]]
        edge_end   = xf[edge_index[1]]
        edge_weights = torch.norm(edge_start - edge_end, dim=1)
        edge_feats = torch.cat([edge_start, edge_end], dim=1)
        edge_feats = torch.cat([edge_feats, edge_weights.unsqueeze(1)], dim=1)

        edge_probs = self.edge_pred(edge_feats)

        edge_labels = self.compute_edge_labels(data.y, edge_index, device)

        return edge_probs, edge_labels

    def knn_update(self, x, seeds, k, device, init):
        if(init): knn_features = x[:, :4].to(device)  # Use first four features and move to device
        else:  knn_features = x[:].to(device)
    
        seed_tracks = seeds.nonzero(as_tuple=True)[0].to(device)  # Indices of seed tracks
    
        # Perform KNN on the entire set of tracks
        edge_index = knn_graph(knn_features, k=k, batch=None, loop=False, flow="source_to_target").to(device)
    
        # Filter edge_index to retain only edges where the source node is a seed track
        mask_source = torch.isin(edge_index[0], seed_tracks).to(device)  # Keep edges starting from seed tracks
        mask_target = torch.isin(edge_index[1], seed_tracks).to(device)
        mask = mask_source | mask_target
        edge_index = edge_index[:, mask].to(device)  # Apply mask
    
        return edge_index


    def old_compute_edge_labels(self, labels, edge_index, device):

        edge_labels = []  # To store the edge labels

        for i, j in zip(edge_index[0], edge_index[1]):
            # Compute edge label based on label comparison between track i and track j
            label_start = labels[i].sum().item()
            label_end = labels[j].sum().item()
            label_diff = torch.norm(labels[i] - labels[j], p=2).item()

            # Label conditions as per your earlier structure
            if label_start + label_end == 2 and label_diff == 0:
                edge_label = 0.99  # Correct signal-to-signal connection
            elif label_start + label_end == 2 and label_diff != 0:
                edge_label = 0.5  # Incorrect signal-to-signal connection
            elif label_start + label_end == 1:
                edge_label = 0.1  # Signal-to-background connection
            else:
                edge_label = 0.01  # Background-to-background connection

            # Add the computed edge label to the list
            edge_labels.append(edge_label)


        # Convert the list of edge labels to a tensor and return it
        edge_labels = torch.tensor(edge_labels, dtype=torch.float).to(device)

        return edge_labels

    def compute_edge_labels(self, labels, edge_index, device):
        # Precompute label sums and differences for all edges at once
        label_start = labels[edge_index[0]].sum(dim=1)  # Shape: [num_edges]
        label_end = labels[edge_index[1]].sum(dim=1)    # Shape: [num_edges]
        label_diff = torch.norm(labels[edge_index[0]] - labels[edge_index[1]], p=2, dim=1)  # Shape: [num_edges]

        # Initialize edge labels based on conditions
        edge_labels = torch.zeros(edge_index.size(1), dtype=torch.float, device=device)  # Shape: [num_edges]

        # Apply conditions
        edge_labels[(label_start + label_end == 2) & (label_diff == 0)] = 0.99  # Correct signal-to-signal connection
        edge_labels[(label_start + label_end == 2) & (label_diff != 0)] = 0.5   # Incorrect signal-to-signal connection
        edge_labels[(label_start + label_end == 1)] = 0.1                       # Signal-to-background connection
        edge_labels[(label_start + label_end == 0)] = 0.01                      # Background-to-background connection

        return edge_labels


