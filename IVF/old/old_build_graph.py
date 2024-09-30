import torch
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
import argparse
import pickle

def old_knn_data(data, k=5):
    seed_tracks = data.seeds.nonzero(as_tuple=True)[0]  # Indices of seed tracks
    knn_features = data.knn_x  # Features to use for knn
    feature_matrix = data.x  # Full feature matrix

    # Create a NearestNeighbors model to find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(knn_features)

    edge_index = []
    all_nodes = []

    # Loop over each seed track and find its k-nearest neighbors
    for seed_idx in seed_tracks:
        seed_feature = knn_features[seed_idx].unsqueeze(0)  # Get seed feature
        distances, neighbors = nbrs.kneighbors(seed_feature)  # Get k-nearest neighbors
        neighbors = neighbors.flatten()

        # Build edge indices for seed and its neighbors
        for neighbor in neighbors:
            if seed_idx.item() != neighbor.item():  # Remove self-loop
                edge_index.append([seed_idx.item(), neighbor.item()])  # Edge from seed to neighbor

        # Collect all nodes (seed + neighbors)
        all_nodes.append(seed_idx.item())
        all_nodes.extend(neighbors.tolist())

    # Remove duplicates from all_nodes
    all_nodes = list(set(all_nodes))

    # Build the edge_index tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Subset features and labels for the selected nodes
    new_x = feature_matrix[all_nodes]
    new_y = data.y[all_nodes] if data.y is not None else None

    # Create a new Data object with selected features, edge index, and labels
    new_data = Data(
        x=new_x,
        edge_index=edge_index,
        y=new_y
    )

    return new_data

def knn_data(data, k):
    seed_tracks = data.seeds.nonzero(as_tuple=True)[0]  # Indices of seed tracks
    knn_features = data.knn_x  # Features used for knn
    feature_matrix = data.x  # Full feature matrix
    labels = data.y

    # Initialize lists to store edges and new nodes
    edge_index = []
    edge_labels = []
    all_nodes = set()  # Using set to avoid duplicates

    # Create a mapping from original index to new index
    original_to_new = {}

    # Fit NearestNeighbors once for all seeds
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(knn_features)

    # Loop over each seed track to find its k-nearest neighbors
    seed_neighbors = {}  # Store neighbors for each seed to avoid recomputing

    for seed_idx in seed_tracks:
        seed_feature = knn_features[seed_idx].unsqueeze(0)  # Get seed feature

        # Find the k-nearest neighbors for the current seed
        distances, neighbors = nbrs.kneighbors(seed_feature)
        neighbors = neighbors.flatten()

        # Add seed and its neighbors to the new node set
        all_nodes.add(seed_idx.item())
        seed_neighbors[seed_idx.item()] = neighbors.tolist()  # Store the neighbors

        for neighbor in neighbors:
            all_nodes.add(int(neighbor))

    # Create a mapping from original index to new index for the reduced set
    all_nodes = list(all_nodes)
    for new_idx, original_idx in enumerate(all_nodes):
        original_to_new[original_idx] = new_idx

    # Build the edge_index using stored neighbors
    for seed_idx in seed_tracks:
        seed_new_idx = original_to_new[seed_idx.item()]
        neighbors = seed_neighbors[seed_idx.item()]

        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor in original_to_new:
                neighbor_new_idx = original_to_new[neighbor]
                if seed_new_idx != neighbor_new_idx:  # Remove self-loop
                    edge_index.append([seed_new_idx, neighbor_new_idx])

                    # Compute edge label based on your conditions
                    label_start = labels[seed_idx].sum().item()
                    label_end = labels[neighbor].sum().item()

                    label_diff = torch.norm(labels[seed_idx] - labels[neighbor], p=2).item()

                    if label_start + label_end == 2 and label_diff == 0:
                        edge_label = 1.0  # Correct signal-to-signal connection
                    elif label_start + label_end == 2 and label_diff != 0:
                        edge_label = 0.5  # Incorrect signal-to-signal connection
                    elif label_start + label_end == 1:
                        edge_label = 0.1  # Signal-to-background connection
                    else:
                        edge_label = 0.0  # Background-to-background connection

                    edge_labels.append(edge_label)

    # Convert edge_index to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_labels = torch.tensor(edge_labels, dtype=torch.float)

    # Subset features and labels for the selected nodes
    new_x = feature_matrix[all_nodes]
    new_y = data.y[all_nodes] if data.y is not None else None

    # Create a new Data object with the new features, edge index, and labels
    new_data = Data(
        x=new_x,
        y=new_y,
        edge_index=edge_index,
        edge_labels=edge_labels
    )

    return new_data



parser = argparse.ArgumentParser("Building KNN graph with labelled data")

parser.add_argument("-k", "--knn", default=10, help="Number of neighbours")
parser.add_argument("-l", "--load", default="", help="Path of datafile to load")
parser.add_argument("-s", "--save", default="", help="Path of savefile")

args = parser.parse_args()

graph_data = []

if args.load != "":
    print(f"Loading seed data from {args.load}...")
    with open(args.load, 'rb') as f:
        evt_data = pickle.load(f)
        
for i, data in enumerate(evt_data):
    print("Building graph for event ", i)
    graph_data.append(knn_data(data, int(args.knn)))

if args.save != "":
    print(f"Saving graph data to {args.save}...")
    with open(args.save, 'wb') as f:
        pickle.dump(graph_data, f)
    
    

