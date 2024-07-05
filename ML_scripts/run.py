import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.nn import EdgeConv, global_max_pool
from torch.nn import Sequential, Linear, ReLU, CosineSimilarity
import torch_cluster
from torch_cluster import knn_graph
import math
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.preprocessing import StandardScaler
import networkx as nx
from torch.nn.functional import cosine_similarity
from torch.nn import Linear
from gnn_model import GNN
import random
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-i", "--inp", default="", help="Input csv file containing training samples")
parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-tr", "--train", default=False, action="store_true", help="Training")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-tt", "--test", default=False, action="store_true", help="Testing")
parser.add_argument("-m", "--model", default="", help="Path to model for testing")
parser.add_argument("-nt", "--numtest", default=10, help="Number of test samples to run on")

args = parser.parse_args()

df = pd.read_csv(args.inp)

num_edges = 6 #Num edges to display in visualization
#Hyperparameters
num_features = 5
knn_edges = 10
init_costhres = 0.6
init_beta = [0.1,0,3]
beta0 = 0
beta1 = 0
beta2 = 4

signal_rows_with_nan = df[(df['is_signal'] == 1) & df.isna().any(axis=1)]

if signal_rows_with_nan.empty:
    print("Removing background rows with NaN values....")
    df.dropna(inplace=True)
else:
    print("Signal rows with NaN values:\n", signal_rows_with_nan)
    print("Removing these and other rows with NaN values....")
    df.dropna(inplace=True)


def create_graphs(df):
    graphs = []
    for bhad_num in df['bhad_num'].unique():
        sub_df = df[df['bhad_num'] == bhad_num]

        if (sub_df[sub_df['is_signal'] == 1].shape[0]) < 3: #Atleast 3 signal tracks per training sample
            continue

        node_features = sub_df[['trks_pt', 'trks_eta', 'trks_phi', 'trks_ip2d', 'trks_ip3d']].values

        x = torch.tensor(node_features, dtype=torch.float)
        signal_label = torch.tensor(sub_df['is_signal'].values, dtype=torch.long)  # Convert to tensor of type long

        edge_index = knn_graph(x, k=12, loop=False)
        edge_labels = torch.zeros(edge_index.size(1), dtype=torch.float)

        for i in range(edge_index.size(1)):
            node1 = edge_index[0, i]
            node2 = edge_index[1, i]
            if signal_label[node1] == 1 and signal_label[node2] == 1:
                edge_labels[i] = 1

        graph = Data(x=x, edge_index=edge_index, y=edge_labels, signal_label=signal_label.tolist())

        graphs.append(graph)

    return graphs


def new_create_graphs(df):

    event_graphs = []

    for evt_num, event_df in df.groupby('evt_num'):
        # Skip events with less than 3 signal tracks
        if event_df[event_df['is_signal'] == 1].shape[0] < 3:
            continue

        # Initialize lists to store the combined features and labels
        node_features_list = []
        edge_index_list = []
        edge_labels_list = []
        signal_labels_list = []

        node_offset = 0

        # Group by bhad_num within the event
        for bhad_num, sub_df in event_df.groupby('bhad_num'):
            if sub_df[sub_df['is_signal'] == 1].shape[0] < 3:  # At least 3 signal tracks per bhadron
                continue

            node_features = sub_df[['trks_pt', 'trks_eta', 'trks_phi', 'trks_ip2d', 'trks_ip3d', 'trks_ip2dsig', 'trks_ip3dsig']].values
            x = torch.tensor(node_features, dtype=torch.float)
            signal_label = torch.tensor(sub_df['is_signal'].values, dtype=torch.long)  # Convert to tensor of type long

            edge_index = knn_graph(x, k=knn_edges, loop=False)
            edge_labels = torch.zeros(edge_index.size(1), dtype=torch.float)

            # Set edge labels based on signal labels of nodes
            #for i in range(edge_index.size(1)):
            #    node1 = edge_index[0, i]
            #    node2 = edge_index[1, i]
            #    if signal_label[node1] == 1 and signal_label[node2] == 1:
            #        edge_labels[i] = 1


            # Set edge labels based on signal labels of nodes
            edge_labels[(signal_label[edge_index[0]] == 1) & (signal_label[edge_index[1]] == 1)] = 1

            edge_index += node_offset
            node_offset += x.size(0)

            node_features_list.append(x)
            edge_index_list.append(edge_index)
            edge_labels_list.append(edge_labels)
            signal_labels_list.append(signal_label)

        if node_features_list:
            x_combined = torch.cat(node_features_list, dim=0)
            edge_index_combined = torch.cat(edge_index_list, dim=1)
            edge_labels_combined = torch.cat(edge_labels_list, dim=0)
            signal_label_combined = torch.cat(signal_labels_list, dim=0)

            graph = Data(x=x_combined, edge_index=edge_index_combined, y=edge_labels_combined, signal_label=signal_label_combined.tolist())
            event_graphs.append(graph)

    return event_graphs


def update_graphs(high_dim_x, batch, costhreshold):
    cos = cosine_similarity  # Use cosine similarity in the high dimensional latent space to update edge connections between points in graph
    n_nodes = high_dim_x.shape[0]
    similarity_matrix = torch.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Only fill upper triangle
            similarity_matrix[i, j] = cos(high_dim_x[i].unsqueeze(0), high_dim_x[j].unsqueeze(0))

    similarity_matrix = similarity_matrix + similarity_matrix.t()
    edges = (similarity_matrix > costhreshold).nonzero(as_tuple=True) #Filter out edges
    edges = (edges[0][edges[0] != edges[1]], edges[1][edges[0] != edges[1]]) #Remove self loops
    edge_index = torch.stack(edges, dim=0) #Convert to edge index
    edge_labels = torch.zeros(edge_index.size(1), dtype=torch.float)
    sig_label_tensor = torch.tensor(batch.signal_label, dtype=torch.long).squeeze()
    sig_label = sig_label_tensor.squeeze().tolist()

    # Assign labels to edges based on the 'signal_label'
    for i in range(edge_index.size(1)):
        node1 = edge_index[0, i].item()
        node2 = edge_index[1, i].item()
        if sig_label[node1] == 1 and sig_label[node2] == 1:
            edge_labels[i] = 1

    # Create a new graph object
    graph = Data(x=batch.x, edge_index=edge_index, y=edge_labels, signal_label=sig_label)

    return graph

def adjust_beta(beta, epoch):
    # Example strategy: Increase the penalty for missed signal connections over epochs
    if epoch % 10 == 0 and epoch > 0:
        beta[0] += beta0 #Missed signal to signal
        beta[1] += beta1 #Signal to background
        beta[2] += beta2 #Background to background

    return beta

def custom_loss(predicted_edge_probs, edge_labels, signal_flags, beta):
    """
    Custom loss function to heavily penalize incorrect predictions involving signal tracks.

    Args:
    - predicted_edge_probs (torch.Tensor): The predicted probabilities of edges being present.
    - edge_labels (torch.Tensor): The ground truth labels for the edges (1 for present, 0 for absent).
    - signal_flags (torch.Tensor): Flags indicating whether each node in the edge pair is a signal (1) or background (0).
                                   This should be of shape [2, num_edges], where the first row is for source nodes and
                                   the second row is for target nodes of each edge.
    - beta (float): The factor by which to scale the penalty for incorrect signal connections.

    Returns:
    - torch.Tensor: The computed loss.
    """

    # Calculate base binary cross-entropy loss
    loss_func = torch.nn.BCELoss()
    bce_loss = loss_func(predicted_edge_probs, edge_labels)

    # Identify edges involving signal tracks
    # Both nodes are signal
    signal_to_signal = (signal_flags.sum(0) == 2).float()
    # One node is signal, one is background
    signal_to_background = (signal_flags.sum(0) == 1).float()
    # Both nodes are background
    background_to_background = (signal_flags.sum(0) == 0).float()


    # Calculate penalty terms
    # Should be signal to signal but isn't
    missed_sig_connections = signal_to_signal * edge_labels * (1 - torch.sigmoid(predicted_edge_probs))
    # Background to signal
    sig_bkg_connections = signal_to_background * torch.sigmoid(predicted_edge_probs)

    bkg_bkg_connections = background_to_background * torch.sigmoid(predicted_edge_probs)

    # Sum penalty terms with scaling
    penalty = (beta[0]*missed_sig_connections.sum()+ beta[1]*sig_bkg_connections.sum() + beta[2]*bkg_bkg_connections.sum())*bce_loss #Scale to bce_loss 

    return bce_loss + penalty

print("Creating graphs....")
graphs = create_graphs(df)

random.seed(a=5)
random.shuffle(graphs)
# Split data into training, validation, and test sets
train_split = int(0.70 * len(graphs))  # 70% for training
val_split = int(0.85 * len(graphs))   # Additional 15% for validation, total 85%

train_graphs = graphs[:train_split]
val_graphs = graphs[train_split:val_split]
test_graphs = graphs[val_split:]

train_data_list = []
val_data_list = []
test_data_list = []

scaler = StandardScaler()

print("Scaling data....")
# Normalize the training data
for train_g in train_graphs:
    train_x = train_g.x.numpy()
    normalized_train_x = scaler.fit_transform(train_x)
    normalized_train_x_tensor = torch.from_numpy(normalized_train_x).float()
    train_data_list.append(Data(x=normalized_train_x_tensor, edge_index=train_g.edge_index, y=train_g.y, signal_label=train_g.signal_label))

# Normalize the validation data
for val_g in val_graphs:
    val_x = val_g.x.numpy()
    normalized_val_x = scaler.transform(val_x)  # Use the scaler fitted on the training data
    normalized_val_x_tensor = torch.from_numpy(normalized_val_x).float()
    val_data_list.append(Data(x=normalized_val_x_tensor, edge_index=val_g.edge_index, y=val_g.y, signal_label=val_g.signal_label))

# Normalize the test data
for test_g in test_graphs:
    test_x = test_g.x.numpy()
    normalized_test_x = scaler.transform(test_x)
    normalized_test_x_tensor = torch.from_numpy(normalized_test_x).float()
    test_data_list.append(Data(x=normalized_test_x_tensor, edge_index=test_g.edge_index, y=test_g.y, signal_label=test_g.signal_label))

print("Creating dataloaders....")
# Create PyTorch-Geometric DataLoaders for batching
train_loader = DataLoader(train_data_list, batch_size=1, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data_list, batch_size=1, shuffle=False)


model = GNN(num_features)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
#scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# criterion = torch.nn.BCELoss()


#Training functions
def update_cos_threshold(thres, epoch):
      if epoch % 2 == 0:  # Even epoch numbers
        thres += 0.04
      else:  # Odd epoch numbers
        thres -= 0.02

    # Adjust if thres exceeds boundaries
      if thres > 0.9:
          thres = 0.9

      return thres


def validate(model, loader, beta):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # No gradients needed for validation
        for data in loader:
            data = data.to(device)
            high_dim_x, out = model(data.x, data.edge_index)
            if out.nelement() == 0:
                print("No output produced for a validation batch.")
                continue
            sig_label_tensor = torch.tensor(data.signal_label, dtype=torch.long).squeeze().to(device)
            signal_flags = torch.stack((sig_label_tensor[data.edge_index[0]], sig_label_tensor[data.edge_index[1]])).to(device)
            loss = custom_loss(out.squeeze(), data.y, signal_flags, beta)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_batch(data, beta):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)

    high_dim_x, out = model(data.x, data.edge_index)

     # Check if the output tensor 'out' is empty
    if out.nelement() == 0:
        print("No output produced. Skipping this batch.")
        return high_dim_x, None  # Return zero loss if no output

    # loss = criterion(out.squeeze(), data.y)
    sig_label_tensor = torch.tensor(data.signal_label, dtype=torch.long).squeeze().to(device)
    signal_flags = torch.stack((sig_label_tensor[data.edge_index[0]], sig_label_tensor[data.edge_index[1]])).to(device)
    loss = custom_loss(out.squeeze(), data.y, signal_flags, beta)
    loss.backward()
    optimizer.step()

    return high_dim_x, loss.item()

def plot_histogram(i, edge_probs, edge_labels):
    signal_probs = [prob for prob, label in zip(edge_probs, edge_labels) if label == 1]
    bkg_probs = [prob for prob, label in zip(edge_probs, edge_labels) if label == 0]

    plt.figure(figsize=(10, 6))
    plt.hist(signal_probs, bins=100, alpha=0.75, color='red', label='Signal')
    plt.hist(bkg_probs, bins=100, alpha=0.50, color='blue', label='Background')
    plt.title(f'Histogram of Edge Probabilities Graph {i}')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'pics/hist_{i}.png')
    plt.close()

def create_filtered_edge_index(edge_index, edge_probs, n):
    # Sort the edge probabilities and get the indices of the top n edges
    _, top_n_indices = torch.topk(edge_probs, n)
    filtered_edge_index = edge_index[:, top_n_indices]
    return filtered_edge_index

def visualize_graph(i, data, node_size=70, figsize=(24,24), scale=2):
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    num_nodes = data.x.shape[0]
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from([(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])])

    pos1 = {i: (data.x[i, 1]*scale, data.x[i, 2]*scale) for i in range(num_nodes)}  # Position nodes based on eta and phi

    pos2 = nx.circular_layout(G)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    nx.draw(G, pos1, ax=axes[0], node_color=data.signal_label, node_size=node_size, cmap=plt.cm.winter, with_labels=False, edge_color='k')
    axes[0].set_title(f"Eta-Phi space")

    nx.draw(G, pos2, ax=axes[1], node_color=data.signal_label, node_size=node_size, cmap=plt.cm.winter, with_labels=False, edge_color='k')
    axes[1].set_title(f"Circular layout")

    plt.suptitle(f"Graph Visualization - Top {num_edges} edges Graph {i}")
    plt.savefig(f"pics/graph_{i}.png")
    plt.close()

#Training
if(args.train):
    costhreshold = init_costhres
    num_epochs = args.epochs
    beta= init_beta
    saved = False
    print("Starting training...")
    for epoch in range(int(num_epochs)):
        new_graphs = []
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        tobreak = False
        print("Beta", beta)
        print("Cos thres",costhreshold)
        for batch in pbar:
            high_dim_x, loss = train_batch(batch, beta)
            if(loss == None):
              print("No output edges, saving model and exiting.")
              torch.save(model.state_dict(), f'models/model_exitepoch_{epoch}_{args.modeltag}_{timestamp}.pth')
              saved = True
              tobreak = True
              break
            else:
              newg = update_graphs(high_dim_x, batch, costhreshold)
              new_graphs.append(newg)
              total_loss += loss
              # Optional: display current loss in the progress bar
              pbar.set_postfix({'loss': loss})
        if(tobreak): break
        val_loss = validate(model, val_loader, beta)
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')
    
        train_loader = DataLoader(new_graphs, batch_size=1)
        costhreshold = update_cos_threshold(costhreshold, epoch)
        beta = adjust_beta(beta, epoch)
        # scheduler.step()
        # print(f'Current Learning Rate: {scheduler.get_last_lr()}')
    
        print(f'Epoch {epoch}, Average Training Loss: {total_loss / len(train_loader)}')
    
    if(not saved):
        torch.save(model.state_dict(), f'models/model_endepoch{epoch}_{args.modeltag}_{timestamp}.pth')

if(args.test):
    if(not args.train):
        print("Loading model", args.model)
        model.load_state_dict(torch.load(args.model))
        
    model.eval()
    probs = []
    # Disable gradient computation
    print("Testing and saving plots....")
    with torch.no_grad():
        for i,data in enumerate(test_loader):
            # Run the model forward pass
            if(i>int(args.numtest)): break
            feat, outprob = model(data.x, data.edge_index)
            # probs.extend(outprob.squeeze())
            plot_histogram(i,outprob.squeeze(), data.y)


            filt_edge_index = create_filtered_edge_index(data.edge_index, outprob.squeeze(), num_edges)
            visualize_graph(i, Data(data.x, edge_index=filt_edge_index, signal_label=data.signal_label))

