import argparse
import uproot
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F
import math
from model import *
import pprint
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

def plot_hists(edge_probs, edge_labels):
    # Convert to numpy arrays for easier handling in matplotlib
    edge_probs_np = edge_probs_flat.cpu().numpy()
    edge_labels_np = edge_labels_flat.cpu().numpy()
    
    # Create groups based on labels
    bb_probs = edge_probs_np[edge_labels_np == 0.01]
    sb_probs = edge_probs_np[edge_labels_np == 0.1]
    ssx_probs = edge_probs_np[edge_labels_np == 0.5]
    ss_probs = edge_probs_np[edge_labels_np == 0.99]
    
    # Plot the histograms
    plt.figure(figsize=(10, 6))
    
    # Define bins for the histograms
    bins = 30  # Adjust this number to change the bin size
    
    # Plot histograms for each category
    plt.hist(bb_probs, bins=bins, color='blue', alpha=0.6, label='bb (background-to-background)')
    plt.hist(sb_probs, bins=bins, color='orange', alpha=0.6, label='sb (signal-to-background)')
    plt.hist(ssx_probs, bins=bins, color='red', alpha=0.6, label='ssx (incorrect signal-to-signal)')
    plt.hist(ss_probs, bins=bins, color='green', alpha=0.6, label='ss (correct signal-to-signal)')
    
    # Add labels and legend
    plt.xlabel("Edge Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Edge Probabilities by Edge Type")
    plt.legend()
    
    # Show the plot
    plt.show()



if args.load_train != "":
    print(f"Loading testing data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

train_graphs = train_graphs[-500:-300]



model = GNNModel(len(trk_features), 1024)

model.load_state_dict(torch.load(args.load_model))

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

    
for data in train_graphs:
    with torch.no_grad():

        total_correct_signal_edges = 0  # To track the number of correctly identified signal edges
        total_signal_edges = 0  # To track the total number of signal edges across all samples
        background_edge_probs = []  # To track edge probs for background edges
        sig_edge_probs = []

        data = data.to(device)

        edge_probs, edge_labels = model(data, device)

        edge_probs_flat = edge_probs.view(-1)
        edge_labels_flat = edge_labels.view(-1)

        signal_indices = (edge_labels_flat == 0.99).nonzero(as_tuple=True)[0]
        n_signal_edges = signal_indices.size(0)

        total_signal_edges += n_signal_edges

        if n_signal_edges > 0:
            top_n_indices = torch.topk(edge_probs_flat, n_signal_edges).indices

            correct_signal_edges = (torch.isin(top_n_indices, signal_indices)).sum().item()

            total_correct_signal_edges += correct_signal_edges

        background_indices = (edge_labels_flat == 0.01).nonzero(as_tuple=True)[0]
        background_edge_probs.extend(edge_probs_flat[background_indices].cpu().numpy())
        sig_edge_probs.extend(edge_probs_flat[signal_indices].cpu().numpy())

        mean_background_edge_prob = np.mean(background_edge_probs) if background_edge_probs else 0
        mean_signal_edge_prob = np.mean(sig_edge_probs) if sig_edge_probs else 0

        correct_signal_percentage = (total_correct_signal_edges / total_signal_edges) * 100 if total_signal_edges > 0 else 0

        print(f"Current % of correctly predicted signal edges: {correct_signal_percentage}")

        print(f"Mean signal background edge probability: {mean_signal_edge_prob}")

        print(f"Mean background edge probability: {mean_background_edge_prob}")

        #print(np.array2string(np.array(edge_probs.tolist()), separator=', '))

        print(edge_probs_flat)
        print(edge_labels_flat)

        plot_hists(edge_probs_flat, edge_labels_flat)

        print("NEXT")





