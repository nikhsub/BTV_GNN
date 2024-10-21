import warnings
warnings.filterwarnings("ignore")


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
import joblib


parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-had", "--hadron", default=False, action='store_true', help="Testing on hadron level?")
parser.add_argument("-t",  "--thres", default=0.5, help="Threshold for plotting graph")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

if args.load_train != "":
    print(f"Loading testing data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

def scale_data(data_list, scaler):
    for data in data_list:
        data.x = torch.tensor(scaler.transform(data.x), dtype=torch.float)
    return data_list

scaler = joblib.load('scaler.pkl')

val_graphs = train_graphs[-220:-210]

val_graphs = scale_data(val_graphs, scaler)

model = GNNModel(len(trk_features), 512)

model.load_state_dict(torch.load(args.load_model))

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def plot_graphs(data, probs, index, thres):
    # Extract eta and phi from data.x (assuming eta is at [:, 0] and phi is at [:, 1])
    eta = data.x[:, 0].cpu().numpy()  # Converting to NumPy for plotting
    phi = data.x[:, 1].cpu().numpy()

    # Convert signal indices to NumPy for easier handling
    siginds = data.siginds.cpu().numpy()

    # Create a figure for plotting
    plt.figure(figsize=(10, 10))

    # Plot all tracks as blue dots
    plt.scatter(eta, phi, color='blue', label='Background Tracks')

    # Highlight signal tracks as red dots
    plt.scatter(eta[siginds], phi[siginds], color='red', label='Signal Tracks', marker='x')

    # Draw edges between the tracks
    for i, (start, end) in enumerate(index.T.cpu().numpy()):
        # Draw a line between the start and end points
        edge_prob = probs[i].item()
        if edge_prob > thres:
            plt.plot([eta[start], eta[end]], [phi[start], phi[end]], 'k-', alpha=0.5)  # Grey lines

            # Display the edge probability along the edge
            midpoint_eta = (eta[start] + eta[end]) / 2
            midpoint_phi = (phi[start] + phi[end]) / 2
            plt.text(midpoint_eta, midpoint_phi, f'{edge_prob:.2f}', fontsize=8, color='green')

    # Add plot labels and legends
    plt.xlabel('Eta')
    plt.ylabel('Phi')
    plt.title(f'Event {data.evt}')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Show the plot
    plt.savefig(f'Evt{data.evt}_graph.png', bbox_inches='tight')
    plt.close()

def plot_graphs_had(data, probs, index, thres):
    num_nodes = data.x.size(0)
    half_num_nodes = num_nodes // 2

    # Generate circular positions for the nodes (using polar coordinates)
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)

    # Create a figure for plotting
    plt.figure(figsize=(10, 10))

    # Plot signal nodes as red dots (first half of data.x)
    plt.scatter(x_pos[:half_num_nodes], y_pos[:half_num_nodes], color='red', label='Signal Tracks', marker='x')

    # Plot background nodes as blue dots (second half of data.x)
    plt.scatter(x_pos[half_num_nodes:], y_pos[half_num_nodes:], color='blue', label='Background Tracks')

    # Draw edges between the tracks based on edge_probs and threshold
    for i, (start, end) in enumerate(index.T.cpu().numpy()):
        edge_prob = probs[i].item()
        if edge_prob > thres:
            plt.plot([x_pos[start], x_pos[end]], [y_pos[start], y_pos[end]], 'k-', alpha=0.5)
            midpoint_x = (x_pos[start] + x_pos[end]) / 2
            midpoint_y = (y_pos[start] + y_pos[end]) / 2
            plt.text(midpoint_x, midpoint_y, f'{edge_prob:.2f}', fontsize=8, color='green')

    # Add plot labels, legends, and grid
    plt.title(f'Event {data.evt} Hadron {data.had}')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(f'Evt{data.evt}Had{data.had}_graph.png', bbox_inches='tight')
    plt.close()


def old_plot_hists(data, probs, index):
    siginds = torch.tensor(data.siginds, dtype=torch.int16, device=edge_index.device)

    #print(siginds)

    # Create masks for signal-to-signal edges and background edges
    edge_start, edge_end = edge_index[0], edge_index[1]
    signal_mask_start = torch.isin(edge_start, siginds)
    signal_mask_end = torch.isin(edge_end, siginds)
    signal_mask = signal_mask_start & signal_mask_end

    # Separate signal-to-signal and background edge probabilities
    signal_edge_probs = probs[signal_mask].cpu().numpy()
    print(signal_edge_probs)
    bg_edge_probs = probs[~signal_mask].cpu().numpy()
    print(bg_edge_probs)
    

    # Plot histogram for signal-to-signal edges
    plt.figure(figsize=(8, 6))

    signal_counts, _, _ = plt.hist(signal_edge_probs, bins=20, color='red', alpha=0.7, label='Signal-Signal Edges', density=True)
    plt.hist(bg_edge_probs, bins=20, color='blue', alpha=0.5, label='Background Edges', density=True)

    # Add labels, title, and legend
    #plt.yscale('log')
    plt.xlabel('Edge Probability')
    plt.ylabel('Normalized count')
    plt.title(f'Event {data.evt} - Edge Probability Distribution')
    plt.legend(loc='upper right')

    # Save the plot as a PNG file
    plt.savefig(f'Evt{data.evt}_hist.png', bbox_inches='tight')
    plt.close()


def plot_hists(data, probs, index):
    siginds = torch.tensor(data.siginds, dtype=torch.int16, device=edge_index.device)
    
    edge_start, edge_end = edge_index[0], edge_index[1]
    signal_mask_start = torch.isin(edge_start, siginds)
    signal_mask_end = torch.isin(edge_end, siginds)

    # Mask for signal-to-signal edges
    signal_mask = signal_mask_start & signal_mask_end
    signal_edge_probs = probs[signal_mask].cpu().numpy()
    print(signal_edge_probs)

    # Mask for signal-to-background edges
    background_mask_start = signal_mask_start & ~signal_mask_end
    background_mask_end = signal_mask_end & ~signal_mask_start
    signal_background_mask = background_mask_start | background_mask_end
    signal_to_bg_edge_probs = probs[signal_background_mask].cpu().numpy()

    # Mask for background edges
    bg_edge_probs = probs[~signal_mask & ~signal_background_mask].cpu().numpy()

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 18))

    # Histogram for Signal-Signal Edges
    axs[0].hist(signal_edge_probs, bins=10, color='red', alpha=0.7, label='Signal-Signal Edges', density=True)
    axs[0].set_xlabel('Edge Probability')
    axs[0].set_ylabel('Normalized Count')
    axs[0].set_title(f'Signal-Signal Edge Probability Distribution (Event {data.evt})')
    axs[0].legend(loc='upper right')

    # Histogram for Background Edges
    axs[1].hist(bg_edge_probs, bins=20, color='blue', alpha=0.5, label='Background Edges', density=True)
    axs[1].set_xlabel('Edge Probability')
    axs[1].set_ylabel('Normalized Count')
    axs[1].set_title(f'Background Edge Probability Distribution (Event {data.evt})')
    axs[1].legend(loc='upper right')

    # Histogram for Signal to Background Edges
    axs[2].hist(signal_to_bg_edge_probs, bins=20, color='green', alpha=0.5, label='Signal to Background Edges', density=True)
    axs[2].set_xlabel('Edge Probability')
    axs[2].set_ylabel('Normalized Count')
    axs[2].set_title(f'Signal to Background Edge Probability Distribution (Event {data.evt})')
    axs[2].legend(loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f'Evt{data.evt}_hists.png', bbox_inches='tight')
    plt.close()



def plot_hists_had(data, probs, labels):

    edge_probs_flat = probs.view(-1)
    edge_labels_flat = labels.view(-1)   
    signal_indices = (edge_labels_flat == 0.99).nonzero(as_tuple=True)[0]
    background_indices = ((edge_labels_flat == 0.01)| (edge_labels_flat == 0.1)).nonzero(as_tuple=True)[0]
    bg_edge_probs = edge_probs_flat[background_indices].cpu().numpy()
    signal_edge_probs = edge_probs_flat[signal_indices].cpu().numpy()

    plt.figure(figsize=(8, 6))
    
    plt.hist(bg_edge_probs, bins=5, color='blue', alpha=0.5, label='Background Edges', density=False)
    signal_counts, _, _ = plt.hist(signal_edge_probs, bins=10, color='red', alpha=0.7, label='Signal-Signal Edges', density=False)
    #plt.hist(bg_edge_probs, bins=5, color='blue', alpha=0.5, label='Background Edges', density=False)

    # Add labels, title, and legend
    #plt.yscale('log')
    plt.ylim(0, max(signal_counts.max(), bg_edge_probs.size) * 1.1)
    plt.xlabel('Edge Probability')
    plt.ylabel('Normalized count')
    plt.title(f'Hadron {data.had} Event {data.evt} - Edge Probability Distribution')
    plt.legend(loc='upper right')

    # Save the plot as a PNG file
    plt.savefig(f'Evt{data.evt}Had{data.had}_hist.png', bbox_inches='tight')
    plt.close()


    
for i, data in enumerate(val_graphs):
    with torch.no_grad():

        data = data.to(device)
        hadron = args.hadron
        thres = float(args.thres)

        print(f"Plotting for Evt/Had {i}....")

        #plot_graphs(data, edge_probs, edge_index, 0.95)
        if not args.hadron: 
            edge_probs, edge_index = model(data, device, hadron)
            print("PROBS", edge_probs)
            print("LABELS", edge_labels)
            #print("INDEX", edge_index)
            #print("SIGINDS", data.siginds)
            plot_hists(data, edge_probs, edge_index)
            #plot_graphs(data, edge_probs, edge_index, thres)

        else: 
            edge_probs, edge_labels, edge_index = model(data, device, hadron)
            print("PROBS", edge_probs)
            print("LABELS", edge_labels)
            #print("INDEX", edge_index)
            plot_hists_had(data, edge_probs, edge_labels)
            #plot_graphs_had(data, edge_probs, edge_index, thres)








