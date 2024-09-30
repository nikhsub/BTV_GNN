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

parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")

args = parser.parse_args()

glob_sf = 1
prev_acc = 0.0

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

if args.load_train != "":
    print(f"Loading training data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

train_graphs = train_graphs[:1000]


train_len = int(0.8 * len(train_graphs))
train_data, test_data = random_split(train_graphs, [train_len, len(train_graphs) - train_len])

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNModel(indim=len(trk_features), outdim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def compute_class_weights(edge_labels):

    global glob_sf

    num_samples = edge_labels.size(0)

    # Count occurrences of each edge label value in a single pass
    counts = {
        "bb": (edge_labels == 0.01).sum().float(),
        "sb": (edge_labels == 0.1).sum().float(),
        "ssx": (edge_labels == 0.5).sum().float(),
        "ss": (edge_labels == 0.99).sum().float()
    }

    # Precompute weights for each label value, avoid division by zero
    weights = {
        "bb": num_samples / counts["bb"] if counts["bb"] > 0 else 0,
        "sb": num_samples / counts["sb"] if counts["sb"] > 0 else 0,
        "ssx": (num_samples / counts["ssx"] if counts["ssx"] > 0 else 0) * glob_sf,
        "ss": (num_samples / counts["ss"] if counts["ss"] > 0 else 0) * glob_sf
    }

    # Vectorized assignment of class weights
    class_weights = torch.zeros_like(edge_labels).float()
    for label_value, weight in zip([0.01, 0.1, 0.5, 0.99], weights.values()):
        class_weights[edge_labels == label_value] = weight

    # Scale the weights by the scaling factor
    return class_weights


def custom_loss(edge_probs, edge_labels, class_weights, epoch, reg_factor=0.2):

    loss_per_edge = (edge_probs - edge_labels) ** 2

    weighted_loss = loss_per_edge * class_weights

    reg_loss = reg_factor * torch.sum((edge_probs < 0.01).float() * edge_probs ** 2)

    total_loss = weighted_loss.mean() + reg_loss

    return total_loss

def focal_loss(edge_probs, edge_labels, class_weights, epoch, gamma=4.0, alpha=1.0, eps=1e-8, reg_factor=0.03) :
    pt = torch.where(edge_labels == 0.99, edge_probs, 1 - edge_probs)
    pt = torch.clamp(pt, min=eps, max=1.0 - eps)
    #gamma = 1 + (3.0 - 1.0) * (epoch / int(args.epochs))
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)

    weighted_loss = loss * class_weights
    reg_loss = reg_factor * torch.sum((edge_probs < 0.0001).float() * edge_probs ** 2)
    return weighted_loss.mean() + reg_loss

def train(model, train_loader, optimizer, device, epoch):
    global glob_sf

    model.to(device)
    model.train()
    total_loss=0
    nosigevts = 0

    print("Current signal scale factor: ", glob_sf)

    for data in tqdm(train_loader, desc="Training", unit="Event"):
        data = data.to(device)

        optimizer.zero_grad()
        
        #with torch.profiler.profile() as prof:
        edge_probs, edge_labels = model(data, device)

        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
        if(edge_labels.shape[0] == 0): continue

        num_signal_edges = torch.sum(edge_labels == 0.99).item() +  torch.sum(edge_labels == 0.5).item()
        num_bkg_edges = torch.sum(edge_labels == 0.1).item() +  torch.sum(edge_labels == 0.01).item()
        rat = num_signal_edges/num_bkg_edges
        
        #if(rat < 0.02):
        #    nosigevts+=1
        #    continue

        #print(edge_labels)
        #print(edge_probs.squeeze())

        #print(np.mean(np.array(edge_probs.squeeze().tolist())))

        #print(np.array2string(np.array(edge_probs.tolist()), separator=', '))

        class_weights = compute_class_weights(edge_labels)

        class_weights = class_weights.to(device)


        loss = custom_loss(edge_probs, edge_labels, class_weights, epoch)


        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    print(f"Events without minimum sig edges = {nosigevts}")
    
    return total_loss / len(train_loader)


def update_sf(current_acc):
    global glob_sf, prev_acc
    
    glob_sf +=2

    glob_sf = min(100, glob_sf)

    #if(current_acc <= prev_acc): glob_sf+=1
    #else: 
    #    glob_sf = max(2, glob_sf-1)

    prev_acc = current_acc



def test(model, test_loader, device, epoch):
    model.to(device)
    model.eval()

    total_loss = 0
    total_correct_signal_edges = 0  # To track the number of correctly identified signal edges
    total_signal_edges = 0  # To track the total number of signal edges across all samples
    background_edge_probs = []  # To track edge probs for background edges
    sig_edge_probs = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", unit="Event"):
            data = data.to(device)
            edge_probs, edge_labels = model(data, device)

            if(edge_labels.shape[0]==0): continue

            class_weights = compute_class_weights(edge_labels)

            class_weights = class_weights.to(device)

            loss = custom_loss(edge_probs, edge_labels, class_weights, epoch)
            total_loss += loss.item()

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
    
    #if(epoch %2 ==0 and epoch>0): update_sf(correct_signal_percentage)

    print(f"Current % of correctly predicted signal edges: {correct_signal_percentage}")

    print(f"Mean signal edge probability: {mean_signal_edge_prob}")

    print(f"Mean background edge probability: {mean_background_edge_prob}")


    return total_loss / len(test_loader)
           
best_test_loss = float('inf')

for epoch in range(int(args.epochs)):

    train_loss = train(model, train_loader,  optimizer, device, epoch)
    test_loss = test(model, test_loader,  device, epoch)

    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    if(epoch> 0 and epoch%10==0):
        savepath = os.path.join("files_models", f"model_{args.modeltag}_e{epoch}.pth")
        torch.save(model.state_dict(), savepath)
        print(f"Model saved to {savepath}")


    if test_loss < best_test_loss and epoch > 0:
        best_test_loss = test_loss
        save_path = os.path.join("files_models", f"model_{args.modeltag}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

