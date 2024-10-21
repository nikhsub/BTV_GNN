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
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import torch.nn.functional as F
import math
from model import *
import joblib

parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-lh", "--load_had", default="", help="Load training hadron data from a file")
parser.add_argument("-le", "--load_evt", default="", help="Load validation event data from a file")

args = parser.parse_args()

glob_beta = [2, 1, 1]
glob_gamma = 1
glob_alpha = 1

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

#LOADING DATA
if args.load_had != "":
    print(f"Loading training data from {args.load_had}...")
    with open(args.load_had, 'rb') as f:
        train_hads = pickle.load(f)

if args.load_evt != "":
    print(f"Loading validation data from {args.load_evt}...")
    with open(args.load_evt, 'rb') as f:
        val_data = pickle.load(f)

#DATA SIZE
train_hads = train_hads[:6000]
val_data   = val_data[-10:]

#SPLITTING, SCALING AND LOADING
def scale_data(data_list, scaler):
    for data in data_list:
        data.x = torch.tensor(scaler.transform(data.x), dtype=torch.float)
    return data_list

train_len = int(0.8 * len(train_hads))
train_data, test_data = random_split(train_hads, [train_len, len(train_hads) - train_len])

scaler = StandardScaler()
for data in train_data:
    scaler.partial_fit(data.x)

joblib.dump(scaler, 'scaler.pkl')

train_data = scale_data(train_data, scaler)
test_data = scale_data(test_data, scaler)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
val_loader  = DataLoader(val_data, batch_size=1, shuffle=False)

#DEVICE AND MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNModel(indim=len(trk_features), outdim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

#def compute_class_weights(edge_labels):
#
#    global glob_sf
#
#    num_samples = edge_labels.size(0)
#
#    # Count occurrences of each edge label value in a single pass
#    counts = {
#        "bb": (edge_labels == 0.01).sum().float(),
#        "sb": (edge_labels == 0.1).sum().float(),
#        "ssx": (edge_labels == 0.5).sum().float(),
#        "ss": (edge_labels == 0.99).sum().float()
#    }
#
#    # Precompute weights for each label value, avoid division by zero
#    weights = {
#        "bb": num_samples / counts["bb"] if counts["bb"] > 0 else 0,
#        "sb": num_samples / counts["sb"] if counts["sb"] > 0 else 0,
#        "ssx": (num_samples / counts["ssx"] if counts["ssx"] > 0 else 0) * glob_sf,
#        "ss": (num_samples / counts["ss"] if counts["ss"] > 0 else 0) * glob_sf
#    }
#
#    # Vectorized assignment of class weights
#    class_weights = torch.zeros_like(edge_labels).float()
#    for label_value, weight in zip([0.01, 0.1, 0.5, 0.99], weights.values()):
#        class_weights[edge_labels == label_value] = weight
#
#    # Scale the weights by the scaling factor
#    return class_weights


def focal_loss(edge_probs, edge_labels, epoch, gamma=glob_gamma, alpha=glob_alpha, reg=0.5) :
    bce_loss = nn.BCELoss(reduction='none')(edge_probs, edge_labels)
    pt = edge_probs * edge_labels + (1 - edge_probs) * (1 - edge_labels)
    focal_weight = (1 - pt) ** gamma
    alpha_weight = edge_labels * alpha + (1 - edge_labels) * (1 - alpha)
    focal_loss = alpha_weight * focal_weight * bce_loss

    regularization_loss = reg * ((edge_probs - 0.5) ** 2).sum()

    #print("Focal loss", focal_loss.mean().item())
    #print("Reg loss", regularization_loss.item())


    return focal_loss.mean()


def custom_loss(edge_probs, edge_labels, beta, epoch, scale=1.0):
    bce_loss = nn.BCELoss(reduction='none')(edge_probs, edge_labels)

    num_edges = edge_labels.size(0)

    signal_to_signal_mask = (edge_labels == 0.99).float()
    signal_to_background_mask = (edge_labels == 0.01).float()
    background_to_background_mask = (edge_labels == 0.001).float()

    missed_signal_to_signal = signal_to_signal_mask * (1 - edge_probs)
    incorrect_signal_to_background = signal_to_background_mask * edge_probs
    incorrect_background_to_background = background_to_background_mask * edge_probs

    loss_mag = torch.pow(10, torch.floor(torch.log10(bce_loss.mean())))


    penalty = (beta[0] * missed_signal_to_signal.sum() +
               beta[1] * incorrect_signal_to_background.sum() +
               beta[2] * incorrect_background_to_background.sum())/num_edges

    pen_mag = torch.pow(10, torch.floor(torch.log10(penalty)))

    penalty = penalty*(scale*loss_mag/pen_mag)

    epsilon = 1e-5
    reg_loss = -1 * ((edge_probs * torch.log(edge_probs + epsilon)) +
                                        ((1 - edge_probs) * torch.log(1 - edge_probs + epsilon))).sum()

    reg_mag = torch.pow(10, torch.floor(torch.log10(reg_loss)))

    reg_loss = reg_loss*(scale*loss_mag/reg_mag)

    if(torch.isnan(reg_loss)): reg_loss = 0
    
    total_loss = bce_loss.mean() + reg_loss + penalty

    #print("BCE", bce_loss.mean())
    #print("Penalty", penalty)
    #print("Reg Loss", reg_loss)
    #print("Total loss", total_loss)
    #print("NEXT")

    return total_loss

#TRAIN
def train(model, train_loader, optimizer, device, epoch):
    global glob_beta

    model.to(device)
    model.train()
    total_loss=0
    nosigseeds = 0
    nosigevts = 0

    print("Current weights", glob_beta)

    for data in tqdm(train_loader, desc="Training", unit="Hadron"):
        data = data.to(device)

        if(data.seeds.size(0) < 3): 
            nosigseeds+=1
            continue

        optimizer.zero_grad()
        
        edge_probs, edge_labels, _ = model(data, device, True) #True for training/testing, false for validation on evt level
        
        if(edge_labels.shape[0] == 0): 
            nosigevts+=1
            continue

        #num_signal_edges = torch.sum(edge_labels == 0.99).item() +  torch.sum(edge_labels == 0.5).item()
        #num_bkg_edges = torch.sum(edge_labels == 0.1).item() +  torch.sum(edge_labels == 0.01).item()
        ##rat = num_signal_edges/num_bkg_edges
        
        #if(rat < 0.02):
        #    nosigevts+=1
        #    continue

        #print(edge_labels)
        #print(edge_probs.squeeze())

        #print(np.mean(np.array(edge_probs.squeeze().tolist())))

        #print(np.array2string(np.array(edge_probs.tolist()), separator=', '))

        #loss_fn = nn.BCELoss()
        loss = custom_loss(edge_probs.squeeze(), edge_labels, glob_beta, epoch)
        #loss = loss_fn(edge_probs.squeeze(), edge_labels)
        
        loss.backward()

        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(f"Gradient for {name}: {param.grad}")
        #    else:
        #        print(f"Gradient for {name}: None")


        optimizer.step()
        total_loss+=loss.item()

    print(f"No seed hadrons: {nosigseeds}")
    print(f"No output edges: {nosigevts}")

    return total_loss / len(train_loader)


def test(model, test_loader, device, epoch):
    model.to(device)
    model.eval()
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(f"Parameter: {name}, Shape: {param.shape}")
    #        print(param)  # Uncomment this line to also print the values of the parameters

    total_loss = 0
    total_correct_signal_edges = 0  # To track the number of correctly identified signal edges
    total_signal_edges = 0  # To track the total number of signal edges across all samples
    notestevts = 0
    background_edge_probs = []  # To track edge probs for background edges
    sig_edge_probs = []
    probs = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", unit="Hadron"):
            data = data.to(device)
            edge_probs, edge_labels, _ = model(data, device, True)

            #print(edge_probs)
            #print(edge_labels)

            if(edge_labels.shape[0]==0): 
                notestevts+=1
                continue

            loss_fn = nn.BCELoss()
            loss = loss_fn(edge_probs.squeeze(), edge_labels)
            total_loss += loss.item()

            edge_probs_flat = edge_probs.view(-1)
            probs.append(edge_probs_flat)
            edge_labels_flat = edge_labels.view(-1)

            #signal_indices = (edge_labels_flat == 0.99).nonzero(as_tuple=True)[0]
            #n_signal_edges = signal_indices.size(0)

            #total_signal_edges += n_signal_edges

            #if n_signal_edges > 0:
            #    top_n_indices = torch.topk(edge_probs_flat, n_signal_edges).indices

            #    correct_signal_edges = (torch.isin(top_n_indices, signal_indices)).sum().item()

            #    total_correct_signal_edges += correct_signal_edges

            #background_indices = ((edge_labels_flat == 0.001)| (edge_labels_flat == 0.01)).nonzero(as_tuple=True)[0]
            #background_edge_probs.extend(edge_probs_flat[background_indices].cpu().numpy())
            #sig_edge_probs.extend(edge_probs_flat[signal_indices].cpu().numpy())
            

    #mean_background_edge_prob = np.mean(background_edge_probs) if background_edge_probs else 0
    #mean_signal_edge_prob = np.mean(sig_edge_probs) if sig_edge_probs else 0

    #correct_signal_percentage = (total_correct_signal_edges / total_signal_edges) * 100 if total_signal_edges > 0 else 0
    #
    ##if(epoch %2 ==0 and epoch>0): update_sf(correct_signal_percentage)

    #print(f"Number of hadrons with no edges: {notestevts}")

    #print(f"Current % of correctly predicted signal edges: {correct_signal_percentage}")

    #print(f"Mean SIGNAL edge probability: {mean_signal_edge_prob}")
    #print(f"Number of SIGNAL edges: {len(sig_edge_probs)}")

    #print(f"Mean BACKGROUND edge probability: {mean_background_edge_prob}")
    #print(f"Number of BACKGROUND edges: {len(background_edge_probs)}")

    print(probs)

    return total_loss / len(test_loader), 0, 0 #mean_signal_edge_prob, mean_background_edge_prob

def validate(model, val_loader, device, epoch):

    for data in tqdm(val_loader, desc="Validation", unit="Event"):
            data = data.to(device)
            edge_probs, edge_index = model(data, device, False)

            siginds = data.siginds


            edge_start, edge_end = edge_index[0], edge_index[1]

            # Create a mask to identify signal-to-signal edges based on siginds
            signal_mask_start = torch.isin(edge_start, siginds)
            signal_mask_end = torch.isin(edge_end, siginds)
            signal_mask = signal_mask_start & signal_mask_end  # Both start and end should be signal indices

            # Separate signal edge probabilities and background edge probabilities
            signal_edge_probs = edge_probs[signal_mask]
            bg_edge_probs = edge_probs[~signal_mask]  # Inverse mask for background edges

            # Display signal edge probabilities
            if len(signal_edge_probs) > 0:
                print("EVENT VALIDATION")
                print(f"Number of signal edges: {signal_edge_probs.size(0)}")
                avg_sig_prob = signal_edge_probs.mean().item()
                print(f"\nEpoch {epoch} - Average signal Edge Probability: {avg_sig_prob}")

            # Compute and display average background edge probability
            if len(bg_edge_probs) > 0:
                avg_bg_prob = bg_edge_probs.mean().item()
                print(f"Epoch {epoch} - Average Background Edge Probability: {avg_bg_prob}")
                print("NEXT")



           
best_test_loss = float('inf')

for epoch in range(int(args.epochs)):
    #print("Global alpha", glob_alpha)
    #print("Global gamma", glob_gamma)
    train_loss = train(model, train_loader,  optimizer, device, epoch)
    test_loss, sig_prob, bkg_prob = test(model, test_loader,  device, epoch)
    #if(bkg_prob > 0.95):
    #    glob_beta = [glob_beta[0], glob_beta[1]*2, glob_beta[2]*2]
    #if(sig_prob < 0.05):
    #    glob_beta = [glob_beta[0]*2, glob_beta[1], glob_beta[2]]
    #if((test_loss > best_test_loss and epoch > 0) or sig_prob < 0.001):
    #    glob_beta = [glob_beta[0]*1.5, glob_beta[1]*1.25, glob_beta[2]*2]
    #validate(model, val_loader, device, epoch)
    #glob_weight = glob_weight*2

    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    if((epoch+1)%5==0):
        glob_beta[0] += 0.1
        glob_beta[1] += 0.4
        glob_beta[2] += 0.2

    if(epoch> 0 and epoch%10==0):
        savepath = os.path.join("model_files", f"model_{args.modeltag}_e{epoch}.pth")
        torch.save(model.state_dict(), savepath)
        print(f"Model saved to {savepath}")


    if test_loss < best_test_loss and epoch > 0: #and sig_prob > bkg_prob:
        best_test_loss = test_loss
        save_path = os.path.join("model_files", f"model_{args.modeltag}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

