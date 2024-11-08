import warnings
warnings.filterwarnings("ignore")

import json
import argparse
import uproot
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
import pickle
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import torch.nn.functional as F
import math
from conmodel import *
import joblib
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-lh", "--load_had", default="", help="Load training hadron data from a file")

args = parser.parse_args()
glob_test_thres = 0.5

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

#LOADING DATA
if args.load_had != "":
    print(f"Loading training data from {args.load_had}...")
    with open(args.load_had, 'rb') as f:
        train_hads = pickle.load(f)

train_hads = train_hads[:] #Control number of input samples here - see array splicing for more

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

joblib.dump(scaler, 'scaler_'+args.modeltag+'.pkl')

train_data = scale_data(train_data, scaler)
test_data = scale_data(test_data, scaler)

train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

#DEVICE AND MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNModel(indim=len(trk_features), outdim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

def drop_random_edges(edge_index, drop_rate=0.2, device):
    """Randomly drop a percentage of edges from the graph"""
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=device) > drop_rate
    return edge_index[:, mask]

def contrastive_loss(x1, x2, temperature=0.5):
    """Compute the contrastive loss"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    batch_size = x1.size(0)

    similarity_matrix = torch.mm(x1, x2.t()) / temperature

    labels = torch.arange(batch_size).to(x1.device)
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss

def class_weighted_bce(preds, labels, pos_weight=3.0, neg_weight=1.0):
    """Class-weighted binary cross-entropy loss"""

    pos_weight = torch.tensor(pos_weight, device=preds.device)
    neg_weight = torch.tensor(neg_weight, device=preds.device)

    weights = torch.where(labels == 1, pos_weight, neg_weight)
    bce_loss = F.binary_cross_entropy(preds, labels, weight=weights)
    return bce_loss

#TRAIN
def train(model, train_loader, optimizer, device, epoch, drop_rate=0.5, temp=0.3, scale=2):

    model.to(device)
    model.train()
    total_loss=0
    total_node_loss = 0
    total_cont_loss = 0
    nosigseeds = 0

    for data in tqdm(train_loader, desc="Training", unit="Hadron"):
        data = data.to(device)

        if(data.seeds.size(0) < 3):
            nosigseeds+=1
            continue
        

        optimizer.zero_grad()
        num_nodes = data.x.size(0)
        edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2).t()

        edge_index1 = drop_random_edges(edge_index, drop_rate, device)
        #edge_index2 = drop_random_edges(edge_index, drop_rate)
        
        node_embeds1, preds1 = model(data, edge_index1, device)
        assert preds1.device == device, f"preds1 is not on the device: {preds1.device}"
        #node_embeds2, preds2 = model(data, edge_index2, device)

        #cont_loss = contrastive_loss(node_embeds1, node_embeds2, temp)
        cont_loss = torch.tensor(0.0, device=device)

        #node_loss = F.binary_cross_entropy(preds1, data.y.float().unsqueeze(1))
        node_loss = class_weighted_bce(preds1, data.y.float().unsqueeze(1))*scale
        #node_loss = torch.tensor(0.0, device=device) #Testing with only contloss

        loss = cont_loss + node_loss

        loss.backward()
        optimizer.step()

        total_loss      += loss.item()
        total_node_loss += node_loss.item()
        total_cont_loss += cont_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_node_loss = total_node_loss / len(train_loader)
    avg_cont_loss = total_cont_loss / len(train_loader)

    print(f"No seed hadrons: {nosigseeds}")

    return avg_loss, avg_node_loss, avg_cont_loss

#TEST
def test(model, test_loader, device, epoch, k=5, thres=0.5):
    model.to(device)
    model.eval()

    correct_bkg = 0
    total_bkg = 0
    correct_signal = 0
    total_signal = 0
    total_loss = 0
    mean_sig_prob = []
    mean_bkg_prob = []
    all_preds = []
    all_labels = []
    nosigtest = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="Hadron"):
            batch = batch.to(device)
            num_nodes = batch.x.size(0)

            if(batch.seeds.size(0) < 1):
                nosigtest+=1
                continue

            edge_index = knn_graph(batch.x, k=k, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)

            _, preds = model(batch, edge_index, device)

            all_preds.extend(preds.squeeze().cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            batch_loss = F.binary_cross_entropy(preds, batch.y.float().unsqueeze(1))
            total_loss += batch_loss.item()

            signal_mask = (batch.y == 1)  # Mask for signal nodes
            background_mask = (batch.y == 0)

            mean_sig_prob.append(preds[signal_mask].mean().item())
            mean_bkg_prob.append(preds[background_mask].mean().item())

            preds = (preds > thres).float().squeeze()

            # Signal-specific accuracy
            correct_signal += (preds[signal_mask] == batch.y[signal_mask].float()).sum().item()
            total_signal += signal_mask.sum().item()

            correct_bkg += (preds[background_mask] == batch.y[background_mask].float()).sum().item()
            total_bkg += background_mask.sum().item()

            #print(f"Predictions: {preds}")
            #print(f"True labels: {batch.y}")
            #print(f"Signal Mask: {signal_mask}")
            #print(f"Correct Signal: {correct_signal}, Total Signal: {total_signal}")
            #print(f"Correct total: {correct}, Total: {total}")

    bkg_accuracy = correct_bkg / total_bkg if total_bkg > 0 else 0
    sig_accuracy = correct_signal / total_signal if total_signal > 0 else 0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    auc = roc_auc_score(all_labels, all_preds)
    print(f"No sig hadrons test sample: {nosigtest}")
    #print("Mean signal probs per had", mean_sig_prob)
    #print("Mean background probs per had", mean_bkg_prob)

    return bkg_accuracy, sig_accuracy, avg_loss, auc


best_auc = 0

stats = {
    "epochs": [],
    "best_epoch": {
        "epoch": None,
        "total_loss": None,
        "node_loss": None,
        "cont_loss": None,
        "auc": None,
        "bkg_acc": None,
        "sig_acc": None,
        "test_loss": None,
        "total_acc": None
    }
}

for epoch in range(int(args.epochs)):
    tot_loss, node_loss, cont_loss = train(model, train_loader,  optimizer, device, epoch)
    bkg_acc, sig_acc, test_loss, auc = test(model, test_loader,  device, epoch, thres=glob_test_thres)
    sum_acc = bkg_acc + sig_acc

    epoch_stats = {
        "epoch": epoch + 1,
        "total_loss": tot_loss,
        "node_loss": node_loss,
        "cont_loss": cont_loss,
        "auc": auc,
        "bkg_acc": bkg_acc,
        "sig_acc": sig_acc,
        "test_loss": test_loss,
        "total_acc": sum_acc
    }
    stats["epochs"].append(epoch_stats)
    
    print(f"Epoch {epoch+1}/{args.epochs}, Total Loss: {tot_loss:.4f}, Node Loss: {node_loss:.4f}")
    print(f"AUC: {auc:.4f}, Bkg Acc: {bkg_acc*100:.4f}%, Sig Acc: {sig_acc*100:.4f}%, Test Loss: {test_loss:.4f}")


    if(epoch> 0 and epoch%10==0):
        savepath = os.path.join("model_files", f"model_{args.modeltag}_e{epoch}.pth")
        torch.save(model.state_dict(), savepath)
        print(f"Model saved to {savepath}")

    
    if auc > best_auc and epoch > 0: #and sig_prob > bkg_prob:
        
        stats["best_epoch"] = {
            "epoch": epoch + 1,
            "total_loss": tot_loss,
            "node_loss": node_loss,
            "cont_loss": cont_loss,
            "bkg_acc": bkg_acc,
            "auc": auc,
            "sig_acc": sig_acc,
            "test_loss": test_loss,
            "total_acc": sum_acc
        }

        save_path = os.path.join("model_files", f"model_{args.modeltag}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    if auc > best_auc: best_auc = auc

    with open("training_stats_"+args.modeltag+".json", "w") as f:
        json.dump(stats, f, indent=4)

