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
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, auc
import optuna
from functools import partial

def class_weighted_bce(preds, labels, pos_weight=5.0, neg_weight=1.0):
    """Class-weighted binary cross-entropy loss"""
    pos_weight = torch.tensor(pos_weight, device=preds.device)
    neg_weight = torch.tensor(neg_weight, device=preds.device)
    weights = torch.where(labels == 1, pos_weight, neg_weight)
    bce_loss = F.binary_cross_entropy(preds, labels, weight=weights)
    return bce_loss

def contrastive_loss(x1, x2, temperature=0.5):
    """Compute the contrastive loss"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    batch_size = x1.size(0)

    similarity_matrix = torch.mm(x1, x2.t()) / temperature

    labels = torch.arange(batch_size).to(x1.device)
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss

def shuffle_edge_index(edge_index):
    num_edges = edge_index.size(1)

    # Shuffle the target nodes (row 1) while keeping the sources (row 0) the same
    shuffled_targets = edge_index[1, torch.randperm(num_edges, device=edge_index.device)]

    # Combine the original sources with the shuffled targets
    new_edge_index = torch.stack([edge_index[0], shuffled_targets], dim=0)

    return new_edge_index

#TRAIN
def train(model, train_loader, optimizer, device, epoch, pos, neg, bce_loss=True):
    model.to(device)
    model.train()
    total_loss=0
    total_node_loss = 0
    total_cont_loss = 0

    for data in tqdm(train_loader, desc="Training", unit="Batch"):
        data = data.to(device)

        optimizer.zero_grad()
        num_nodes = data.x.size(0)
        batch_had_weight = 1
        
        node_embeds1, preds1 = model(data.x, data.edge_index)
        node_loss = class_weighted_bce(preds1, data.y.float().unsqueeze(1), pos_weight=pos, neg_weight=neg)*batch_had_weight

        if bce_loss:
            cont_loss = torch.tensor(0.0, device=device)
        else:
            edge_index2 = shuffle_edge_index(data.edge_index).to(device)
            node_embeds2, preds2 = model(data.x, edge_index2)
            cont_loss = contrastive_loss(node_embeds1, node_embeds2)

        #batch_had_weight = data.had_weight[data.batch].mean().to(device)

        loss = (cont_loss * 0.05) + node_loss

        loss.backward()
        optimizer.step()
        #scheduler.step()

        total_loss      += loss.item()
        total_node_loss += node_loss.item()
        total_cont_loss += cont_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_node_loss = total_node_loss / len(train_loader)
    avg_cont_loss = total_cont_loss / len(train_loader)

    #print(f"No seed hadrons: {nosigseeds}")
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
    #nosigtest = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="Batch"):
            batch = batch.to(device)
            num_nodes = batch.x.size(0)
            
            #batch.x = scale_features(batch.x, scale_mean, scale_std)
            
            edge_index = knn_graph(batch.x, k=k, batch=batch.batch, loop=False, cosine=False, flow="source_to_target").to(device)

            _, preds = model(batch.x, edge_index)

            all_preds.extend(preds.squeeze().cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            batch_loss = F.binary_cross_entropy(preds, batch.y.float().unsqueeze(1))
            total_loss += batch_loss.item()

            signal_mask = (batch.y == 1)  # Mask for signal nodes
            background_mask = (batch.y == 0)

            #mean_sig_prob.append(preds[signal_mask].mean().item())
            #mean_bkg_prob.append(preds[background_mask].mean().item())

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
    #print(f"No sig hadrons test sample: {nosigtest}")
    #print("Mean signal probs per had", mean_sig_prob)
    #print("Mean background probs per had", mean_bkg_prob)

    return bkg_accuracy, sig_accuracy, avg_loss, auc

def validate(model, val_graphs, device, epoch, k=5):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    num_samps = 0

    for i, data in enumerate(val_graphs):
        with torch.no_grad():
            data = data.to(device)
            edge_index = knn_graph(data.x, k=k, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
            _, preds = model(data.x, edge_index)
            preds = preds.squeeze()
            siginds = data.siginds.cpu().numpy()
            #labels = np.zeros(len(preds))
            labels = torch.zeros(len(preds), device=device)
            labels[siginds] = 1  # Set signal indices to 1
            evt_loss = F.binary_cross_entropy(preds, labels)
            total_loss += evt_loss.item()
            num_samps +=1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    avg_loss = total_loss / num_samps if num_samps > 0 else 0

    return roc_auc, avg_loss

def objective(trial, train_loader, val_loader, device):
    # Hyperparameter suggestions
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32])
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 10])
    dropout_rate = trial.suggest_uniform("dropout", 0.0, 0.5)
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    pos_weight = trial.suggest_uniform("pos_weight", 1.0, 10.0)
    neg_weight = trial.suggest_uniform("neg_weight", 1.0, 10.0)
    k = trial.suggest_int("k", 5.0, 12.0)

    # Initialize model, optimizer, and loss function
    model = GNNModel(indim=len(trk_features), outdim=hidden_dim, heads=num_heads, dropout=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(30):  # Early stopping after 20 epochs
        train_loss, _, _ = train(model, train_loader, optimizer, device, epoch, pos_weight, neg_weight)
        val_auc, _ = validate(model, val_loader, device, epoch, k=k)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5:  # Stop if no improvement for 5 epochs
            break

    return best_val_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser("GNN training")

    parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
    parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
    parser.add_argument("-lh", "--load_had", default="", help="Path to training files")
    parser.add_argument("-le", "--load_evt", default="", help="Absolute path to event level validation file")
    
    args = parser.parse_args()
    glob_test_thres = 0.5
    
    trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
    
    batchsize = 500
    
    #LOADING DATA
    train_hads = []
    val_evts = []
    if args.load_had != "":
        if os.path.isdir(args.load_had):
            print(f"Loading training data from {args.load_had}...")
            pkl_files = [os.path.join(args.load_had, f) for f in os.listdir(args.load_had) if f.endswith('.pkl')]
        for pkl_file in pkl_files:
            print(f"Loading {pkl_file}...")
            with open(pkl_file, 'rb') as f:
                train_hads.extend(pickle.load(f))
    
    if args.load_evt != "":
        print(f"Loading event level data from {args.load_evt}...")
        with open(args.load_evt, 'rb') as f:
            val_evts = pickle.load(f)
    
    train_hads = train_hads[:] #Control number of input samples here - see array splicing for more
    val_evts   = val_evts[0:1500]
    
    train_len = int(1 * len(train_hads))
    train_data, test_data = random_split(train_hads, [train_len, len(train_hads) - train_len])
    
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=8)
    #test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=8)
    
    #DEVICE AND MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    study = optuna.create_study(direction="maximize")  # Optimize validation AUC
    study.optimize(partial(objective, train_loader=train_loader, val_loader=val_evts, device=device),n_trials=50)
    
    print("Best hyperparameters:", study.best_params)
    print("Best validation AUC:", study.best_value)

    with open("best_params_"+args.modeltag+".json", "w") as f:
        json.dump({"best_params": study.best_params, "best_value": study.best_value}, f, indent=4)


    print(f"Best hyperparameters saved")
    
    model = GNNModel(indim=len(trk_features), outdim=study.best_params["hidden_dim"], heads=study.best_params["num_heads"], dropout=study.best_params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=study.best_params["lr"])

    best_val_auc = 0.0  # Track the best validation AUC
    best_model_path = "best_hyperopt_model_"+args.modeltag+".pth"  # Path to save the best model

    for epoch in range(int(args.epochs)):
        train_loss, _, _ = train(model, train_loader, optimizer, device, epoch, study.best_params["pos_weight"], study.best_params["neg_weight"])
        val_auc, _ = validate(model, val_evts, device, epoch, k=study.best_params["k"])
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val AUC = {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val AUC = {best_val_auc:.4f}")
