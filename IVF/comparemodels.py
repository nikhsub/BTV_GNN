import warnings
warnings.filterwarnings("ignore")


import argparse
import uproot
import numpy as np
import torch
import json
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import knn_graph
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F
import math
from GCNModel import *
import pprint
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import xgboost as xgb


parser = argparse.ArgumentParser("Model comparison")
parser.add_argument("-f", "--file", required=True, help="Testing data file")
parser.add_argument("-m1", "--model1", required=True, help="Path to first model (XGBoost or other)")
parser.add_argument("-m2", "--model2", required=True, help="Path to second model (XGBoost or other)")
parser.add_argument("-t1", "--tag1", required=True, help="Tag for model 1")
parser.add_argument("-t2", "--tag2", required=True, help="Tag for model 2")
parser.add_argument("-st", "--savetag", default="", help="Savetag for pngs")
parser.add_argument("-had", "--hadron", default=False, action='store_true', help="Testing on hadron samples")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']

def evaluate_xgb(graphs, model):

    all_preds, all_labels = [], []

    for data in graphs:
        preds = model.predict_proba(data.x)[:,1]  # Assuming data.x contains the features
    
        if(not args.hadron):
            siginds = data.siginds.cpu().numpy()
            labels = np.zeros(len(preds))
            labels[siginds] = 1
        else:
            labels = data.y.squeeze().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    return precision, recall, pr_auc, fpr, tpr, roc_auc, all_preds, all_labels

def evaluate(graphs, model, device):

    all_preds, all_labels = [], []
    sv_tp, sv_fp, sv_tn, sv_fn = 0, 0, 0, 0

    for data in graphs:
        with torch.no_grad():
            data = data.to(device)
            _, logits = model(data.x, data.edge_index, data.edge_attr)
            preds = torch.sigmoid(logits)
            preds = preds.squeeze().cpu().numpy()
                
            if(not args.hadron):
                siginds = data.siginds.cpu().numpy()
                svinds = data.svinds.cpu().numpy()
                ntrks = len(preds)

                labels = np.zeros(len(preds))
                labels[siginds] = 1
                
                tp = len(set(siginds) & set(svinds))
                tn = ntrks - len(set(siginds) | set(svinds))
                fp = len(set(svinds) - set(siginds))
                fn = len(set(siginds) - set(svinds))

                sv_tp += tp
                sv_fp += fp
                sv_tn += tn
                sv_fn += fn

 
            else:
                labels = data.y.squeeze().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    sv_tpr = sv_tp / (sv_tp + sv_fn) if (sv_tp + sv_fn) > 0 else 0
    sv_fpr = sv_fp / (sv_fp + sv_tn) if (sv_fp + sv_tn) > 0 else 0
    sv_precision = sv_tp / (sv_tp + sv_fp) if (sv_tp + sv_fp) > 0 else 0

    return precision, recall, pr_auc, sv_precision, sv_tpr, fpr, tpr, roc_auc, sv_tpr, sv_fpr, all_preds, all_labels #Recall is the same thing as tpr

#model1 = GNNModel(len(trk_features), 16, heads=8, dropout=0.11)  # Adjust input_dim if needed
model1 = GNNModel(len(trk_features), 96, edge_dim=len(edge_features))
model1.load_state_dict(torch.load(args.model1, map_location=torch.device('cpu')))
model1.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1.to(device)

total_params = sum(p.numel() for p in model1.parameters())
trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

with open(args.model2, "rb") as f:
    model2 = pickle.load(f)

print(f"Loading data from {args.file}...")
with open(args.file, 'rb') as f:
    graphs = pickle.load(f)

# Evaluate both files
print("Running GNN inference....")
p1, r1, auc1, sv_p1, sv_r1, fpr1, tpr1, roc_auc1, sv_tpr1, sv_fpr1, all_preds_GNN, all_labels_GNN = evaluate(graphs, model1, device)
print("Running xgb inference...")
p2, r2, auc2, fpr2, tpr2, roc_auc2, all_preds_XGB, all_labels_XGB = evaluate_xgb(graphs, model2)

# Plot the ROC curves
plt.figure(figsize=(10, 8))
plt.plot(r1, p1, label=f"{args.tag1} (AUC = {auc1:.2f})", color="red")
plt.plot(r2, p2, label=f"{args.tag2} (AUC = {auc2:.2f})", color="blue")
if(not args.hadron): plt.scatter([sv_r1], [sv_p1], color="black", label=f"IVF Recall={sv_r1:.2f}, Precision={sv_p1:.2f}", zorder=5)

plt.xlabel("Recall(Signal Efficiency)")
plt.ylabel("Precision")
plt.title('PR Curve')
#plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig(f"PR_modcompare_{args.savetag}.png")
plt.close()

plt.figure(figsize=(10, 8))
plt.plot(tpr1, fpr1, label=f"{args.tag1} (AUC = {roc_auc1:.2f})", color="red")
plt.plot(tpr2, fpr2, label=f"{args.tag2} (AUC = {roc_auc2:.2f})", color="blue")
if(not args.hadron): plt.scatter([sv_tpr1], [sv_fpr1], color="black", label=f"IVF TPR={sv_tpr1:.2f}, FPR={sv_fpr1:.2f}", zorder=5)

plt.xlabel("Signal Efficiency")
plt.ylabel("Background Mistag")
plt.title('ROC Curve')
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig(f"ROC_modcompare_{args.savetag}_log.png")
plt.close()

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else tensor

gnn_preds = to_numpy(all_preds_GNN)
gnn_labels = to_numpy(all_labels_GNN)

xgb_preds = to_numpy(all_preds_XGB)
xgb_labels = to_numpy(all_labels_XGB)

# Separate signal and background
gnn_sig = gnn_preds[gnn_labels == 1]
gnn_bkg = gnn_preds[gnn_labels == 0]
xgb_sig = xgb_preds[xgb_labels == 1]
xgb_bkg = xgb_preds[xgb_labels == 0]

# Plot side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

bins = 50

# GNN subplot
axs[0].hist(gnn_sig, bins=bins, alpha=0.8, label='Signal', color='red', density=False)
axs[0].hist(gnn_bkg, bins=bins, alpha=0.5, label='Background', color='blue', density=False)
axs[0].set_yscale('log')
axs[0].set_title("GNN Prediction Scores")
axs[0].set_xlabel("Prediction Score")
axs[0].set_ylabel("# of tracks")
axs[0].legend()
axs[0].grid(True)

# XGB subplot
axs[1].hist(xgb_sig, bins=bins, alpha=0.8, label='Signal', color='red', density=False)
axs[1].hist(xgb_bkg, bins=bins, alpha=0.5, label='Background', color='blue', density=False)
axs[1].set_yscale('log')
axs[1].set_title("XGBoost Prediction Scores")
axs[1].set_xlabel("Prediction Score")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig(f"output_dist_{args.savetag}.png", dpi=300)

