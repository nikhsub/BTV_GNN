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
from conmodel import *
import pprint
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


parser = argparse.ArgumentParser("GNN testing")
parser.add_argument("-f1", "--file1", required=True, help="First testing data file")
parser.add_argument("-f2", "--file2", required=True, help="Second testing data file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-st", "--savetag", default="", help="Savetag for pngs")
parser.add_argument("-had", "--hadron", default=False, action="store_true", help="Testing on hadronic samples")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

def evaluate(file, model, device):
    print(f"Loading data from {file}...")
    with open(file, 'rb') as f:
        graphs = pickle.load(f)

    all_preds, all_labels = [], []
    sv_tp, sv_fp, sv_tn, sv_fn = 0, 0, 0, 0

    for data in graphs:
        with torch.no_grad():
            data = data.to(device)
            edge_index = knn_graph(data.x, k=5, batch=None, loop=False).to(device)
            _, preds = model(data.x, edge_index)
            preds = preds.squeeze().cpu().numpy()

            siginds = data.siginds.cpu().numpy()
            svinds = data.svinds.cpu().numpy()
            ntrks = len(preds)

            labels = np.zeros(len(preds))
            labels[siginds] = 1

            all_preds.extend(preds)
            all_labels.extend(labels)

            tp = len(set(siginds) & set(svinds))
            tn = ntrks - len(set(siginds) | set(svinds))
            fp = len(set(svinds) - set(siginds))
            fn = len(set(siginds) - set(svinds))

            sv_tp += tp
            sv_fp += fp
            sv_tn += tn
            sv_fn += fn

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    sv_tpr = sv_tp / (sv_tp + sv_fn) if (sv_tp + sv_fn) > 0 else 0
    sv_fpr = sv_fp / (sv_fp + sv_tn) if (sv_fp + sv_tn) > 0 else 0

    return fpr, tpr, roc_auc, sv_tpr, sv_fpr

model = GNNModel(len(trk_features), 32, heads=8, dropout=0.3865)  # Adjust input_dim if needed
model.load_state_dict(torch.load(args.load_model))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate both files
fpr1, tpr1, auc1, sv_tpr1, sv_fpr1 = evaluate(args.file1, model, device)
fpr2, tpr2, auc2, sv_tpr2, sv_fpr2 = evaluate(args.file2, model, device)

# Plot the ROC curves
plt.figure(figsize=(10, 8))
plt.plot(tpr1, fpr1, label=f"All pt (AUC = {auc1:.2f})", color="red")
plt.plot(tpr2, fpr2, label=f"pt<20GeV (AUC = {auc2:.2f})", color="blue")
plt.scatter([sv_tpr1], [sv_fpr1], color="darkred", label=f"All pt IVF TPR={sv_tpr1:.2f}, FPR={sv_fpr1:.2f}", zorder=5)
plt.scatter([sv_tpr2], [sv_fpr2], color="darkblue", label=f"pt<20GeV IVF TPR={sv_tpr2:.2f}, FPR={sv_fpr2:.2f}", zorder=5)

plt.xlabel("Signal Accuracy")
plt.ylabel("Background Mistag")
if (args.hadron): plt.title('ROC - Hadronic Events')
elif (not args.hadron): plt.title('ROC - Leptonic Events')
plt.yscale("log")
plt.legend()
plt.grid()
if (args.hadron): plt.savefig(f"ROC_{args.savetag}_hadevts.png")
elif (not args.hadron): plt.savefig(f"ROC_{args.savetag}_lepevts.png")
plt.close()



                


        
            

