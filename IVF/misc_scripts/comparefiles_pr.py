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
parser.add_argument("-t1", "--tag1", required=True, help="Tag for file 1")
parser.add_argument("-t2", "--tag2", required=True, help="Tag for file 2")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-st", "--savetag", default="", help="Savetag for pngs")

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
            edge_index = knn_graph(data.x, k=6, batch=None, loop=False).to(device)
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

    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)

    sv_tpr = sv_tp / (sv_tp + sv_fn) if (sv_tp + sv_fn) > 0 else 0
    sv_precision = sv_tp / (sv_tp + sv_fp) if (sv_tp + sv_fp) > 0 else 0

    return precision, recall, pr_auc, sv_precision, sv_tpr #Recall is the same thing as tpr

model = GNNModel(len(trk_features), 16, heads=4, dropout=0.354)  # Adjust input_dim if needed
model.load_state_dict(torch.load(args.load_model))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Evaluate both files
p1, r1, auc1, sv_p1, sv_r1 = evaluate(args.file1, model, device)
p2, r2, auc2, sv_p2, sv_r2 = evaluate(args.file2, model, device)

# Plot the ROC curves
plt.figure(figsize=(10, 8))
plt.plot(r1, p1, label=f"{args.tag1} (AUC = {auc1:.2f})", color="red")
plt.plot(r2, p2, label=f"{args.tag2} (AUC = {auc2:.2f})", color="blue")
plt.scatter([sv_r1], [sv_p1], color="darkred", label=f"{args.tag1} IVF Recall={sv_r1:.2f}, Precision={sv_p1:.2f}", zorder=5)
plt.scatter([sv_r2], [sv_p2], color="darkblue", label=f"{args.tag2} IVF Recall={sv_r2:.2f}, Precision={sv_p2:.2f}", zorder=5)

plt.xlabel("Recall(Signal Efficiency)")
plt.ylabel("Precision")
plt.title('PR Curve')
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig(f"PR_{args.savetag}_log.png")
plt.close()



                


        
            

