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
from sklearn.calibration import calibration_curve
import xgboost as xgb

parser = argparse.ArgumentParser("Model comparison with diagnostics")
parser.add_argument("-f", "--file", required=True, help="Testing data file")
parser.add_argument("-m1", "--model1", required=True, help="Path to first model (GNN)")
parser.add_argument("-m2", "--model2", required=True, help="Path to second model (XGB)")
parser.add_argument("-t1", "--tag1", required=True, help="Tag for model 1")
parser.add_argument("-t2", "--tag2", required=True, help="Tag for model 2")
parser.add_argument("-st", "--savetag", default="", help="Savetag for pngs")
parser.add_argument("-had", "--hadron", default=False, action='store_true', help="Testing on hadron samples")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig',
                'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2',
                 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']


def evaluate(graphs, model, device):
    all_preds, all_labels = [], []
    all_logits, all_feats = [], []  ### DIAGNOSTIC
    sv_tp, sv_fp, sv_tn, sv_fn = 0, 0, 0, 0

    for data in graphs:
        with torch.no_grad():
            data = data.to(device)
            _, logits = model(data.x.unsqueeze(0), data.edge_index.unsqueeze(0), data.edge_attr.unsqueeze(0))

            preds = torch.sigmoid(logits).squeeze().cpu().numpy()
            logits_np = logits.squeeze().cpu().numpy()

            if not args.hadron:
                siginds = data.siginds.cpu().numpy()
                svinds = data.svinds.cpu().numpy()
                ntrks = len(preds)
                labels = np.zeros(ntrks, dtype=np.int8)
                labels[siginds] = 1

                tp = len(set(siginds) & set(svinds))
                tn = ntrks - len(set(siginds) | set(svinds))
                fp = len(set(svinds) - set(siginds))
                fn = len(set(siginds) - set(svinds))
                sv_tp += tp; sv_fp += fp; sv_tn += tn; sv_fn += fn
            else:
                labels = data.y.squeeze().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_logits.extend(logits_np)                ### DIAGNOSTIC
            all_feats.append(data.x.cpu().numpy())      ### DIAGNOSTIC

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)                  ### DIAGNOSTIC
    all_feats = np.concatenate(all_feats, axis=0)      ### DIAGNOSTIC

    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    sv_tpr = sv_tp / (sv_tp + sv_fn) if (sv_tp + sv_fn) > 0 else 0
    sv_fpr = sv_fp / (sv_fp + sv_tn) if (sv_fp + sv_tn) > 0 else 0
    sv_precision = sv_tp / (sv_tp + sv_fp) if (sv_tp + sv_fp) > 0 else 0

    return (precision, recall, pr_auc, sv_precision, sv_tpr,
            fpr, tpr, roc_auc, sv_tpr, sv_fpr,
            all_preds, all_labels, all_logits, all_feats)


# ---------------------------
# Load models
# ---------------------------
model1 = GNNModel(len(trk_features), 64, edge_dim=len(edge_features))
model1.load_state_dict(torch.load(args.model1, map_location=torch.device('cpu')))
model1.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1.to(device)

with open(args.model2, "rb") as f:
    model2 = pickle.load(f)

print(f"Loading data from {args.file}...")
with open(args.file, 'rb') as f:
    graphs = pickle.load(f)

# ---------------------------
# Run evaluation
# ---------------------------
print("Running GNN inference....")
(p1, r1, auc1, sv_p1, sv_r1,
 fpr1, tpr1, roc_auc1, sv_tpr1, sv_fpr1,
 all_preds_GNN, all_labels_GNN, all_logits_GNN, all_feats_GNN) = evaluate(graphs, model1, device)

print("Running XGB inference...")
def evaluate_xgb(graphs, model):
    all_preds, all_labels = [], []
    for data in graphs:
        preds = model.predict_proba(data.x)[:,1]
        preds = np.nan_to_num(preds, nan=0.0)
        labels = data.y.squeeze().cpu().numpy() if args.hadron else np.zeros(len(preds))
        if not args.hadron:
            siginds = data.siginds.cpu().numpy()
            labels[siginds] = 1
        all_preds.extend(preds); all_labels.extend(labels)
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    return precision, recall, pr_auc, fpr, tpr, roc_auc, all_preds, all_labels

(p2, r2, auc2, fpr2, tpr2, roc_auc2,
 all_preds_XGB, all_labels_XGB) = evaluate_xgb(graphs, model2)

# ---------------------------
# Diagnostics
# ---------------------------

# 1. Logits histogram for background
bkg_mask = (all_labels_GNN == 0)
plt.figure(figsize=(8,6))
plt.hist(all_logits_GNN[bkg_mask], bins=80, histtype="step", color="blue")
plt.axvline(x=-2.2, linestyle="--", color="red", label="sigmoid≈0.1")
plt.xlabel("Raw GNN logit (pre-sigmoid)"); plt.ylabel("# of tracks"); plt.yscale("log")
plt.title("Background logits (GNN)")
plt.legend(); plt.grid(True)
plt.savefig(f"logits_bkg_{args.savetag}.png", dpi=200)
plt.close()

# 2. In-peak vs off-peak feature summary
lo, hi = 0.08, 0.12
in_peak  = bkg_mask & (all_preds_GNN >= lo) & (all_preds_GNN < hi)
off_peak = bkg_mask & ~in_peak

summary = []
for i, name in enumerate(trk_features):
    a = all_feats_GNN[in_peak, i]; b = all_feats_GNN[off_peak, i]
    summary.append({
        "feat": name,
        "mean_peak": float(np.nanmean(a)) if a.size else np.nan,
        "mean_off":  float(np.nanmean(b)) if b.size else np.nan
    })
with open(f"peak_vs_off_peak_{args.savetag}.json", "w") as f:
    json.dump(summary, f, indent=2)

# 3. Reliability curve
prob_true, prob_pred = calibration_curve(all_labels_GNN, all_preds_GNN, n_bins=20, strategy="quantile")
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker="o", label="GNN")
plt.plot([0,1],[0,1],"--", alpha=0.6)
plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
plt.title("Calibration / Reliability curve")
plt.grid(True); plt.legend()
plt.savefig(f"reliability_{args.savetag}.png", dpi=200)
plt.close()

obs_rate = all_labels_GNN[in_peak].mean() if in_peak.any() else float("nan")
print(f"Observed positive rate in [{lo},{hi}): {obs_rate:.4f}")

print("Diagnostics complete. Plots + JSON saved with savetag.")

