#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import torch
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from AttnModel import *

HF_CLASSES = {2, 3, 4}  # heavy flavour for IVF comparison

# ---------------------------------------------------------
# One-hot → integer labels
# ---------------------------------------------------------
def labels_from_onehot(y):
    return torch.argmax(y, dim=1).cpu().numpy()

# ---------------------------------------------------------
# Run GNN and get per-node class probabilities
# ---------------------------------------------------------
def run_gnn(graphs, model, device):
    all_probs = []
    all_labels = []

    for data in graphs:
        data = data.to(device)
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr)
            node_logits = out[0] if isinstance(out, (list, tuple)) else out
            if node_logits.dim() == 3:
                node_logits = node_logits.squeeze(0)
            probs = torch.softmax(node_logits, dim=1).cpu().numpy()

        labels = labels_from_onehot(data.y)
        all_probs.append(probs)
        all_labels.append(labels)

    return np.vstack(all_probs), np.hstack(all_labels)

# ---------------------------------------------------------
# Run XGB and get per-node class probabilities
# ---------------------------------------------------------
def run_xgb(graphs, xgb_model):
    all_probs = []
    all_labels = []

    for data in graphs:
        X = data.x.cpu().numpy()
        probs = xgb_model.predict_proba(X)  # (N,7)
        labels = labels_from_onehot(data.y)

        all_probs.append(probs)
        all_labels.append(labels)

    return np.vstack(all_probs), np.hstack(all_labels)

# ---------------------------------------------------------
# IVF HF-vs-rest ROC point
# ---------------------------------------------------------
def evaluate_ivf_hf(graphs):
    preds = []
    labels = []

    for data in graphs:
        n = data.y.shape[0]
        true_lbls = labels_from_onehot(data.y)
        y_bin = np.array([1 if c in HF_CLASSES else 0 for c in true_lbls])

        ivf_pred = np.zeros(n, dtype=np.int32)
        ivf_pred[data.ivf_inds.cpu().numpy()] = 1

        preds.append(ivf_pred)
        labels.append(y_bin)

    preds = np.hstack(preds)
    labels = np.hstack(labels)

    # IVF gives fixed prediction → one operating point
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # AUC not super meaningful for a single point, but you can compute it via roc_curve if needed
    return tpr, fpr

# ---------------------------------------------------------
# Plot OVR ROC for a single class
# ---------------------------------------------------------
def plot_ovr_roc_for_class(
    cls, gnn_probs, xgb_probs, labels, tag1, tag2,
    ivf_tpr, ivf_fpr, savetag
):
    class_name = label_names[cls]

    # Binary OVR labels
    y_true = (labels == cls).astype(int)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print(f"Class {cls} ({class_name}): trivial labels — skipping.")
        return

    # Scores
    gnn_scores = gnn_probs[:, cls]
    xgb_scores = xgb_probs[:, cls]

    # ROC curves
    fpr_gnn, tpr_gnn, _ = roc_curve(y_true, gnn_scores)
    auc_gnn = auc(fpr_gnn, tpr_gnn)

    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_scores)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    plt.figure(figsize=(8, 6))

    plt.plot(tpr_gnn, fpr_gnn,
             label=f"{tag1} (AUC={auc_gnn:.3f})",
             color='red', linewidth=2)

    plt.plot(tpr_xgb, fpr_xgb,
             label=f"{tag2} (AUC={auc_xgb:.3f})",
             color='blue', linewidth=2)

    # IVF only for HF-like classes
    if cls in {2, 3, 4}:
        plt.scatter(
            [ivf_tpr], [ivf_fpr],
            color='black', marker='x', s=80,
            label=f"IVF (TPR={ivf_tpr:.2f}, FPR={ivf_fpr:.2e})"
        )

    plt.xlabel("Signal Efficiency (TPR)")
    plt.ylabel("Background Mistag (FPR)")
    plt.yscale("log")
    plt.grid(True)

    plt.title(f"{class_name} — One-vs-Rest ROC")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ROC_ovr_{class_name}_{savetag}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# Plot OVR histograms for a single class (GNN vs XGB)
# ---------------------------------------------------------
def plot_ovr_hist_for_class(cls, gnn_probs, xgb_probs, labels, tag1, tag2, savetag):
    class_name = label_names[cls]

    y_true = (labels == cls).astype(int)

    gnn_scores = gnn_probs[:, cls]
    xgb_scores = xgb_probs[:, cls]

    gnn_sig = gnn_scores[y_true == 1]
    gnn_bkg = gnn_scores[y_true == 0]

    xgb_sig = xgb_scores[y_true == 1]
    xgb_bkg = xgb_scores[y_true == 0]

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    bins = 50

    # --------------------------
    # GNN
    # --------------------------
    axs[0].hist(gnn_sig, bins=bins, alpha=0.75, label="Signal", color='red')
    axs[0].hist(gnn_bkg, bins=bins, alpha=0.7, label="Background", color='blue')

    axs[0].set_yscale("log")
    axs[0].set_title(f"GNN ({tag1}) — {class_name} OVR")
    axs[0].set_xlabel("Score")
    axs[0].set_ylabel("Tracks")
    axs[0].grid(True)
    axs[0].legend()

    # --------------------------
    # XGB
    # --------------------------
    axs[1].hist(xgb_sig, bins=bins, alpha=0.75, label="Signal", color='red')
    axs[1].hist(xgb_bkg, bins=bins, alpha=0.7, label="Background", color='blue')

    axs[1].set_yscale("log")
    axs[1].set_title(f"XGB ({tag2}) — {class_name} OVR")
    axs[1].set_xlabel("Score")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"HIST_ovr_{class_name}_{savetag}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
parser = argparse.ArgumentParser("Model comparison: OVR ROC + histograms")
parser.add_argument("-f",  "--file",   required=True, help="Test .pt file with graphs")
parser.add_argument("-m1", "--model1", required=True, help="GNN model path (.pth)")
parser.add_argument("-m2", "--model2", required=True, help="XGB model path (.pkl)")
parser.add_argument("-t1", "--tag1",   required=True, help="Name/tag for GNN")
parser.add_argument("-t2", "--tag2",   required=True, help="Name/tag for XGB")
parser.add_argument("-st", "--savetag", default="",   help="Suffix for output PNGs")
args = parser.parse_args()

print("Loading graphs...")
graphs = torch.load(args.file)

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_dz', 'trk_dzsig','trk_ip2dsig', 'trk_ip3dsig',  'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']
label_names = ["Hard", "PU", "B", "BtoC", "C", "Oth", "Fake"]

# --- Load GNN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gnn = TrackEdgeGNN(node_in_dim=len(trk_features), edge_in_dim=len(edge_features), hidden_dim=64, heads=2)
gnn.load_state_dict(torch.load(args.model1, map_location="cpu"))
gnn.to(device)
gnn.eval()

print("Running GNN inference...")
gnn_probs, gnn_labels = run_gnn(graphs, gnn, device)

# --- Load XGB ---
print("Loading XGB model...")
with open(args.model2, "rb") as f:
    xgb_model = pickle.load(f)

print("Running XGB inference...")
xgb_probs, xgb_labels = run_xgb(graphs, xgb_model)

# Sanity: labels from GNN and XGB should match; we can trust gnn_labels
labels = gnn_labels

# --- IVF HF-vs-rest ROC point (shared across cls 2,3,4 plots) ---
print("Evaluating IVF HF-vs-rest point...")
ivf_tpr, ivf_fpr = evaluate_ivf_hf(graphs)
print(f"IVF HF ROC point: TPR={ivf_tpr:.3f}, FPR={ivf_fpr:.3f}")

# --- For each class: OVR ROC + OVR histograms ---
for cls in range(7):
    print(f"Processing class {cls}...")
    plot_ovr_roc_for_class(
        cls, gnn_probs, xgb_probs, labels,
        args.tag1, args.tag2,
        ivf_tpr, ivf_fpr,
        args.savetag
    )
    plot_ovr_hist_for_class(
        cls, gnn_probs, xgb_probs, labels,
        args.tag1, args.tag2,
        args.savetag
    )

print("\nDone. Generated OVR ROC and OVR histograms for all 7 classes.\n")

