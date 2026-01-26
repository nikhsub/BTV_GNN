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
from GCNModel import *

HF_CLASSES = {2, 3, 4}  # heavy flavour for IVF comparison

C_CLASS    = 4
LIGHT_CLASSES = {0, 1, 5, 6}  # Hard, PU, Oth, Fake (your "light-like" classes)

# ---------------------------------------------------------
# One-hot → integer labels
# ---------------------------------------------------------
def labels_from_onehot(y):
    return torch.argmax(y, dim=1).cpu().numpy()

# ---------------------------------------------------------
# Run GNN and get per-node class probabilities
# ---------------------------------------------------------
def run_gnn(graphs, model, device):
    all_node_probs = []
    all_node_labels = []

    all_edge_scores = []   # prob(edge = 1)
    all_edge_labels = []   # binary edge_y

    for data in graphs:
        data = data.to(device)

        with torch.no_grad():
            # Model returns exactly: node_logits, edge_logits, node_probs, edge_probs
            node_logits, edge_logits, node_probs, edge_probs = model(
                data.x, data.edge_index, data.edge_attr
            )

        # --- nodes ---
        # node_probs: [N, num_classes]
        node_probs_np = node_probs.cpu().numpy()
        node_labels_np = labels_from_onehot(data.y)   # data.y is one-hot

        all_node_probs.append(node_probs_np)
        all_node_labels.append(node_labels_np)

        # --- edges ---
        # edge_probs: [E, 1] → flatten to [E]
        if edge_probs is not None and hasattr(data, "edge_y"):
            edge_scores = edge_probs.view(-1).cpu().numpy()      # P(edge=1)
            edge_labels = data.edge_y.view(-1).cpu().numpy()     # 0/1

            all_edge_scores.append(edge_scores)
            all_edge_labels.append(edge_labels)

    node_probs_all  = np.vstack(all_node_probs)
    node_labels_all = np.hstack(all_node_labels)

    if all_edge_scores:
        edge_scores_all = np.hstack(all_edge_scores)
        edge_labels_all = np.hstack(all_edge_labels)
    else:
        edge_scores_all = None
        edge_labels_all = None

    # return both node- and edge-level info
    return node_probs_all, node_labels_all, edge_scores_all, edge_labels_all


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

    plt.title(f"{class_name} — One-vs-Rest")

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
    axs[0].hist(gnn_sig, bins=bins, alpha=0.8, label="Signal", color='red')
    axs[0].hist(gnn_bkg, bins=bins, alpha=0.5, label="Background", color='blue')

    axs[0].set_yscale("log")
    axs[0].set_title(f"GNN — {class_name} OVR")
    axs[0].set_xlabel("Score")
    axs[0].set_ylabel("Tracks")
    axs[0].grid(True)
    axs[0].legend()

    # --------------------------
    # XGB
    # --------------------------
    axs[1].hist(xgb_sig, bins=bins, alpha=0.8, label="Signal", color='red')
    axs[1].hist(xgb_bkg, bins=bins, alpha=0.5, label="Background", color='blue')

    axs[1].set_yscale("log")
    axs[1].set_title(f"XGB — {class_name} OVR")
    axs[1].set_xlabel("Score")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(f"HIST_ovr_{class_name}_{savetag}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# Combined HF (2,3,4) vs rest ROC (with IVF point)
# ---------------------------------------------------------
def plot_combined_hf_roc(
    gnn_probs, xgb_probs, labels, tag1, tag2,
    ivf_tpr, ivf_fpr, savetag
):
    # Binary label: 1 = HF (2,3,4), 0 = others
    y_true = np.isin(labels, list(HF_CLASSES)).astype(int)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("Combined HF vs rest: trivial labels — skipping.")
        return

    # Scores: sum probabilities over HF classes
    hf_idx = sorted(HF_CLASSES)
    gnn_scores = gnn_probs[:, hf_idx].sum(axis=1)
    xgb_scores = xgb_probs[:, hf_idx].sum(axis=1)

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

    # IVF HF-vs-rest operating point
    plt.scatter(
        [ivf_tpr], [ivf_fpr],
        color='black', marker='x', s=80,
        label=f"IVF (TPR={ivf_tpr:.2f}, FPR={ivf_fpr:.2e})"
    )

    plt.xlabel("Signal Efficiency (TPR)")
    plt.ylabel("Background Mistag (FPR)")
    plt.yscale("log")
    plt.grid(True)

    plt.title("Combined HF (B, B→C, C) vs Rest")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ROC_HF_combined_{savetag}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# Pairwise HF-vs-HF ROC (e.g., 2 vs 3, 2 vs 4, 3 vs 4)
# ---------------------------------------------------------
def plot_pairwise_hf_roc(
    cls_pos, cls_neg, gnn_probs, xgb_probs, labels,
    tag1, tag2, savetag
):
    name_pos = label_names[cls_pos]
    name_neg = label_names[cls_neg]

    # Restrict to samples that are either cls_pos or cls_neg
    mask = np.isin(labels, [cls_pos, cls_neg])
    if mask.sum() == 0:
        print(f"Pairwise {cls_pos} vs {cls_neg}: no samples — skipping.")
        return

    labels_pair = labels[mask]
    # Binary: 1 = cls_pos, 0 = cls_neg
    y_true = (labels_pair == cls_pos).astype(int)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print(f"Pairwise {cls_pos} vs {cls_neg}: trivial labels — skipping.")
        return

    gnn_scores = gnn_probs[mask, cls_pos]
    xgb_scores = xgb_probs[mask, cls_pos]

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

    plt.xlabel(f"Signal Efficiency (TPR) — {name_pos}")
    plt.ylabel(f"Mistag Rate (FPR) — {name_neg}")
    plt.yscale("log")
    plt.grid(True)

    plt.title(f"{name_pos} vs {name_neg}")

    plt.legend()
    plt.tight_layout()
    fname = f"ROC_pair_{name_pos}_vs_{name_neg}_{savetag}.png"
    # Replace spaces just in case
    fname = fname.replace(" ", "")
    plt.savefig(fname, dpi=300)
    plt.close()

# ---------------------------------------------------------
# C vs light ROC (CvsLight)
# ---------------------------------------------------------
def plot_c_vs_light_roc(
    gnn_probs, xgb_probs, labels,
    tag1, tag2, savetag,
    light_classes=LIGHT_CLASSES, c_class=C_CLASS
):
    # Keep only C or light labels
    valid_labels = list(light_classes) + [c_class]
    mask = np.isin(labels, valid_labels)
    if mask.sum() == 0:
        print("CvsLight: no C/light samples — skipping.")
        return

    labels_sub = labels[mask]
    y_true = (labels_sub == c_class).astype(int)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("CvsLight: trivial labels — skipping.")
        return

    # GNN scores: pC / (pC + p_light)
    gnn_sub = gnn_probs[mask]
    pC_gnn = gnn_sub[:, c_class]
    pL_gnn = gnn_sub[:, list(light_classes)].sum(axis=1)
    score_gnn = pC_gnn / (pC_gnn + pL_gnn + 1e-9)

    fpr_gnn, tpr_gnn, _ = roc_curve(y_true, score_gnn)
    auc_gnn = auc(fpr_gnn, tpr_gnn)

    # XGB scores
    xgb_sub = xgb_probs[mask]
    pC_xgb = xgb_sub[:, c_class]
    pL_xgb = xgb_sub[:, list(light_classes)].sum(axis=1)
    score_xgb = pC_xgb / (pC_xgb + pL_xgb + 1e-9)

    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, score_xgb)
    auc_xgb = auc(fpr_xgb, tpr_xgb)

    plt.figure(figsize=(8, 6))

    plt.plot(tpr_gnn, fpr_gnn,
             label=f"{tag1} (AUC={auc_gnn:.3f})",
             color='red', linewidth=2)

    plt.plot(tpr_xgb, fpr_xgb,
             label=f"{tag2} (AUC={auc_xgb:.3f})",
             color='blue', linewidth=2)

    plt.xlabel("Signal Efficiency (TPR) — C")
    plt.ylabel("Mistag Rate (FPR) — light")
    plt.yscale("log")
    plt.grid(True)
    plt.title("CvsLight")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ROC_CvsLight_{savetag}.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# Edge ROC (binary) for GNN only
# ---------------------------------------------------------
def plot_edge_roc(edge_scores, edge_labels, tag1, savetag):
    """
    edge_scores: 1D array of probabilities for edge=1
    edge_labels: 1D array of 0/1 labels
    """
    if edge_scores is None or edge_labels is None:
        print("No edge scores/labels provided — skipping edge ROC.")
        return

    y_true = edge_labels.astype(int)
    # need both classes present
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        print("Edge ROC: trivial labels — skipping.")
        return

    fpr, tpr, _ = roc_curve(y_true, edge_scores)
    auc_val = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(tpr, fpr, label=f"{tag1} (AUC={auc_val:.3f})", linewidth=2)

    plt.xlabel("Signal Efficiency (TPR) — true edges")
    plt.ylabel("Mistag Rate (FPR) — fake edges")
    plt.yscale("log")
    plt.grid(True)
    plt.title("Edge Classification — ROC")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"EDGE_ROC_{savetag}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------
# Edge score histograms: true vs fake edges
# ---------------------------------------------------------
def plot_edge_hist(edge_scores, edge_labels, tag1, savetag):
    if edge_scores is None or edge_labels is None:
        print("No edge scores/labels provided — skipping edge hist.")
        return

    y_true = edge_labels.astype(int)
    scores_sig = edge_scores[y_true == 1]
    scores_bkg = edge_scores[y_true == 0]

    if len(scores_sig) == 0 or len(scores_bkg) == 0:
        print("Edge hist: need both positive and negative edges — skipping.")
        return

    plt.figure(figsize=(8, 6))
    bins = 50

    plt.hist(scores_sig, bins=bins, alpha=0.8, label="True edges", color='red', density=False)
    plt.hist(scores_bkg, bins=bins, alpha=0.5, label="Fake edges", color='blue', density=False)

    plt.yscale("log")
    plt.xlabel("Edge score")
    plt.ylabel("Edges")
    plt.grid(True)
    plt.title(f"Edge Scores — {tag1}")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"EDGE_HIST_{savetag}.png", dpi=300)
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
#gnn = TrackEdgeGNN(node_in_dim=len(trk_features), edge_in_dim=len(edge_features), hidden_dim=64)
gnn = GNNModel(indim=len(trk_features), outdim=128, edge_dim=len(edge_features))
gnn.load_state_dict(torch.load(args.model1, map_location="cpu"))
gnn.to(device)
gnn.eval()

total_params = sum(p.numel() for p in gnn.parameters())
trainable_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("Running GNN inference...")
gnn_probs, gnn_labels, gnn_edge_scores, gnn_edge_labels = run_gnn(graphs, gnn, device)

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

# --- Combined HF (2,3,4) vs rest ROC with IVF point ---
print("Processing combined HF (2,3,4) vs rest ROC...")
plot_combined_hf_roc(
    gnn_probs, xgb_probs, labels,
    args.tag1, args.tag2,
    ivf_tpr, ivf_fpr,
    args.savetag
)

# --- Pairwise HF-vs-HF ROC: 2 vs 3, 2 vs 4, 3 vs 4 ---
hf_list = sorted(HF_CLASSES)
print("Processing pairwise HF-vs-HF ROC curves...")
for i in range(len(hf_list)):
    for j in range(i + 1, len(hf_list)):
        cls_pos = hf_list[i]
        cls_neg = hf_list[j]
        print(f"  Pairwise ROC: {cls_pos} ({label_names[cls_pos]}) vs {cls_neg} ({label_names[cls_neg]})")
        plot_pairwise_hf_roc(
            cls_pos, cls_neg,
            gnn_probs, xgb_probs, labels,
            args.tag1, args.tag2,
            args.savetag
        )

print("Processing CvsLight, CvsB, CvsBtoC ROC curves...")
plot_c_vs_light_roc(
    gnn_probs, xgb_probs, labels,
    args.tag1, args.tag2,
    args.savetag
)

# C vs B (C as signal)
plot_pairwise_hf_roc(
    C_CLASS, 2,
    gnn_probs, xgb_probs, labels,
    args.tag1, args.tag2,
    args.savetag
)

# C vs BtoC (C as signal)
plot_pairwise_hf_roc(
    C_CLASS, 3,
    gnn_probs, xgb_probs, labels,
    args.tag1, args.tag2,
    args.savetag
)

# --- Edge-level ROC + histograms (GNN only) ---
print("Processing edge-level ROC and histograms...")
plot_edge_roc(
    gnn_edge_scores, gnn_edge_labels,
    args.tag1, args.savetag
)
plot_edge_hist(
    gnn_edge_scores, gnn_edge_labels,
    args.tag1, args.savetag
)

print("\nDone.\n")

