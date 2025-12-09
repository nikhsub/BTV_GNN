import warnings
warnings.filterwarnings("ignore")

import json
import argparse
import uproot
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
#import torch5 as torch
#import torch
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, label_binarize
import os
import subprocess
import torch.nn.functional as F
import math
#from AttnModel import *
from GCNModel import *
import joblib
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
import random

parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-le", "--load_evt", default="", help="Absolute path to event level validation file")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_dz', 'trk_dzsig','trk_ip2dsig', 'trk_ip3dsig',  'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']

batchsize = 1000

def count_node_classes(graphs):
    counts = np.zeros(7, dtype=np.int64)
    for g in graphs:
        y = g.y.cpu().numpy()          # [N,7] one-hot
        counts += y.sum(axis=0).astype(np.int64)
    print("Class counts:", counts, "total:", counts.sum())

def compute_ce_node_weight(graphs, device):
    """
    Computes class weights for CrossEntropyLoss from one-hot labels.
    Intended for exclusive 7-class classification.
    """
    class_counts = np.zeros(7, dtype=np.int64)

    for g in graphs:
        y = g.y.cpu().numpy()  # [N,7]
        class_counts += y.sum(axis=0).astype(np.int64)
    # Total number of nodes
    total = class_counts.sum()
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    # inverse frequency weighting
    inv_freq = total / class_counts  # bigger weight = rarer class
    # normalize (optional but recommended)
    inv_freq = inv_freq / inv_freq.mean()

    return torch.tensor(inv_freq, dtype=torch.float32, device=device)

def compute_edge_pos_weight(graphs, device):
    total_pos = 0
    total_neg = 0
    for g in graphs:
        if g.edge_y.numel() == 0:
            continue
        y = g.edge_y.cpu().numpy()  # shape [E, 1]
        total_pos += (y == 1).sum()
        total_neg += (y == 0).sum()

    if total_pos == 0:
        return torch.tensor([1.0], device=device)

    weight = total_neg / total_pos
    return torch.tensor([weight], device=device)


EOS_PREFIX = "root://cmseos.fnal.gov/"
EOS_DIR = "/store/user/nvenkata/BTV/proc_fortrain_ttbarhad_1311/test"
local_cache = "/uscms/home/nvenkata/nobackup/BTV/IVF/tmp_pt"
os.makedirs(local_cache, exist_ok=True)
train_hads = []
# Get the list of .pt files from the EOS area
result = subprocess.run(
    ["xrdfs", EOS_PREFIX.replace("root://", "").rstrip("/"), "ls", "-u", EOS_DIR],
    stdout=subprocess.PIPE, text=True, check=True
)
pt_files = [line.strip() for line in result.stdout.splitlines() if line.endswith(".pt")]
pt_files.sort()

print(f"Found {len(pt_files)} .pt files in {EOS_DIR}")

# Load all torch files (copy -> load -> delete)
for remote_file in tqdm(pt_files, desc="Loading EOS .pt files", unit="file"):
    local_path = os.path.join(local_cache, os.path.basename(remote_file))

    # Copy from EOS to local temp
    subprocess.run(
        ["xrdcp", "-f", remote_file, local_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Load torch
    train_hads.extend(torch.load(local_path))

    # Delete local file after loading to save space
    os.remove(local_path)

print(f"Loaded {len(train_hads)} records from {len(pt_files)} torch files.")

#train_hads = train_hads[:] #Control number of input samples here - see array splicing for more
#val_evts   = val_evts[0:1500]

train_len = int(0.9 * len(train_hads))
train_data, test_data = random_split(train_hads, [train_len, len(train_hads) - train_len])

train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=0)

val_loader = []
if args.load_evt != "":
    print(f"Loading event level data from {args.load_evt}...")
    val_loader = torch.load(args.load_evt)

# device, model, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#count_node_classes(train_hads)
pos_w_node = compute_ce_node_weight(train_hads, device)
pos_w_edge = compute_edge_pos_weight(train_hads, device)
print(f"Node pos weight: {pos_w_node}")
print(f"Edge pos weight: {pos_w_edge.item():.2f}")
#model = TrackEdgeGNN(node_in_dim=len(trk_features), edge_in_dim=len(edge_features), hidden_dim=64).to(device)
model = GNNModel(indim=len(trk_features), outdim=64, edge_dim=len(edge_features))
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def focal_loss(preds, labels, gamma=2.7, alpha=0.9):
    """Focal loss to emphasize hard-to-classify samples"""
    gamma = float(gamma)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = loss_fn(preds, labels)
    pt = torch.exp(-bce_loss)  # Probability of correct classification
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * bce_loss
    return loss.mean()

#TRAIN
def train(model, train_loader, optimizer, device,
          node_loss_weight=1.0, edge_loss_weight=0.5):

    model.to(device)
    model.train()

    total_loss = 0.0
    total_node_loss = 0.0
    total_edge_loss = 0.0
    batch_count = 0

    # -------------------------------
    # Loss functions
    # -------------------------------
    ce_loss_fn  = torch.nn.CrossEntropyLoss(weight=pos_w_node)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w_edge)

    for batch in tqdm(train_loader, desc="Training", unit="Batch"):
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        node_logits, edge_logits, *_ = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr
        )

        # Convert one-hot → integer labels
        node_labels = batch.y.argmax(dim=1).long()

        # Node loss (class-weighted CE)
        node_loss = ce_loss_fn(node_logits, node_labels)

        # Edge loss (binary)
        if batch.edge_index.size(1) > 0:
            edge_loss = bce_loss_fn(edge_logits.view(-1), batch.edge_y.float())
        else:
            edge_loss = torch.tensor(0.0, device=device)

        # Combined weighted loss
        loss = node_loss_weight * node_loss + edge_loss_weight * edge_loss

        # Backprop
        loss.backward()
        optimizer.step()

        num_nodes = batch.x.size(0)
        num_edges = batch.edge_index.size(1)
        denom = max(1, num_nodes + num_edges)

        total_loss_batch = (
            node_loss.item() * num_nodes +
            edge_loss_weight * edge_loss.item() * num_edges
        ) / denom


        # Stats
        total_loss      += total_loss_batch
        total_node_loss += node_loss.item()
        total_edge_loss += edge_loss.item()
        batch_count += 1

    # Averages
    return (
        total_loss      / batch_count,
        total_node_loss / batch_count,
        total_edge_loss / batch_count
    )


#TEST
def test(model, test_loader, device, thres=0.5):
    model.to(device)
    model.eval()
    
    ce_loss_fn  = torch.nn.CrossEntropyLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    total_node_loss = 0.0
    total_edge_loss = 0.0
    total_batches = 0

    # Storage for metrics
    all_node_probs = []
    all_node_labels = []

    all_edge_probs = []
    all_edge_labels = []


    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="Batch"):
            batch = batch.to(device)

            node_logits, edge_logits, node_probs, edge_probs = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr
            )

            node_labels = batch.y.argmax(dim=1).long()

            node_logits = node_logits.squeeze(0)      # [N,7]
            node_probs  = node_probs.squeeze(0)       # [N,7]
            edge_logits = edge_logits.squeeze(0)      # [E,1]
            edge_probs  = edge_probs.squeeze(0)       # [E,1]
            
            node_loss = ce_loss_fn(node_logits, node_labels)
            edge_loss = bce_loss_fn(edge_logits.view(-1), batch.edge_y.float())

            total_node_loss += node_loss.item()
            total_edge_loss += edge_loss.item()
            total_batches   += 1

            # Store full softmax output (needed for OvR AUC)
            all_node_probs.append(node_probs.cpu())      # [N,7]
            all_node_labels.append(batch.y.cpu())        # [N,7]

            # Store edge results
            all_edge_probs.append(edge_probs.cpu().view(-1))
            all_edge_labels.append(batch.edge_y.cpu().view(-1))

    all_node_probs  = torch.cat(all_node_probs, dim=0).numpy()   # [TotalN,7]
    all_node_labels = torch.cat(all_node_labels, dim=0).numpy()  # [TotalN,7]

    all_edge_probs  = torch.cat(all_edge_probs, dim=0).numpy()
    all_edge_labels = torch.cat(all_edge_labels, dim=0).numpy()

    # =============================
    # Metrics: Node AUC (OvR)
    # =============================
    true_class = all_node_labels.argmax(axis=1)  # integer labels

    num_classes = all_node_probs.shape[1]
    
    y_true_onehot = label_binarize(
        true_class,
        classes=np.arange(num_classes)
    )

    node_auc = roc_auc_score(
        y_true_onehot,
        all_node_probs,          # full 7-dim probabilities
        multi_class="ovr",
        average=None             # returns vector of per-class AUC
    )

    # =============================
    # Metrics: Node accuracy
    # =============================
    pred_class = all_node_probs.argmax(axis=1)
    node_accuracy = accuracy_score(true_class, pred_class)

    # =============================
    # Metrics: Edge AUC and Acc
    # =============================
    edge_auc = roc_auc_score(all_edge_labels, all_edge_probs)
    edge_pred = (all_edge_probs > 0.5).astype(int)
    edge_accuracy = accuracy_score(all_edge_labels, edge_pred)

    return {
        "node_loss": total_node_loss / total_batches,
        "edge_loss": total_edge_loss / total_batches,
        "node_accuracy": node_accuracy,
        "node_auc": node_auc,          # vector of 7 per-class AUCs
        "edge_accuracy": edge_accuracy,
        "edge_auc": edge_auc,
    }

def validate(model, val_loader, device):
    model.eval()
    model.to(device)

    ce_loss_fn  = torch.nn.CrossEntropyLoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()

    total_node_loss = 0.0
    total_edge_loss = 0.0
    total_batches   = 0

    all_node_probs  = []
    all_node_labels = []

    all_edge_probs  = []
    all_edge_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            node_logits, edge_logits, node_probs, edge_probs = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr
            )

            # Remove batch dimension
            node_logits = node_logits.squeeze(0)   # [N,7]
            node_probs  = node_probs.squeeze(0)    # [N,7]
            edge_logits = edge_logits.squeeze(0)   # [E,1]
            edge_probs  = edge_probs.squeeze(0)    # [E,1]

            # Convert one-hot → integer labels
            node_labels = batch.y.argmax(dim=1).long()

            # Loss
            node_loss = ce_loss_fn(node_logits, node_labels)
            edge_loss = bce_loss_fn(edge_logits.view(-1), batch.edge_y.float())

            total_node_loss += node_loss.item()
            total_edge_loss += edge_loss.item()
            total_batches   += 1

            # Store probabilities
            all_node_probs.append(node_probs.cpu())
            all_node_labels.append(batch.y.cpu())

            all_edge_probs.append(edge_probs.cpu().view(-1))
            all_edge_labels.append(batch.edge_y.cpu().view(-1))

    # Stack results
    all_node_probs  = torch.cat(all_node_probs, dim=0).numpy()   # [TotalN, 7]
    all_node_labels = torch.cat(all_node_labels, dim=0).numpy()  # [TotalN, 7]

    all_edge_probs  = torch.cat(all_edge_probs, dim=0).numpy()
    all_edge_labels = torch.cat(all_edge_labels, dim=0).numpy()

    # Convert one-hot → integer
    true_class = all_node_labels.argmax(axis=1)

    num_classes = all_node_probs.shape[1]
    y_true_onehot = label_binarize(
        true_class,
        classes=np.arange(num_classes)
    )

    # Node ROC AUC (multi-class OvR)
    node_auc = roc_auc_score(
        y_true_onehot,
        all_node_probs,
        multi_class="ovr",
        average=None
    )

    # Edge AUC
    edge_auc = roc_auc_score(all_edge_labels, all_edge_probs)

    return {
        "node_loss": total_node_loss / total_batches,
        "edge_loss": total_edge_loss / total_batches,
        "node_auc": node_auc,
        "edge_auc": edge_auc,
    }

# Class name mapping for nicer output
CLASS_NAMES = {
    0: "primary",
    1: "pileup",
    2: "fromB",
    3: "fromBC",
    4: "fromC",
    5: "other",
    6: "fake"
}

best_metric = -1
patience = 8
no_improve = 0
val_every = 10

best_epoch_init = {
                "epoch": None,
                "train_tot_loss": None,
                "train_node_loss": None,
                "train_edge_loss": None,
                "test_node_loss": None,
                "test_edge_loss": None, 
                "val_node_loss": None,
                "val_edge_loss": None,
                "test_edge_auc": None,
                "val_edge_auc": None,
                "best_metric": None,
            }

for c in CLASS_NAMES:
    best_epoch_init[f"test_auc_{CLASS_NAMES[c]}"] = None,
    best_epoch_init[f"val_auc_{CLASS_NAMES[c]}"] = None

stats = {
    "epochs": [],
    "best_epoch": best_epoch_init
}

for epoch in range(int(args.epochs)):
    
    print("EPOCH: ", epoch)

    # -------------------------
    # Train
    # -------------------------
    train_tot_loss, train_node_loss, train_edge_loss = train(
        model, train_loader, optimizer, device, epoch
    )

    print(f"[Train] Total Loss: {train_tot_loss:.4f} | "
          f"Node Loss: {train_node_loss:.4f} | "
          f"Edge Loss: {train_edge_loss:.4f}")

    # -------------------------
    # Test metrics
    # -------------------------
    test_results = test(model, test_loader, device)
    test_node_auc = test_results["node_auc"]  # dict class → auc
    test_edge_auc = test_results["edge_auc"]
    test_node_loss = test_results["node_loss"]
    test_edge_loss = test_results["edge_loss"]

    avg_test_node_auc = np.nanmean(test_node_auc)

    print(f"[Test] Node Loss: {test_node_loss:.4f} | "
          f"Edge Loss: {test_edge_loss:.4f}")

    print(f"[Test] Edge AUC: {test_edge_auc:.4f}")
    print("[Test] Node AUC per class:")

    label_names = [
        "Hard", "PU  ", "B   ", "BtoC", "C   ",
        "Oth ", "Fake"
    ]

    for i, name in enumerate(label_names):
        print(f"    {name}: {test_node_auc[i]:.4f}")

    print(f"[Test] Avg Node AUC: {avg_test_node_auc:.4f}")

    # -------------------------
    # Validation
    # -------------------------

    val_node_auc = [-1]*len(CLASS_NAMES)
    val_edge_auc = -1
    val_node_loss = -1
    val_edge_loss = -1
    val_metric = -1

    if (epoch + 1) % val_every == 0:
        print(f"\nValidating at epoch {epoch}...")

        val_results = validate(model, val_loader, device)

        val_node_auc = val_results["node_auc"]
        val_edge_auc = val_results["edge_auc"]
        val_node_loss = val_results["node_loss"]
        val_edge_loss = val_results["edge_loss"]

        avg_val_node_auc = np.nanmean(val_node_auc)

        print(f"[Val] Node Loss: {val_node_loss:.4f} | "
              f"Edge Loss: {val_edge_loss:.4f}")
        print(f"[Val] Edge AUC: {val_edge_auc:.4f}")
        print("[Val] Node AUC per class:")

        for i, name in enumerate(label_names):
            print(f"    {name}: {val_node_auc[i]:.4f}")

        print(f"[Val] Avg Node AUC: {avg_val_node_auc:.4f}")

        # --------------------------------------------
        # Compute BEST METRIC = average of edge AUC + classes 2,3,4 node AUC
        # --------------------------------------------
        important_classes = [2, 3, 4]

        val_metric = (sum(val_node_auc[c] for c in important_classes) / len(important_classes) + val_edge_auc)/2

        print(f"[Val] Combined Metric: {val_metric:.4f}")

        # Check improvement
        if val_metric > best_metric:
            print(">>> New BEST model found! Saving checkpoint.")

            best_metric = val_metric
            no_improve = 0

            best_epoch_stats = {
                "epoch": epoch + 1,
                "train_tot_loss": train_tot_loss,
                "train_node_loss": train_node_loss,
                "train_edge_loss": train_edge_loss,
                "test_node_loss": test_node_loss,
                "test_edge_loss": test_edge_loss,
                "val_node_loss": val_node_loss,
                "val_edge_loss": val_edge_loss,
                "test_edge_auc": test_edge_auc,
                "val_edge_auc": val_edge_auc,
                "metric": val_metric,
            }

            # Add node AUCs for each class to best stats
            for c in CLASS_NAMES:
                best_epoch_stats[f"test_auc_{CLASS_NAMES[c]}"] = test_node_auc[c]
                best_epoch_stats[f"val_auc_{CLASS_NAMES[c]}"] = val_node_auc[c]

            stats["best_epoch"] = best_epoch_stats

            # Save model
            save_path = os.path.join("model_files", f"model_{args.modeltag}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model → {save_path}")

        else:
            no_improve += 1
            print("No improvement in validation metric.")

        # Early stopping
        if no_improve >= patience:
            print(f"\nEARLY STOPPING after {patience} validations without improvement.")
            break

    # -------------------------
    # Save per-epoch stats
    # -------------------------
    epoch_stats = {
        "epoch": epoch + 1,
        "train_tot_loss": train_tot_loss,
        "train_node_loss": train_node_loss,
        "train_edge_loss": train_edge_loss,
        "test_node_loss": test_node_loss,
        "test_edge_loss": test_edge_loss,
        "val_node_loss": val_node_loss,
        "val_edge_loss": val_edge_loss,
        "test_edge_auc": test_edge_auc,
        "val_edge_auc": val_edge_auc,
        "metric": val_metric,
    }

    # Add AUCs for each class to epoch stats
    for c in CLASS_NAMES:
        epoch_stats[f"test_auc_{CLASS_NAMES[c]}"] = test_node_auc[c]
        epoch_stats[f"val_auc_{CLASS_NAMES[c]}"] = val_node_auc[c]

    stats["epochs"].append(epoch_stats)

    # Save stats every epoch
    with open("training_stats_" + args.modeltag + ".json", "w") as f:
        json.dump(stats, f, indent=4)

    # Periodic checkpoint
    if epoch > 0 and epoch % 10 == 0:
        ckpt = os.path.join("model_files", f"model_{args.modeltag}_e{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Checkpoint saved → {ckpt}")
