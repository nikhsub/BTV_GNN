#!/usr/bin/env python3
import os
import subprocess
import argparse
import numpy as np
import torch
import xgboost as xgb
import pickle
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data

parser = argparse.ArgumentParser("XGBoost baseline on EOS .pt files")
parser.add_argument("-ld", "--load_train", required=True,
                    help="EOS directory containing .pt graph files (/store/.../*)")
parser.add_argument("-td", "--tmpdir", required=True,
                    help="Local scratch directory to stream files through")
parser.add_argument("-tag", "--modeltag", default="xgb",
                    help="Tag for saved model filename")
parser.add_argument("--max_files", type=int, default=None,
                    help="Limit number of files for testing")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# EOS HELPERS
# ---------------------------------------------------------------------------

def list_pt_files(eos_dir):
    """Return full EOS paths to .pt files in eos_dir."""
    if not eos_dir.startswith("/"):
        eos_dir = "/" + eos_dir

    print(f"Listing EOS directory: {eos_dir}")

    proc = subprocess.Popen(
        ["xrdfs", "root://cmseos.fnal.gov", "ls", eos_dir],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = proc.communicate()
    if proc.returncode != 0:
        print("xrdfs error:", err.decode())
        raise RuntimeError("Failed to list EOS directory")

    files = []
    for line in out.decode().splitlines():
        line = line.strip()
        if line.endswith(".pt"):
            # Ensure absolute path
            if not line.startswith("/store/"):
                line = eos_dir.rstrip("/") + "/" + line
            files.append(line)

    print(f"Found {len(files)} .pt files.")
    return files


def copy_from_eos(eos_path, tmpdir):
    """Download EOS file into tmpdir."""
    os.makedirs(tmpdir, exist_ok=True)
    dest = tmpdir if tmpdir.endswith("/") else tmpdir + "/"

    cmd = f"xrdcp root://cmseos.fnal.gov/{eos_path} {dest}"
    print(f"Copying: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Failed to xrdcp {eos_path}")

    return os.path.join(tmpdir, os.path.basename(eos_path))


# ---------------------------------------------------------------------------
# CONVERT GRAPHS → FEATURE MATRIX + LABEL VECTOR
# ---------------------------------------------------------------------------
def extract_from_graph_list(graphs):
    """Convert list of PyG graphs → (X, y) arrays."""
    X_list = []
    y_list = []

    for g in graphs:
        X_list.append(g.x.cpu().numpy())
        y_list.append(g.y.argmax(dim=1).cpu().numpy())  # 7-class integer labels

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y


# ---------------------------------------------------------------------------
# STREAM PT FILES FROM EOS → X, y
# ---------------------------------------------------------------------------
def load_dataset_streaming(eos_dir, tmpdir, max_files=None):
    files = list_pt_files(eos_dir)
    if max_files:
        files = files[:max_files]

    X_all = []
    y_all = []

    for eos_path in files:
        local_file = copy_from_eos(eos_path, tmpdir)

        print(f"Loading {local_file}...")
        graphs = torch.load(local_file)

        X, y = extract_from_graph_list(graphs)
        X_all.append(X)
        y_all.append(y)

        # Immediately delete from disk to save space!
        os.remove(local_file)

    # Combine into single dataset
    X = np.vstack(X_all)
    y = np.hstack(y_all)

    print(f"Final dataset: X = {X.shape}, y = {y.shape}")
    return X, y


# ---------------------------------------------------------------------------
# TRAIN AND EVALUATE XGBOOST
# ---------------------------------------------------------------------------
def train_xgb_multiclass(X_train, y_train):
    """Train 7-class XGBoost classifier."""
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=7,
        eval_metric="mlogloss",
        learning_rate=0.05,
        n_estimators=400,
        max_depth=8,
        tree_method="hist",
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    return model


def compute_ovr_auc(model, X, y):
    """Compute OVR per-class AUC."""
    probs = model.predict_proba(X)
    aucs = {}
    for c in range(7):
        y_true = (y == c).astype(int)
        y_score = probs[:, c]
        try:
            aucs[c] = roc_auc_score(y_true, y_score)
        except:
            aucs[c] = float("nan")
    return aucs


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------

print("\n========== LOADING DATASET ==========")
X, y = load_dataset_streaming(args.load_train, args.tmpdir, args.max_files)

# Train/test split
n = len(y)
idx = np.arange(n)
np.random.shuffle(idx)

train_idx = idx[: int(0.8*n)]
test_idx  = idx[int(0.8*n):]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"Train size = {len(y_train)}, Test size = {len(y_test)}")

print("\n========== TRAINING XGBOOST ==========")
model = train_xgb_multiclass(X_train, y_train)

# Save model
os.makedirs("model_files", exist_ok=True)
model_path = f"model_files/xgb_model_{args.modeltag}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Saved XGBoost model to {model_path}")

print("\n========== EVALUATION ==========")
aucs = compute_ovr_auc(model, X_test, y_test)

for c in range(7):
    print(f"Class {c} AUC = {aucs[c]}")

print("\nDone.")

