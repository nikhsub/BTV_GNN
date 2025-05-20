import numpy as np
import argparse
import xgboost as xgb
import os
import pickle
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve, precision_recall_curve
from torch.utils.data import random_split
from torch_geometric.data import Data, DataLoader

parser = argparse.ArgumentParser("Baseline training")

parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-lt", "--load_train", default="", help="Path to training files")
args = parser.parse_args()

def extract_node_features(train_loader):
    X, y = [], []
    for data in train_loader:
        X.append(data.x.cpu().numpy())  # Node features
        y.append(data.y.cpu().numpy())  # Labels (0 = background, 1 = signal)
    return np.vstack(X), np.hstack(y)

def evaluate_xgb(y_true, y_probs, target_sigeff=0.70):
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)

    # Find threshold for 70% signal efficiency
    target_idx = np.argmin(np.abs(tpr - target_sigeff))
    threshold_at_target_eff = thresholds_roc[target_idx]

    precision_at_sigeff = precision[np.argmin(np.abs(recall - tpr[target_idx]))]
    bg_rejection_at_sigeff = 1 - fpr[target_idx]

    return roc_auc, pr_auc, precision_at_sigeff, bg_rejection_at_sigeff

train_hads = []
if args.load_train != "":
    if os.path.isdir(args.load_train):
        print(f"Loading training data from {args.load_train}...")
        pkl_files = [os.path.join(args.load_train, f) for f in os.listdir(args.load_train) if f.endswith('.pkl')]
    for pkl_file in pkl_files:
        print(f"Loading {pkl_file}...")
        with open(pkl_file, 'rb') as f:
            train_hads.extend(pickle.load(f))

train_hads = train_hads[:100000] #Control number of input samples here - see array splicing for more

train_len = int(0.8 * len(train_hads))
train_data, test_data = random_split(train_hads, [train_len, len(train_hads) - train_len])

train_loader = DataLoader(train_data)
test_loader = DataLoader(test_data)

X_train, y_train = extract_node_features(train_loader)
X_val, y_val = extract_node_features(test_loader)


xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=10,  # Adjust this based on your signal-to-background ratio
    learning_rate=0.05,
    n_estimators=500,
    max_depth=6,
    colsample_bytree=0.8
)

# Train the model
xgb_model.fit(X_train, y_train)

model_save_path = f"model_files/xgb_model_{args.modeltag}.pkl"
with open(model_save_path, "wb") as f:
    pickle.dump(xgb_model, f)

print(f"XGBoost model saved to {model_save_path}")

# Predict probabilities on the validation set
X_val = np.vstack(X_val)
y_probs = xgb_model.predict_proba(X_val)[:, 1]

roc_auc, pr_auc, precision_at_sigeff, bg_rejection_at_sigeff = evaluate_xgb(y_val, y_probs)

print(f"XGBoost ROC AUC: {roc_auc:.3f}")
print(f"XGBoost PR AUC: {pr_auc:.3f}")
print(f"XGBoost Precision at 70% Signal Efficiency: {precision_at_sigeff:.3f}")
print(f"XGBoost Background Rejection at 70% Signal Efficiency: {bg_rejection_at_sigeff:.3f}")



