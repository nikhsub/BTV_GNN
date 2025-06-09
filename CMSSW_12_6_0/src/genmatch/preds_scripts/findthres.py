import ROOT
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# Open ROOT file and tree
infile = "ntup_onnxfix_0906.root"
Infile = ROOT.TFile(infile, 'READ')
tree = Infile.Get("tree")

# Storage for global labels and scores
y_true = []
y_score = []

for i, evt in enumerate(tree):
    sig_set = set(evt.sig_ind)
    preds = evt.preds
    for idx, score in enumerate(preds): 
        if not np.isfinite(score):  # Skip nan or inf
            score = 0.0
        y_score.append(score)
        y_true.append(1 if idx in sig_set else 0)

# Convert to numpy arrays
y_true = np.array(y_true)
y_score = np.array(y_score)


# Sweep thresholds
thresholds = np.linspace(0.0, 1.0, 200)
best_f1 = 0
best_thresh = 0
best_metrics = {}

for t in thresholds:
    y_pred = (y_score >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t
        best_metrics = {
            'precision': p,
            'recall': r,
            'f1': f1,
            'threshold': t,
            'tp': int(np.sum((y_pred == 1) & (y_true == 1))),
            'fp': int(np.sum((y_pred == 1) & (y_true == 0))),
            'fn': int(np.sum((y_pred == 0) & (y_true == 1))),
            'tn': int(np.sum((y_pred == 0) & (y_true == 0)))
        }

# Final AUCs
roc_auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)

# Print best threshold results
print("=== Best Threshold for Max F1 ===")
print(f"Threshold: {best_metrics['threshold']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall:    {best_metrics['recall']:.4f}")
print(f"F1 Score:  {best_metrics['f1']:.4f}")
print(f"TP: {best_metrics['tp']}, FP: {best_metrics['fp']}, FN: {best_metrics['fn']}, TN: {best_metrics['tn']}")

# AUC scores
print("\n=== AUC Scores ===")
print(f"ROC AUC: {roc_auc:.6f}")
print(f"PR  AUC: {pr_auc:.6f}")


