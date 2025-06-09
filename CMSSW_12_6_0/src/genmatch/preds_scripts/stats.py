import ROOT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)

# Load ROOT tree
infile = "ntup_onnxfix_0906.root"
Infile = ROOT.TFile(infile, "READ")
tree = Infile.Get("tree")

# Collect labels and scores
y_true = []
y_score = []

for evt in tree:
    sig_set = set(evt.sig_ind)
    #print("Siginds", sig_set)
    preds = evt.preds

    for idx, score in enumerate(preds):
        #if not np.isfinite(score):
        #    score = 0.0
        y_score.append(score)
        y_true.append(1 if idx in sig_set else 0)


y_score = np.array(y_score)
y_true = np.array(y_true)

# AUC scores
roc_auc = roc_auc_score(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)
print("\n=== AUC Scores ===")
print(f"ROC AUC: {roc_auc:.6f}")
print(f"PR  AUC: {pr_auc:.6f}")

# ROC and PR curves
fpr, tpr, _ = roc_curve(y_true, y_score)
precision, recall, _ = precision_recall_curve(y_true, y_score)

# Score distributions
signal_scores = y_score[y_true == 1]
background_scores = y_score[y_true == 0]

# Plot
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# 1. Score Distribution
bins = np.linspace(0, 1, 50)
axs[0].hist(background_scores, bins=bins, alpha=0.5, label="Background", color="blue", density=False)
axs[0].hist(signal_scores, bins=bins, alpha=0.7, label="Signal", color="red", density=False)
axs[0].set_yscale('log')
axs[0].set_title("Track Score Distribution")
axs[0].set_xlabel("Model Score")
axs[0].set_ylabel("Density")
axs[0].legend()
axs[0].grid(True)

# 2. ROC Curve
axs[1].plot(tpr, fpr, label=f"AUC = {roc_auc:.4f}", color="purple")
axs[1].set_yscale("log")
axs[1].set_title("ROC Curve")
axs[1].set_xlabel("Signal Efficiency (TPR)")
axs[1].set_ylabel("Background Mistag (FPR)")
axs[1].legend()
axs[1].grid(True)

# 3. PR Curve
axs[2].plot(recall, precision, label=f"AUC = {pr_auc:.4f}", color="green")
axs[2].set_title("PR Curve")
axs[2].set_xlabel("Recall")
axs[2].set_ylabel("Precision")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("summary_model_eval.png", dpi=300)

