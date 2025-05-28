import ROOT
from collections import Counter
import numpy as np

infile = "ntup_preds.root"
Infile = ROOT.TFile(infile, 'READ')

tree = Infile.Get('tree')

score_thresh = 0.36

# Initialize global counters
TP_model = FP_model = FN_model = TN_model = 0
TP_ivf = FP_ivf = FN_ivf = TN_ivf = 0

for i, evt in enumerate(tree):
    sig_set = set(evt.sig_ind)         # true signal tracks
    ivf_set = set(evt.SVtrk_ind)       # IVF-selected tracks
    preds = evt.preds                  # model score per track
    all_tracks = set(range(len(preds)))  # all track indices

    # Model-predicted signal tracks
    model_set = set(idx for idx, score in enumerate(preds) if (not np.isnan(score)) and score >= score_thresh)

    # IVF
    TP_ivf += len([idx for idx in ivf_set if idx in sig_set])
    FP_ivf += len([idx for idx in ivf_set if idx not in sig_set])
    FN_ivf += len([idx for idx in sig_set if idx not in ivf_set])
    TN_ivf += len([idx for idx in all_tracks if idx not in ivf_set and idx not in sig_set])

    # Model
    TP_model += len([idx for idx in model_set if idx in sig_set])
    FP_model += len([idx for idx in model_set if idx not in sig_set])
    FN_model += len([idx for idx in sig_set if idx not in model_set])
    TN_model += len([idx for idx in all_tracks if idx not in model_set and idx not in sig_set])

# Metric functions
def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def signal_accuracy(tp, fn):
    return recall(tp, fn)  # Equivalent to recall, renamed for clarity

def background_rejection(tn, fp):
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def f1(tp, fp, fn):
    p = precision(tp, fp)
    r = recall(tp, fn)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

# Report IVF
print("=== IVF Stats ===")
print(f"TP: {TP_ivf}, FP: {FP_ivf}, FN: {FN_ivf}, TN: {TN_ivf}")
print(f"Precision: {precision(TP_ivf, FP_ivf):.4f}")
print(f"Recall:    {recall(TP_ivf, FN_ivf):.4f}")
print(f"F1 Score:  {f1(TP_ivf, FP_ivf, FN_ivf):.4f}")
print(f"Signal Accuracy:       {signal_accuracy(TP_ivf, FN_ivf):.4f}")
print(f"Background Rejection:  {background_rejection(TN_ivf, FP_ivf):.4f}")

# Report Model
print("\n=== Model Stats ===")
print(f"TP: {TP_model}, FP: {FP_model}, FN: {FN_model}, TN: {TN_model}")
print(f"Precision: {precision(TP_model, FP_model):.4f}")
print(f"Recall:    {recall(TP_model, FN_model):.4f}")
print(f"F1 Score:  {f1(TP_model, FP_model, FN_model):.4f}")
print(f"Signal Accuracy:       {signal_accuracy(TP_model, FN_model):.4f}")
print(f"Background Rejection:  {background_rejection(TN_model, FP_model):.4f}")
