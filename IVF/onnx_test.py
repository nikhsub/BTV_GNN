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
import onnx
import onnxruntime as ort
ort.set_default_logger_severity(3)

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-lt", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-st", "--savetag", default="", help="Savetag for pngs")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
edge_features = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1', 'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']

if args.load_train != "":
    print(f"Loading testing data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

val_graphs = train_graphs[:]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_onnx_inference(onnx_model_path, inputs, session):

    # Prepare inputs as a dictionary
    ort_inputs = {
        session.get_inputs()[0].name: inputs[0].cpu().numpy(),
        session.get_inputs()[1].name: inputs[1].cpu().numpy(),
        session.get_inputs()[2].name: inputs[2].cpu().numpy()
    }

    # Run the model
    ort_outs = session.run(None, ort_inputs)

    # Convert ONNX output back to torch tensor
    return torch.tensor(ort_outs[1])

session = ort.InferenceSession(args.load_model)

all_preds = []
all_labels = []
sv_tp = 0
sv_fp = 0
sv_tn = 0
sv_fn = 0
for i, data in enumerate(val_graphs):
    with torch.no_grad():

        print(i)

        data = data.to(device)

        # Run ONNX model
        logits = run_onnx_inference(args.load_model, (data.x, data.edge_index, data.edge_attr), session).to(device)

        preds = torch.sigmoid(logits)
        preds = preds.squeeze()

        preds = preds.cpu().numpy()
        siginds = data.siginds.cpu().numpy()
        svinds = data.svinds.cpu().numpy()
        ntrks = len(preds)

        labels = np.zeros(len(preds))
        labels[siginds] = 1

        all_preds.extend(preds)
        all_labels.extend(labels)

        #IVF compare
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

fpr, tpr, thresholds = roc_curve(all_labels, all_preds)

print(fpr)
print(tpr)

sv_tpr = sv_tp / (sv_tp + sv_fn) if (sv_tp + sv_fn) > 0 else 0
sv_fpr = sv_fp / (sv_fp + sv_tn) if (sv_fp + sv_tn) > 0 else 0

sv_tpr_array = np.full_like(thresholds, sv_tpr)
sv_fpr_array = np.full_like(thresholds, sv_fpr)

print(sv_tpr_array)
print(sv_fpr_array)

# Calculate the AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(tpr, fpr, color='darkorange', label=f'GNN model (AUC = {roc_auc:.2f})')
plt.scatter(sv_tpr_array, sv_fpr_array, color='blue', label=f'IVF TPR={sv_tpr:.2f}, FPR={sv_fpr:.2f}', zorder=5)

plt.xlabel('Signal Accuracy')
plt.ylabel('Background Mistag')

plt.title('ROC')
plt.legend(loc='upper left')
plt.yscale('log')  # Optional: Set y-axis to log scale if desired
plt.grid()

plt.savefig(f"ROC_{args.savetag}_{i+1}evts.png")
plt.close()
