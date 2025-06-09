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
parser.add_argument("-lo",  "--load_onnx", default="", help="Load onnx model file")
parser.add_argument("-lp", "--load_pytorch", default="", help="Load pytorch model file")
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

session = ort.InferenceSession(args.load_onnx)
model1 = GNNModel(len(trk_features), 48, edge_dim=len(edge_features))
model1.load_state_dict(torch.load(args.load_pytorch, map_location=torch.device('cpu')))
model1.eval()

all_preds = []
all_labels = []
all_preds_pth = []                                                                                                                                                                                          
all_labels_pth = []
sv_tp = 0
sv_fp = 0
sv_tn = 0
sv_fn = 0
num_trks = 0
for i, data in enumerate(val_graphs):
    with torch.no_grad():

        print(i)

        data = data.to(device)

        # Run ONNX model
        logits = run_onnx_inference(args.load_onnx, (data.x.unsqueeze(0), data.edge_index.unsqueeze(0), data.edge_attr.unsqueeze(0)), session).to(device)
        _, logits_pth = model1(data.x.unsqueeze(0), data.edge_index.unsqueeze(0), data.edge_attr.unsqueeze(0))


        preds = torch.sigmoid(logits)
        preds = preds.squeeze()

        preds = preds.cpu().numpy()
        siginds = data.siginds.cpu().numpy()
        svinds = data.svinds.cpu().numpy()
        ntrks = len(preds)

        num_trks+=ntrks

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

        preds_pth = torch.sigmoid(logits_pth)
        preds_pth = preds_pth.squeeze()
        preds_pth = preds_pth.cpu().numpy()
        labels_pth = np.zeros(len(preds_pth))
        labels_pth[siginds] = 1

        all_preds_pth.extend(preds_pth)
        all_labels_pth.extend(labels_pth)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_preds_pth = np.array(all_preds_pth)
all_labels_pth = np.array(all_labels_pth)


fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
fpr_pth, tpr_pth, thresholds_pth = roc_curve(all_labels_pth, all_preds_pth)

sv_tpr = sv_tp / (sv_tp + sv_fn) if (sv_tp + sv_fn) > 0 else 0
sv_fpr = sv_fp / (sv_fp + sv_tn) if (sv_fp + sv_tn) > 0 else 0

sv_tpr_array = np.full_like(thresholds, sv_tpr)
sv_fpr_array = np.full_like(thresholds, sv_fpr)


# Calculate the AUC (Area Under the Curve)
roc_auc = auc(fpr, tpr)
roc_auc_pth = auc(fpr_pth, tpr_pth)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(tpr, fpr, color='darkorange', label=f'ONNX model (AUC = {roc_auc:.4f})')
plt.plot(tpr_pth, fpr_pth, color='blue', label=f'pth model (AUC = {roc_auc_pth:.4f})')
plt.scatter(sv_tpr_array, sv_fpr_array, color='black', label=f'IVF TPR={sv_tpr:.4f}, FPR={sv_fpr:.4f}', zorder=5)

plt.xlabel('Signal Accuracy')
plt.ylabel('Background Mistag')

plt.title('ROC')
plt.legend(loc='upper left')
plt.yscale('log')  # Optional: Set y-axis to log scale if desired
plt.grid()

plt.savefig(f"ROC_{args.savetag}_{i+1}evts.png")
plt.close()

print("Total number of tracks", num_trks)

# === Plot the output probability distribution ===
plt.figure(figsize=(8, 6))

# Separate predictions
signal_preds = all_preds[all_labels == 1]
background_preds = all_preds[all_labels == 0]

# Plot histogram
plt.hist(signal_preds, bins=50, alpha=0.6, color='red', label='Signal', density=False)
plt.hist(background_preds, bins=50, alpha=0.6, color='blue', label='Background', density=False)
plt.yscale('log')

plt.xlabel('Predicted Probability')
plt.ylabel('# of tracks')
plt.title('Model Output Distribution')
plt.legend()
plt.grid()

plt.savefig(f"OutputDist_{args.savetag}_{i+1}evts.png")
plt.close()

