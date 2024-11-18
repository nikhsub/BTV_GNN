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
from conmodel import *
import pprint
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-e",  "--event", default=False, action="store_true", help="Running on event?")
parser.add_argument("-st", "--savetag", default="", help="Savetag for pngs")
#parser.add_argument("-s", "--scalar", default="", help="Scalar")
parser.add_argument("-had", "--hadron", default=False, action="store_true", help="Testing on hadronic samples")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

if args.load_train != "":
    print(f"Loading testing data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

def scale_features(x, mean, std, eps=1e-6):
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    x_scaled = (x - mean) / (std + eps)
    return x_scaled

val_graphs = train_graphs[:]

model = GNNModel(len(trk_features), 512)

model.load_state_dict(torch.load(args.load_model))

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#with open(args.scalar, 'r') as f:
#    scale_params = json.load(f)
#
#scale_mean = torch.tensor(scale_params['mean'], device=device)
#scale_std = torch.tensor(scale_params['std'], device=device)

model.to(device)

if(not args.event):
    for i, data in enumerate(val_graphs):
        with torch.no_grad():
    
            data = data.to(device)

            data.x = scale_features(data.x)
            edge_index = knn_graph(data.x, k=5, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
    
            _, preds = model(data, edge_index, device)
    
            print("PREDS", preds)
            print("LABELS", data.y)
    
            preds = preds.squeeze().cpu().numpy()  # Convert to numpy for plotting
            labels = data.y.cpu().numpy()          # Convert labels to numpy
    
            bins = np.linspace(0, 1, 50)
    
            plt.figure(figsize=(8, 6))
            plt.hist(preds[labels == 1], bins=bins, alpha=0.5, label='Signal', color='red')
            plt.hist(preds[labels == 0], bins=bins, alpha=0.5, label='Background', color='blue')
            plt.xlabel("Node Probability")
            plt.ylabel("Frequency")
            plt.title("Hadron Node Probabilities")
            plt.legend()
            
            # Save plot to file
            plt.savefig(f"histogram_hadron_{i}.png")
            plt.close()

if(args.event):

    all_preds = []
    all_labels = []
    sv_tp = 0
    sv_fp = 0
    sv_tn = 0
    sv_fn = 0

    for i, data in enumerate(val_graphs):
        with torch.no_grad():


            data = data.to(device)
            #data.x = scale_features(data.x, scale_mean, scale_std)

            edge_index = knn_graph(data.x, k=5, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
            _, preds = model(data, edge_index, device)
            preds = preds.squeeze().cpu().numpy()
            siginds = data.siginds.cpu().numpy()
            svinds = data.svinds.cpu().numpy()
            ntrks = len(preds)

            #print(preds)

            #sig_preds = [preds[i] for i in siginds]
            ##bkg_preds = [preds[i] for i in data.bkginds.cpu().numpy()]
            #bkg_preds = [preds[i] for i in range(len(preds)) if i not in siginds]

            #plt.figure(figsize=(10, 5))
            #plt.hist(sig_preds, bins=50, color='red', alpha=0.5, label='Signal', range=(0, 1))
            #plt.hist(bkg_preds, bins=50, color='blue', alpha=0.5, label='Background', range=(0, 1))
            #plt.xlabel('Node Probability')
            #plt.ylabel('Frequency')
            #plt.legend()
            #plt.title('Event Node Probabilities')
            #plt.yscale('log')

            ## Save the histogram to a PNG file
            #plt.savefig(f"histogram_had_oneevt_{i}.png")
            #plt.close()

            labels = np.zeros(len(preds))
            labels[siginds] = 1  # Set signal indices to 1

            #print("PREDS", preds)
            #print("LABELS", labels)

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
    
    #print(all_preds)
    #print(all_labels)
    # Get the FPR, TPR, and thresholds for the ROC curve
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

    #plt.annotate(f'TPR={sv_tpr:.2f}, FPR={sv_fpr:.2f}',
    #         xy=(sv_tpr, sv_fpr),
    #         xytext=(sv_tpr + 0.05, sv_fpr - 0.05),
    #         arrowprops=dict(arrowstyle='->', lw=1),
    #         fontsize=10, color='blue')

    #plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line for random guessing
    plt.xlabel('Signal Accuracy')
    plt.ylabel('Background Mistag')

    if (args.hadron): plt.title('ROC - Hadronic Events')
    elif (not args.hadron): plt.title('ROC - Leptonic Events')
    plt.legend(loc='upper left')
    plt.yscale('log')  # Optional: Set y-axis to log scale if desired
    plt.grid()

    if (args.hadron): plt.savefig(f"ROC_{args.savetag}_{i+1}hadevts.png")
    elif (not args.hadron): plt.savefig(f"ROC_{args.savetag}_{i+1}lepevts.png")
    plt.close()

    #precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)

    ## Calculate the Average Precision (AP) score
    #ap_score = average_precision_score(all_labels, all_preds)
    #
    ## Plot the Precision-Recall curve
    #plt.figure(figsize=(8, 6))
    #plt.plot(recall, precision, label=f'AP = {ap_score:.2f}')
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.title('Precision-Recall Curve')
    #plt.legend()
    #plt.grid()
    #plt.savefig(f"PR_2l_3010_{i+1}evts.png")
    #plt.close() 


                


        
            

