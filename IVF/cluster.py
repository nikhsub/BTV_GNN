import warnings
warnings.filterwarnings("ignore")

import argparse
import uproot
import numpy as np
import torch
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
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-s", "--scaler", default="", help="Path to scaler file")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

if args.load_train != "":
    print(f"Loading testing data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

def scale_data(data_list, scaler):
    for data in data_list:
        data.x = torch.tensor(scaler.transform(data.x), dtype=torch.float)
    return data_list

scaler = joblib.load(args.scaler)

val_graphs = train_graphs[:]

val_graphs = scale_data(val_graphs, scaler)

model = GNNModel(len(trk_features), 512)

model.load_state_dict(torch.load(args.load_model))

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

thres = 0.9

all_dbscan_labels = []
all_true_labels = []

for i, data in enumerate(val_graphs):
    with torch.no_grad():
        data = data.to(device)
        edge_index = knn_graph(data.x, k=5, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
        _, preds = model(data, edge_index, device)

        preds = preds.squeeze().cpu().numpy()

        siginds = data.siginds.cpu().numpy()
        sigflags = data.sigflags.cpu().numpy()

        print("Sigflags", sigflags)

        #svinds = data.svinds.cpu().numpy()
        pred_mask = preds > thres
        sigind_to_flag = dict(zip(siginds, sigflags))

        pred_tracks = data.x[pred_mask].cpu().numpy()
        if(pred_tracks.size ==0) : continue

        #k=1
        #neigh = NearestNeighbors(n_neighbors=k)
        #neigh.fit(pred_tracks)
        ## Compute the k-nearest distances (distance to the k-th nearest neighbor)
        #distances, indices = neigh.kneighbors(pred_tracks)
        ## Sort distances to the k-th nearest neighbor for each point
        #k_distances = np.sort(distances[:, k - 1])  # k-th nearest distances for all points
        ## Plot the k-distance graph
        #plt.figure(figsize=(8, 4))
        #plt.plot(k_distances)
        #plt.xlabel("Points sorted by distance to {}-th nearest neighbor".format(k))
        #plt.ylabel("Distance to {}-th nearest neighbor".format(k))
        #plt.title("k-Distance Graph")
        #plt.grid()
        #plt.savefig(f"k-1dist{i}.png")
        
        true_labels = np.full(len(pred_tracks), -1)

        for i, track_idx in enumerate(np.where(pred_mask)[0]):
            if track_idx in sigind_to_flag:   
                true_labels[i] = sigind_to_flag[track_idx]  # Assign corresponding sigflag
            else:
                true_labels[i] = -1  # Assign -1 if not a signal track

        dbscan = DBSCAN(eps=1.0, min_samples=2)
        predicted_labels = dbscan.fit_predict(pred_tracks[:, :2])

        print("TRUE", true_labels)
        print("PRED", predicted_labels)
        print("NEXT")

        all_dbscan_labels.extend(predicted_labels)
        all_true_labels.extend(true_labels)

ari_score = adjusted_rand_score(all_true_labels, all_dbscan_labels)
nmi_score = normalized_mutual_info_score(all_true_labels, all_dbscan_labels)

print(f"Adjusted Rand Index: {ari_score}")
print(f"Normalized Mutual Information Score: {nmi_score}")

    


        

                


        
            

