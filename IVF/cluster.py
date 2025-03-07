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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize
import vector

pion_mass = 0.13957018

parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

if args.load_train != "":
    print(f"Loading testing data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

val_graphs = train_graphs[0:20]

model = GNNModel(len(trk_features), outdim=16, heads=4, dropout=0.354)

model.load_state_dict(torch.load(args.load_model))

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

all_dbscan_labels = []
all_true_labels = []

for i, data in enumerate(val_graphs):
    with torch.no_grad():
        data = data.to(device)
        edge_index = knn_graph(data.x, k=6, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
        _, preds = model(data.x, edge_index)


        preds = preds.squeeze().cpu().numpy()

        if(preds.size==0): 
            print("No predictions")
            continue

        siginds = data.siginds.cpu().numpy()
        sigflags = data.sigflags.cpu().numpy()

        #print("SIGINDS", siginds)
        #print("SIGFLAGS", sigflags)
        #print("NEXT")

        eta  = data.x[:, 0].cpu().numpy()  # First feature (eta)
        phi  = data.x[:, 1].cpu().numpy()  # Second feature (phi)
        pt   = data.x[:, 7].cpu().numpy()
        ip2d = data.x[:, 2].cpu().numpy()
        ip3d = data.x[:, 3].cpu().numpy()

        predictions = preds  # Predicted values (already on CPU and numpy)

        #CLUSTERING
        threshold = 0.8  # Prediction threshold for clustering
        high_pred_indices = np.where(predictions > threshold)[0]
        
        #PLOTTING OUTPUTS
        signal_colors = sigflags  # Assuming sigflags are values (e.g., 0-1) used for coloring
        signal_indices = siginds

        #data_for_clustering = np.vstack((
        #eta[high_pred_indices], phi[high_pred_indices], predictions[high_pred_indices])).T

        #eps = 0.5  # Clustering distance
        #min_samples = 2
        #dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        #cluster_labels = dbscan.fit_predict(data_for_clustering)

        #unique_clusters = np.unique(cluster_labels)
        #markers = ['o', 's', '^', 'v', 'p', '*', 'D', 'X']


        colors = np.array(['red', 'green', 'orange', 'purple', 'black', 'pink'])

        added_colors = set()

        #
        ##3D
        fig1 = plt.figure(figsize=(10, 8))
        ax = fig1.add_subplot(111, projection='3d')
        
        for idx, sigflag in zip(signal_indices, signal_colors):
            color = colors[sigflag]
            label = f'Signal (GV {sigflag+1})' if color not in added_colors else None
            ax.scatter(predictions[idx], eta[idx], phi[idx], 
               c=color,  # Map 0 to blue and 1 to green
               label=label)
            added_colors.add(color)

        background_indices = [i for i in range(len(predictions)) if i not in signal_indices]
        ax.scatter(predictions[background_indices],
           eta[background_indices],
           phi[background_indices],
           c='blue', marker='+', label='Background')

        #if len(high_pred_indices) > 0:
        #    for cluster_id in unique_clusters:
        #        if cluster_id == -1:
        #            # Noise points
        #            noise_indices = high_pred_indices[cluster_labels == -1]
        #            ax.scatter(
        #                predictions[noise_indices], eta[noise_indices], phi[noise_indices],
        #                c='gray', marker='x', s=100, label='Cluster Noise'
        #            )
        #        else:
        #            # Clustered points
        #            cluster_indices = high_pred_indices[cluster_labels == cluster_id]
        #            marker = markers[cluster_id % len(markers)]
        #            ax.scatter(
        #                predictions[cluster_indices], eta[cluster_indices], phi[cluster_indices],
        #                c='none', edgecolor='k', marker=marker, s=150,
        #                label=f'Cluster {cluster_id}'
        #            )
        #            
        #            cluster_eta  = eta[cluster_indices]
        #            cluster_phi  = phi[cluster_indices]
        #            cluster_pt   = pt[cluster_indices]
        #            cluster_ip2d = ip2d[cluster_indices]
        #            cluster_ip3d = ip3d[cluster_indices]
        #            mass = np.full_like(cluster_pt, pion_mass)

        #            tracks = vector.array(
        #                {
        #                    "pt": cluster_pt,
        #                    "eta": cluster_eta,
        #                    "phi": cluster_phi,
        #                    "mass": mass,
        #                }
        #            )

        #            total_lorentz_vector = tracks.sum()

        #            # Calculate the total momentum components (px, py, pz)
        #            px_total = total_lorentz_vector.px
        #            py_total = total_lorentz_vector.py
        #            pz_total = total_lorentz_vector.pz
        #            
        #            # Initialize variables to calculate the weighted average position
        #            total_weight = 0
        #            sum_x, sum_y, sum_z = 0, 0, 0
        #            
        #            # Loop over each track in the cluster to calculate the position contributions
        #            for j in range(len(cluster_pt)):
        #                # Use the inverse of the 3D impact parameter as the weight
        #                weight = 1 / (cluster_ip3d[j] + 1e-5)  # avoid division by zero
        #            
        #                # Assuming the IP is the displacement from the primary vertex (can be a simplification)
        #                track_x = cluster_ip2d[j] * (px_total / cluster_pt[j]) * weight
        #                track_y = cluster_ip2d[j] * (py_total / cluster_pt[j]) * weight
        #                track_z = cluster_ip3d[j] * weight  # Use 3D IP for the z component
        #            
        #                # Accumulate the weighted sum of positions
        #                sum_x += track_x
        #                sum_y += track_y
        #                sum_z += track_z
        #                total_weight += weight
        #            
        #            # Compute the weighted average position for the secondary vertex
        #            sv_x = sum_x / total_weight
        #            sv_y = sum_y / total_weight
        #            sv_z = sum_z / total_weight

        #            # Calculate the components of the total momentum
        #            #total_px = total_lorentz_vector.px
        #            #total_py = total_lorentz_vector.py
        #            #total_pz = total_lorentz_vector.pz
        #            #total_E = total_lorentz_vector.E

        #            #sv_x = total_px / total_E
        #            #sv_y = total_py / total_E
        #            #sv_z = total_pz / total_E

        #            print(f"EVT {i} SV x = {sv_x}, y = {sv_y}, z = {sv_z}")

                                        
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Eta')
        ax.set_zlabel('Phi')
        ax.set_title('Event Track Plot')
        ax.legend(loc='best')
        
        #plt.show()
        plt.savefig(f"Evt{i}_3dcluster.png")
        print("NEXT")

        ##2D

        #plt.figure(figsize=(10, 8))

        #for idx, sigflag in zip(signal_indices, signal_colors):
        #    plt.scatter(eta[idx], phi[idx],
        #        c=colors[sigflag],
        #        label=f'Signal (flag={sigflag})' if idx == signal_indices[0] else None)

        #background_indices = [i for i in range(len(predictions)) if i not in signal_indices]
        #plt.scatter(eta[background_indices],
        #            phi[background_indices],
        #            c='blue', marker='+', alpha=0.3, label='Background')
        #
        ## Add labels, legend, and title
        #plt.xlabel('Eta')
        #plt.ylabel('Phi')
        #plt.title('2D Plot of Tracks in Eta-Phi Space')
        #plt.legend(loc='best')
        #plt.grid(True)
        #
        ## Show the plot
        #plt.savefig(f"Evt{i}_2d.png")



        #print("Sigflags", sigflags)

        ##svinds = data.svinds.cpu().numpy()
        #pred_mask = preds > thres
        #sigind_to_flag = dict(zip(siginds, sigflags))

        #pred_tracks = data.x[pred_mask].cpu().numpy()
        #if(pred_tracks.size ==0) : continue

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
        
        #true_labels = np.full(len(pred_tracks), -1)

        #for i, track_idx in enumerate(np.where(pred_mask)[0]):
        #    if track_idx in sigind_to_flag:   
        #        true_labels[i] = sigind_to_flag[track_idx]  # Assign corresponding sigflag
        #    else:
        #        true_labels[i] = -1  # Assign -1 if not a signal track

        #dbscan = DBSCAN(eps=1.0, min_samples=2)
        #predicted_labels = dbscan.fit_predict(pred_tracks[:, :2])

        #print("TRUE", true_labels)
        #print("PRED", predicted_labels)
        #print("NEXT")

        #all_dbscan_labels.extend(predicted_labels)
        #all_true_labels.extend(true_labels)

#ari_score = adjusted_rand_score(all_true_labels, all_dbscan_labels)
#nmi_score = normalized_mutual_info_score(all_true_labels, all_dbscan_labels)
#
#print(f"Adjusted Rand Index: {ari_score}")
#print(f"Normalized Mutual Information Score: {nmi_score}")

    


        

                


        
            

