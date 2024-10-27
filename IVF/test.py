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
import pprint
import matplotlib.pyplot as plt
import joblib


parser = argparse.ArgumentParser("GNN testing")

parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-lm",  "--load_model", default="", help="Load model file")
parser.add_argument("-e",  "--event", default=False, action="store_true", help="Running on event?")

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

scaler = joblib.load('scaler_contloss_2610.pkl')

val_graphs = train_graphs[-500:-480]

val_graphs = scale_data(val_graphs, scaler)

model = GNNModel(len(trk_features), 512)

model.load_state_dict(torch.load(args.load_model))

model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

if(not args.event):
    for i, data in enumerate(val_graphs):
        with torch.no_grad():
    
            data = data.to(device)
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
    for i, data in enumerate(val_graphs):
        with torch.no_grad():

            data = data.to(device)
            edge_index = knn_graph(data.x, k=5, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
            _, preds = model(data, edge_index, device)
            preds = preds.squeeze().cpu().numpy()
            siginds = data.siginds.cpu().numpy()

            print(preds)

            sig_preds = [preds[i] for i in siginds]
            #bkg_preds = [preds[i] for i in data.bkginds.cpu().numpy()]
            bkg_preds = [preds[i] for i in range(len(preds)) if i not in siginds]

            plt.figure(figsize=(10, 5))
            plt.hist(sig_preds, bins=20, color='red', alpha=0.5, label='Signal', range=(0, 1))
            plt.hist(bkg_preds, bins=20, color='blue', alpha=0.5, label='Background', range=(0, 1))
            plt.xlabel('Node Probability')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title('Event Node Probabilities')

            # Save the histogram to a PNG file
            plt.savefig(f"histogram_evt_{i}_2.png")
            plt.close()


                


        
            

