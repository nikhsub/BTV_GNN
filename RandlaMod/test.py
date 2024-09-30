import argparse
import uproot
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import torch_cluster
from torch_cluster import knn_graph
from randlanet import *
from torch.utils.data import random_split, Subset
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F


parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-l", "--label", default="", help="Label file")
parser.add_argument("-lg", "--load_graphs", default="", help="Load evt_graphs from a file")
parser.add_argument("-f", "--filename", default="", help="Model file name")

args = parser.parse_args()

num_gvs = 6
trk_features = ['trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_phi', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
truth_features = ['flag', 'ind']

datafile  = "files/"+args.data
labelfile = "files/"+args.label

datatree = None
labeltree = None

evt_graphs = None

if args.load_graphs != "":
    print(f"Loading evt_graphs from {args.load_graphs}...")
    with open(args.load_graphs, 'rb') as f:
        evt_graphs = pickle.load(f)

evt_graphs = evt_graphs[3000:3100]

total_events = len(evt_graphs)
test_size = int(total_events)

test_indices = torch.randperm(len(evt_graphs))[:test_size]

test_dataset = Subset(evt_graphs, test_indices)


test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(len(test_loader.dataset))

num_features = len(trk_features)
num_output = num_gvs

model = randlanet(num_features, num_output, 6)

model.load_state_dict(torch.load(f"files_models/{args.filename}"))

def test(model, test_loader):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_with_ones = 0
    low_value_count = 0
    low_variance_count = 0

    with torch.no_grad():
        for data in test_loader:
            ei = knn_graph(data.pos, 6)
            output = model(data.x, data.pos, ei)
            loss = F.binary_cross_entropy(output, data.y)
            total_loss += loss.item()
            
            
            for i in range(data.y.size(0)):
                label = data.y[i]
                pred = output[i]
                
                if label.sum() > 0:  # If there's at least one '1' in the label
                    print("SIGlabel", label)
                    print("SIGpred", pred)
                    mean_pred = pred.mean().item()
                    std_pred = pred.std().item()
                    #print(f"SIG Sample {i}: Mean prediction = {mean_pred:.4f}, Std dev = {std_pred:.4f}")
                    total_with_ones += 1
                    if torch.argmax(pred) == torch.argmax(label):
                        correct_predictions += 1
                else:  # If the label is all zeros
                    print("BKG", pred)
                    mean_pred = pred.mean().item()
                    std_pred = pred.std().item()

                    # Debugging print statements
                    #print(f"BKG Sample {i}: Mean prediction = {mean_pred:.4f}, Std dev = {std_pred:.4f}")

                    if pred.mean().item() < 0.1 and pred.std().item() < 0.05:
                        low_value_count += 1
                    if pred.std().item() < 0.05:
                        low_variance_count += 1

    accuracy = correct_predictions / total_with_ones if total_with_ones > 0 else 0

    low_value_accuracy = low_value_count / (len(test_loader.dataset) - total_with_ones) if len(test_loader.dataset) - total_with_ones > 0 else 0
    low_variance_accuracy = low_variance_count / (len(test_loader.dataset) - total_with_ones) if len(test_loader.dataset) - total_with_ones > 0 else 0

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"Accuracy for samples with at least one '1': {accuracy:.4f}")
    print(f"Low-value accuracy for samples with no '1's: {low_value_accuracy:.4f}")
    print(f"Low-variance accuracy for samples with no '1's: {low_variance_accuracy:.4f}")

    return total_loss / len(test_loader), accuracy, low_value_accuracy, low_variance_accuracy

test(model, test_loader)






