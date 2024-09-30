import argparse
import uproot
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import torch_cluster
from torch_cluster import knn_graph
from model import *
from torch.utils.data import random_split
import pickle
from tqdm import tqdm
import os


parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-l", "--label", default="", help="Label file")
parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-lg", "--load_graphs", default="", help="Load evt_graphs from a file")
parser.add_argument("-sg", "--save_graphs", default="", help="Save evt_graphs to a file")

args = parser.parse_args()

num_gvs = 6
trk_features = ['trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_eta', 'trk_phi', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
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

else:
    print("Loading files...")
    with uproot.open(datafile) as f:
        demo = f['demo']
        datatree = demo['tree']
    
    with uproot.open(labelfile) as f:
        labeltree = f['tree']
    
    
    def create_dataobj(datatree, labeltree, trk_features, k=4, nevts=3):
    
        evt_object = []
        
        for evt in range(len(datatree['trk_p'].array())):
            #if(evt>nevts): break
            print("Processing event", evt) 
    
            features = {} 
            labels = np.zeros((len((datatree['trk_p'].array())[evt]), num_gvs))
            
            for feature in trk_features:
                features[feature] = (datatree[feature].array())[evt]
        
            for trk in range(len((datatree['trk_p'].array())[evt])):
                if trk in (labeltree['ind'].array())[evt]:
                    ind = (labeltree['ind'].array())[evt].tolist().index(trk)
                    flag = (labeltree['flag'].array())[evt][ind]
                    if flag <= num_gvs - 1:
                        labels[trk, flag] = 1
    
            
            feature_matrix = np.stack([features[f] for f in trk_features], axis=1)
            feature_matrix = np.asarray(feature_matrix)
            nan_mask = ~np.isnan(feature_matrix).any(axis=1)
            shape_i = feature_matrix.shape[0]
            feature_matrix = feature_matrix[nan_mask]
            labels = labels[nan_mask]
            shape_f = feature_matrix.shape[0]
            print("NaN rows dropped:", shape_i-shape_f)
    
            print(f"Creating graphs for {shape_f} tracks")
            edge_index = create_graph(feature_matrix, k)
            data = Data(
                x=torch.tensor(feature_matrix, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(labels, dtype=torch.float)
            )
            evt_object.append(data)
    
        return evt_object
    
    
    evt_graphs = create_dataobj(datatree, labeltree, trk_features)

    if args.save_graphs != "":
        print(f"Saving evt_graphs to {args.save_graphs}...")
        with open(args.save_graphs, 'wb') as f:
            pickle.dump(evt_graphs, f)

total_events = len(evt_graphs)
train_size = int(0.8 * total_events)
test_size = total_events - train_size

train_dataset, test_dataset = random_split(evt_graphs, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


num_features = len(trk_features)
num_output = num_gvs

model = GNN(num_features, num_output)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = MulticlassBCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, criterion, optimizer, device):
    model.to(device)
    model.train()
    total_loss = 0

    for data in tqdm(train_loader, desc="Training", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()

    return total_loss / len(test_loader)

best_test_loss = float('inf')

for epoch in range(int(args.epochs)):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss = test(model, test_loader, criterion, device)

    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        save_path = os.path.join("files_models", f"best_model_{args.modeltag}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")




