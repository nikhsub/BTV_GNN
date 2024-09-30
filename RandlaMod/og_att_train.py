import argparse
import uproot
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
from randlanet import *
from buildblock import *
from torch.utils.data import random_split
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F
from tqdm import tqdm
import torch_cluster
from torch_cluster import knn_graph
import math


parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-l", "--label", default="", help="Label file")
parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-lg", "--load_graphs", default="", help="Load evt_graphs from a file")
parser.add_argument("-sg", "--save_graphs", default="", help="Save evt_graphs to a file")

args = parser.parse_args()

num_gvs = 6
ntr_evts = 1200
trk_features = ['trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
pos_features = ['trk_eta', 'trk_phi']
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
#            if(evt>nevts): break
            print("Processing event", evt) 
    
            features = {} 
            pos = {}
            labels = np.zeros((len((datatree['trk_p'].array())[evt]), num_gvs))
            
            for feature in trk_features:
                features[feature] = (datatree[feature].array())[evt]

            for feature in pos_features:
                pos[feature] = (datatree[feature].array())[evt]
 
            for trk in range(len((datatree['trk_p'].array())[evt])):
                if trk in (labeltree['ind'].array())[evt]:
                    ind = (labeltree['ind'].array())[evt].tolist().index(trk)
                    flag = (labeltree['flag'].array())[evt][ind]
                    if flag <= num_gvs - 1:
                        labels[trk, flag] = 1
    
            
            feature_matrix = np.stack([features[f] for f in trk_features], axis=1)
            pos_matrix = np.stack([pos[f] for f in pos_features], axis=1)
            feature_matrix = np.asarray(feature_matrix)
            nan_mask = ~np.isnan(feature_matrix).any(axis=1)
            shape_i = feature_matrix.shape[0]
            feature_matrix = feature_matrix[nan_mask]
            labels = labels[nan_mask]
            pos_matrix = pos_matrix[nan_mask]
            shape_f = feature_matrix.shape[0]
            print("NaN rows dropped:", shape_i-shape_f)
    
            print(f"Creating data for {shape_f} tracks")
            data = Data(
                x=torch.tensor(feature_matrix, dtype=torch.float),
                pos = torch.tensor(pos_matrix, dtype=torch.float),
                y=torch.tensor(labels, dtype=torch.float)
            )
            evt_object.append(data)
    
        return evt_object
    
    
    evt_graphs = create_dataobj(datatree, labeltree, trk_features)

    if args.save_graphs != "":
        print(f"Saving evt_graphs to {args.save_graphs}...")
        with open(args.save_graphs, 'wb') as f:
            pickle.dump(evt_graphs, f)

evt_graphs = evt_graphs[:ntr_evts]

total_events = len(evt_graphs)
train_size = int(0.8 * total_events)
test_size = total_events - train_size

train_dataset, test_dataset = random_split(evt_graphs, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


num_features = len(trk_features)
num_output = num_gvs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = randlanet(num_features, num_output, 6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

glob_weight1 = 50
glob_weight2 = 1
best_acc = 0.0
best_norm = float('inf')

def update_weight(current_norm, current_acc):
    global glob_weight1, glob_weight2, best_acc, best_norm

    print(f"Curr Acc: {current_acc:.5f}")
    print(f"Best Acc: {best_acc:.5f}")
    print(f"Current norm: {current_norm:.5f}")
    print(f"Best norm: {best_norm:.5f}")

    # Define step size for weight adjustment
    step_size1 = 50
    step_size2 = 10

    if current_acc > best_acc and (current_norm > best_norm or current_norm==1):
        # Accuracy improved, but background norm worsened
        glob_weight2 += step_size2  # Increase background weight to penalize more
    elif current_acc > best_acc and current_norm <= best_norm:
        # Accuracy improved and background norm improved
        glob_weight2 = max(5, glob_weight2 - step_size2)  # Decrease background weight slightly, with min bound
    elif current_acc <= best_acc and (current_norm > best_norm or current_norm==1):
        # Accuracy worsened and background norm worsened
        glob_weight1 += step_size1  # Increase signal weight
        glob_weight2 += step_size2  # Increase background weight
    elif current_acc <= best_acc and current_norm <= best_norm:
        # Accuracy worsened but background norm improved
        glob_weight1 += step_size1  # Increase signal weight
        glob_weight2 = max(5, glob_weight2 - step_size2)  # Decrease background weight, with min bound

    if(current_norm==0):
        glob_weight1 += step_size1*3

    glob_weight1 = min(5000, glob_weight1)
    glob_weight2 = min(5000, glob_weight2)

    # Update best accuracy and norm
    if current_acc > best_acc:
        best_acc = current_acc
    if current_norm < best_norm:
        best_norm = current_norm

def custom_loss(output, target, weight):
    
    weight1 = weight[0]

    weight2 = weight[1]

    bce_loss = F.binary_cross_entropy(output, target, reduction='none')

    class_present = target.sum(dim=1) > 0  # This will be a boolean tensor

    weighted_loss = torch.where(
        class_present.unsqueeze(1),
        bce_loss * weight1,
        bce_loss * weight2
    )

    mean_loss = weighted_loss.mean()

    return mean_loss

def custom_test_loss(output, target):

    weight1 =  1#Weight for signal

    weight2 = 1 #Weight for background

    bce_loss = F.binary_cross_entropy(output, target, reduction='none')

    class_present = target.sum(dim=1) > 0  # This will be a boolean tensor

    weighted_loss = torch.where(
        class_present.unsqueeze(1),
        bce_loss * weight1,
        bce_loss * weight2
    )

    mean_loss = weighted_loss.mean()

    return mean_loss


def train(model, train_loader,  optimizer, device, epoch):
    model.train()
    total_loss = 0

    print("Current weight1", glob_weight1)
    print("Current weight2", glob_weight2)

    for data in tqdm(train_loader, desc="Training", unit="Event"):
        ei = knn_graph(data.pos, 6)
        ei.to(device)
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.pos, ei)
        loss = custom_loss(output, data.y, [glob_weight1, glob_weight2])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test(model, test_loader,  device):
    model.eval()
    total_loss = 0
    total_sig = 0
    total_bkg = 0
    bkgpred_norm = 0
    pred_sig = 0

    with torch.no_grad():
        for data in test_loader:
            ei = knn_graph(data.pos, 6)
            ei.to(device) 
            data = data.to(device)
            output = model(data.x, data.pos, ei)
            loss = custom_test_loss(output, data.y)
            total_loss += loss.item()

            for i in range(data.y.size(0)):
                label = data.y[i]
                pred = output[i]
                if label.sum() > 0:
                    total_sig+=1
                    if torch.argmax(pred) == torch.argmax(label):
                        pred_sig+=1
                if label.sum() == 0:
                    total_bkg+=1
                    bkgpred_norm+=torch.norm(pred, p=2)

                    

    accuracy = pred_sig/total_sig
    bkg_avgnorm = bkgpred_norm/total_bkg

    update_weight(bkg_avgnorm, accuracy)

    return total_loss / len(test_loader)

best_test_loss = float('inf')

for epoch in range(int(args.epochs)):

    train_loss = train(model, train_loader,  optimizer, device, epoch)
    test_loss = test(model, test_loader,  device)

    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    #scheduler.step()

    if(epoch> 0 and epoch%2==0):
        savepath = os.path.join("files_models", f"model_{args.modeltag}_e{epoch}.pth")
        torch.save(model.state_dict(), savepath)
        print(f"Model saved to {savepath}")


    if test_loss < best_test_loss:
        best_test_loss = test_loss
        save_path = os.path.join("files_models", f"model_{args.modeltag}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")




