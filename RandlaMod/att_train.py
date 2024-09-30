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

parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-ltr", "--load_train", default="", help="Load training data from a file")
parser.add_argument("-ltt", "--load_test", default="", help="Load testing data from a file")

args = parser.parse_args()

num_gvs = 6
trk_features = ['trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

if args.load_train != "":
    print(f"Loading training data from {args.load_train}...")
    with open(args.load_train, 'rb') as f:
        train_graphs = pickle.load(f)

if args.load_test != "":
    print(f"Loading testing data from {args.load_test}...")
    with open(args.load_test, 'rb') as f:
        test_graphs = pickle.load(f)

#test_graphs = test_graphs[:500]

train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)

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
prev_model_state = None
prev_optimizer_state = None
prev_weights = None
step_size1 = 50
step_size2 = 10


def update_weight(current_norm, current_acc):
    global glob_weight1, glob_weight2, best_acc, best_norm, step_size1, step_size2

    print(f"Curr Acc: {current_acc:.5f}")
    print(f"Best Acc: {best_acc:.5f}")
    print(f"Current norm: {current_norm:.5f}")
    print(f"Best norm: {best_norm:.5f}")

    #if(current_norm==0):
    #    glob_weight1 += step_size1*100
    #    glob_weight2 = max(2, glob_weight2 - step_size2*2)
    if(current_norm==1):
        glob_weight1 = max(50, glob_weight1 - step_size1)
        glob_weight2 += step_size2*5

    else:
        if current_acc > best_acc and current_norm > best_norm:
            # Accuracy improved, but background norm worsened
            glob_weight2 += step_size2  # Increase background weight to penalize more
        elif current_acc > best_acc and current_norm <= best_norm:
            # Accuracy improved and background norm improved
            glob_weight2 = max(2, glob_weight2 - step_size2)  # Decrease background weight slightly, with min bound
        elif current_acc <= best_acc and current_norm > best_norm:
            # Accuracy worsened and background norm worsened
            glob_weight1 += step_size1  # Increase signal weight
            glob_weight2 += step_size2  # Increase background weight
        elif current_acc <= best_acc and current_norm <= best_norm:
            # Accuracy worsened but background norm improved
            glob_weight1 += step_size1  # Increase signal weight
            glob_weight2 = max(2, glob_weight2 - step_size2)  # Decrease background weight, with min bound


    glob_weight1 = min(100000, glob_weight1)
    glob_weight2 = min(5000, glob_weight2)

    # Update best accuracy and norm
    if current_acc > best_acc:
        best_acc = current_acc
    if 0 < current_norm < best_norm:
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
    global glob_weight1, glob_weight2, prev_model_state, prev_optimizer_state, prev_weights, step_size2

    model.eval()
    total_loss = 0
    total_sig = 0
    total_bkg = 0
    bkgpred_norm = 0
    pred_sig = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", unit="Event"):
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

    if bkg_avgnorm == 0:
        print("Norm is zero, reverting to previous model state...")

        # Revert to previous model and optimizer state if saved
        if prev_model_state is not None and prev_optimizer_state is not None and prev_weights is not None:
            model.load_state_dict(prev_model_state)
            optimizer.load_state_dict(prev_optimizer_state)
            glob_weight1, glob_weight2 = prev_weights  # Revert the weights too

            print("Reverted model, optimizer, and weights.")
        else:
            print("No previous model state found to revert to.")
    else:
        prev_model_state = model.state_dict()  # Save the current model state
        prev_optimizer_state = optimizer.state_dict()  # Save the current optimizer state
        prev_weights = (glob_weight1, max(2, glob_weight2-3*step_size2))  # Save the current best weights

    return total_loss / len(test_loader)

best_test_loss = float('inf')

for epoch in range(int(args.epochs)):

    train_loss = train(model, train_loader,  optimizer, device, epoch)
    test_loss = test(model, test_loader,  device)

    print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    #scheduler.step()

    if(epoch> 0 and epoch%10==0):
        savepath = os.path.join("files_models", f"model_{args.modeltag}_e{epoch}.pth")
        torch.save(model.state_dict(), savepath)
        print(f"Model saved to {savepath}")


    if test_loss < best_test_loss:
        best_test_loss = test_loss
        save_path = os.path.join("files_models", f"model_{args.modeltag}_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

