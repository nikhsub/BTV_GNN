import warnings
warnings.filterwarnings("ignore")

import json
import argparse
import uproot
import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
import pickle5 as pickle
from tqdm import tqdm
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import os
import torch.nn.functional as F
import math
from GATModel import *
import joblib
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import roc_curve, precision_recall_curve, auc

parser = argparse.ArgumentParser("GNN training")

parser.add_argument("-e", "--epochs", default=20, help="Number of epochs")
parser.add_argument("-mt", "--modeltag", default="test", help="Tag to add to saved model name")
parser.add_argument("-lh", "--load_had", default="", help="Path to training files")
parser.add_argument("-le", "--load_evt", default="", help="Absolute path to event level validation file")

args = parser.parse_args()
glob_test_thres = 0.5

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
edge_features = ['dca', 'deltaR', 'rel_ip2d', 'rel_ip3d']

batchsize = 700

#LOADING DATA
train_hads = []
val_evts = []
if args.load_had != "":
    if os.path.isdir(args.load_had):
        print(f"Loading training data from {args.load_had}...")
        pkl_files = [os.path.join(args.load_had, f) for f in os.listdir(args.load_had) if f.endswith('.pkl')]
    for pkl_file in tqdm(pkl_files, desc="Loading .pkl files", unit="file"):
        with open(pkl_file, 'rb') as f:
            train_hads.extend(pickle.load(f))


if args.load_evt != "":
    print(f"Loading event level data from {args.load_evt}...")
    with open(args.load_evt, 'rb') as f:
        val_evts = pickle.load(f)

#train_hads = train_hads[:] #Control number of input samples here - see array splicing for more
val_evts   = val_evts[0:1500]

train_len = int(0.8 * len(train_hads))
train_data, test_data = random_split(train_hads, [train_len, len(train_hads) - train_len])

train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=True, num_workers=16)
test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=True, num_workers=8)

#DEVICE AND MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GNNModel(indim=len(trk_features), outdim=32, edge_dim=len(edge_features), heads=4, dropout=0.4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #Was 0.00005
#scheduler = StepLR(optimizer, step_size = 20, gamma=0.95)

def class_weighted_bce(preds, labels, pos_weight=5.0, neg_weight=1.0):
    """Class-weighted binary cross-entropy loss"""
    pos_weight = torch.tensor(pos_weight, device=preds.device)
    neg_weight = torch.tensor(neg_weight, device=preds.device)
    weights = torch.where(labels == 1, pos_weight, neg_weight)
    bce_loss = F.binary_cross_entropy(preds, labels, weight=weights)
    return bce_loss

def focal_loss(preds, labels, gamma=2.0, alpha=0.80):
    """Focal loss to emphasize hard-to-classify samples"""
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    bce_loss = loss_fn(preds, labels)
    pt = torch.exp(-bce_loss)  # Probability of correct classification
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * bce_loss
    return loss.mean()

def target_eff_loss(preds, labels, target_tpr=0.7, scale=0.1):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)  # Shape [N]
    labels = labels.view(-1)  # Shape [N]

    sorted_indices = torch.argsort(preds, descending=True)
    sorted_preds = preds[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Cumulative sum for signal and background counts
    cumsum_labels = torch.cumsum(sorted_labels, dim=0)
    total_signal = cumsum_labels[-1]
    total_background = len(labels) - total_signal

    # Find threshold index for target TPR
    target_index = torch.argmax((cumsum_labels >= target_tpr * total_signal).to(torch.int64))
    threshold = sorted_preds[target_index]

    # Calculate FPR and Precision at this threshold
    tp = cumsum_labels[target_index]
    fp = (target_index + 1) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fpr = fp / total_background if total_background > 0 else 0
    bg_rejection = 1 - fpr

    loss = -scale*(precision+0.1*bg_rejection)
    return loss


def contrastive_loss(x1, x2, temperature=0.5):
    """Compute the contrastive loss"""
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    batch_size = x1.size(0)

    similarity_matrix = torch.mm(x1, x2.t()) / temperature

    labels = torch.arange(batch_size).to(x1.device)
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss

def shuffle_edge_index(edge_index):
    num_edges = edge_index.size(1)

    # Shuffle the target nodes (row 1) while keeping the sources (row 0) the same
    shuffled_targets = edge_index[1, torch.randperm(num_edges, device=edge_index.device)]

    # Combine the original sources with the shuffled targets
    new_edge_index = torch.stack([edge_index[0], shuffled_targets], dim=0)
    bidirectional_edges = torch.cat([new_edges, new_edges.flip(0)], dim=1)

    return bidirectional_edges

def compute_class_weights(labels):
    """Dynamically compute class weights based on dataset distribution."""
    num_pos = labels.sum().item()
    num_neg = len(labels) - num_pos
    pos_weight = num_neg / (num_pos + 1e-8)  # Avoid divide-by-zero
    return pos_weight

#TRAIN
def train(model, train_loader, optimizer, device, epoch, bce_loss=True):
    model.to(device)
    model.train()
    total_loss=0
    total_node_loss = 0
    total_cont_loss = 0
    total_eff_loss  = 0

    for data in tqdm(train_loader, desc="Training", unit="Batch"):
        data= data.to(device)

        optimizer.zero_grad()
        num_nodes = data.x.size(0)
        #batch_had_weight = 1
        
        #edge_index = knn_graph(data.x, k=4, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
        
        node_embeds1, preds1, _,_,_,_,_ = model(data.x, data.edge_index, data.edge_attr)
        weight = torch.tensor(compute_class_weights(data.y.float().unsqueeze(1)), dtype=torch.float, device=device)//2
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
        node_loss = loss_fn(preds1, data.y.float().unsqueeze(1))
        
        #node_loss = class_weighted_bce(preds1, data.y.float().unsqueeze(1), pos_weight=weight, neg_weight=1)*batch_had_weight
        #node_loss = focal_loss(preds1, data.y.float().unsqueeze(1))
        #eff_loss = target_eff_loss(preds1, data.y.float().unsqueeze(1), target_tpr=0.70, scale=0.1)
        eff_loss = torch.tensor(0.0, device=device)

        if bce_loss:
            cont_loss = torch.tensor(0.0, device=device)
        else:
            edge_index2 = shuffle_edge_index(data.edge_index).to(device)
            node_embeds2, logits2, _,_,_,_,_ = model(data.x, edge_index2)
            cont_loss = contrastive_loss(node_embeds1, node_embeds2)

        #batch_had_weight = data.had_weight[data.batch].mean().to(device)

        loss = (cont_loss * 0.05) + node_loss + eff_loss

        loss.backward()
        optimizer.step()
        #scheduler.step()

        total_loss      += loss.item()
        total_node_loss += node_loss.item()
        total_cont_loss += cont_loss.item()
        total_eff_loss  += eff_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_node_loss = total_node_loss / len(train_loader)
    avg_cont_loss = total_cont_loss / len(train_loader)
    avg_eff_loss  = total_eff_loss / len(train_loader)

    #print(f"No seed hadrons: {nosigseeds}")
    return avg_loss, avg_node_loss, avg_cont_loss, avg_eff_loss

#TEST
def test(model, test_loader, device, epoch, k=11, thres=0.5):
    model.to(device)
    model.eval()

    correct_bkg = 0
    total_bkg = 0
    correct_signal = 0
    total_signal = 0
    total_loss = 0
    mean_sig_prob = []
    mean_bkg_prob = []
    all_preds = []
    all_labels = []
    #nosigtest = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="Batch"):
            batch = batch.to(device)
            num_nodes = batch.x.size(0)
            
            #batch.x = scale_features(batch.x, scale_mean, scale_std)
            
            #edge_index = knn_graph(batch.x, k=k, batch=batch.batch, loop=False, cosine=False, flow="source_to_target").to(device)

            _, logits, _,_,_,_,_ = model(batch.x, batch.edge_index, batch.edge_attr)
            preds = torch.sigmoid(logits)

            all_preds.extend(preds.squeeze().cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

            batch_loss = F.binary_cross_entropy(preds, batch.y.float().unsqueeze(1))
            total_loss += batch_loss.item()

            signal_mask = (batch.y == 1)  # Mask for signal nodes
            background_mask = (batch.y == 0)

            #mean_sig_prob.append(preds[signal_mask].mean().item())
            #mean_bkg_prob.append(preds[background_mask].mean().item())

            preds = (preds > thres).float().squeeze()

            # Signal-specific accuracy
            correct_signal += (preds[signal_mask] == batch.y[signal_mask].float()).sum().item()
            total_signal += signal_mask.sum().item()

            correct_bkg += (preds[background_mask] == batch.y[background_mask].float()).sum().item()
            total_bkg += background_mask.sum().item()

            #print(f"Predictions: {preds}")
            #print(f"True labels: {batch.y}")
            #print(f"Signal Mask: {signal_mask}")
            #print(f"Correct Signal: {correct_signal}, Total Signal: {total_signal}")
            #print(f"Correct total: {correct}, Total: {total}")

    bkg_accuracy = correct_bkg / total_bkg if total_bkg > 0 else 0
    sig_accuracy = correct_signal / total_signal if total_signal > 0 else 0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    auc = roc_auc_score(all_labels, all_preds)
    #print(f"No sig hadrons test sample: {nosigtest}")
    #print("Mean signal probs per had", mean_sig_prob)
    #print("Mean background probs per had", mean_bkg_prob)

    return bkg_accuracy, sig_accuracy, avg_loss, auc

def validate(model, val_graphs, device, epoch, k=6, target_sigeff=0.70):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    num_samps = 0

    for i, data in enumerate(val_graphs):
        with torch.no_grad():
            data = data.to(device)
            #edge_index = knn_graph(data.x, k=k, batch=None, loop=False, cosine=False, flow="source_to_target").to(device)
            _, logits, _,_,_,_,_ = model(data.x, data.edge_index, data.edge_attr)
            preds = torch.sigmoid(logits)
            preds = preds.squeeze()
            #labels = data.y.float()
            siginds = data.siginds.cpu().numpy()
            #labels = np.zeros(len(preds))
            labels = torch.zeros(len(preds), device=device)
            labels[siginds] = 1  # Set signal indices to 1
            evt_loss = F.binary_cross_entropy(preds, labels)
            total_loss += evt_loss.item()
            num_samps +=1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(all_labels, all_preds)
    pr_auc = auc(recall, precision)

    target_idx = np.argmin(np.abs(tpr - target_sigeff))
    threshold_at_target_eff = thresholds_roc[target_idx]

    precision_at_sigeff = precision[np.argmin(np.abs(recall - tpr[target_idx]))]
    bg_rejection_at_sigeff = 1 - fpr[target_idx]  # 1 - FPR

    avg_loss = total_loss / num_samps if num_samps > 0 else 0

    return roc_auc, pr_auc, avg_loss, precision_at_sigeff, bg_rejection_at_sigeff

best_metric = 1e6
patience = 10
no_improve = 0
val_every = 10

rolling_bce_loss = []
stabilization_epochs = 5  # Number of epochs to track for stabilization
stabilization_tolerance = 0.000  # Threshold for detecting stabilization
using_bce_loss = True  # Flag to indicate current loss type

stats = {
    "epochs": [],
    "best_epoch": {
        "epoch": None,
        "total_loss": None,
        "node_loss": None,
        "cont_loss": None,
        "eff_loss": None,
        "bkg_acc": None,
        "sig_acc": None,
        "test_auc": None,
        "test_loss": None,
        "total_acc": None,
        "val_auc": None,
        "val_loss": None,
        "pr_auc": None,
        "precision": None,
        "bg_rejection": None,
        "metric": None
    }
}

for epoch in range(int(args.epochs)):
    if using_bce_loss:
        tot_loss, node_loss, cont_loss, eff_loss = train(model, train_loader,  optimizer, device, epoch, bce_loss=True)
        rolling_bce_loss.append(node_loss)
        if len(rolling_bce_loss) > stabilization_epochs:
            rolling_bce_loss.pop(0)
        if len(rolling_bce_loss) == stabilization_epochs:
            loss_change = max(rolling_bce_loss) - min(rolling_bce_loss)
            if loss_change < stabilization_tolerance:
                print(f"Switching to contrastive loss at epoch {epoch + 1}.")
                using_bce_loss = False
    else:
        tot_loss, node_loss, cont_loss, eff_loss = train(model, train_loader,  optimizer, device, epoch, bce_loss=False)


    bkg_acc, sig_acc, test_loss, test_auc = test(model, test_loader,  device, epoch, k=12, thres=glob_test_thres)
    sum_acc = bkg_acc + sig_acc

    val_auc = -1
    val_loss = 1e6
    pr_auc = -1
    prec = -1
    bg_rej = -1
    metric = -1

    if (epoch+1) % val_every == 0:
        print(f"Validating at epoch {epoch}...")
        val_auc, pr_auc, val_loss, prec, bg_rej = validate(model, val_evts, device, epoch, k=12, target_sigeff=0.70)
        print(f"Val AUC: {val_auc:.4f}, Val Loss: {val_loss:.4f}") 

        metric = val_loss

        if metric < best_metric:
            best_metric = metric
            no_improve = 0

            stats["best_epoch"] = {
                "epoch": epoch + 1,
                "total_loss": tot_loss,
                "node_loss": node_loss,
                "cont_loss": cont_loss,
                "eff_loss": eff_loss,
                "bkg_acc": bkg_acc,
                "sig_acc": sig_acc,
                "test_auc": test_auc,
                "test_loss": test_loss,
                "total_acc": sum_acc,
                "val_auc": val_auc,
                "val_loss": val_loss,
                "pr_auc": pr_auc,
                "precision": prec,
                "bg_rejection": bg_rej,
                "metric": metric
            }

            save_path = os.path.join("model_files", f"model_{args.modeltag}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

        else:
            no_improve += 1
            print(f"Validation metric did not improve after {val_every} epochs")

        if no_improve >= patience:
            print(f"Early stopping triggered after {patience*val_every} epochs of no improvement.")
            break

    epoch_stats = {
        "epoch": epoch + 1,
        "total_loss": tot_loss,
        "node_loss": node_loss,
        "cont_loss": cont_loss,
        "eff_loss": eff_loss,
        "bkg_acc": bkg_acc,
        "sig_acc": sig_acc,
        "test_auc": test_auc,
        "test_loss": test_loss,
        "total_acc": sum_acc,
        "val_auc": val_auc,
        "val_loss": val_loss,
        "pr_auc": pr_auc,
        "precision": prec,
        "bg_rejection": bg_rej,
        "metric": metric
    }
    stats["epochs"].append(epoch_stats)
    
    print(f"Epoch {epoch+1}/{args.epochs}, Total Loss: {tot_loss:.4f}, Node Loss: {node_loss:.4f}, Eff Loss: {eff_loss:.4f}")
    print(f"Bkg Acc: {bkg_acc*100:.4f}%, Sig Acc: {sig_acc*100:.4f}%, Test Loss: {test_loss:.4f}")


    if(epoch> 0 and epoch%10==0):
        savepath = os.path.join("model_files", f"model_{args.modeltag}_e{epoch}.pth")
        torch.save(model.state_dict(), savepath)
        print(f"Model saved to {savepath}")

    
    with open("training_stats_"+args.modeltag+".json", "w") as f:
        json.dump(stats, f, indent=4)

