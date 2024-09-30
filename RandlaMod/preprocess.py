import argparse
import uproot
import numpy as np
import torch
import pickle
import torch_geometric
from torch_geometric.data import Data, DataLoader
import random

parser = argparse.ArgumentParser("Creating labels")

parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-l", "--label", default="", help="Label file")
parser.add_argument("-sg", "--save_graphs", default="", help="Save evt_graphs to a file")
parser.add_argument("-gv", "--maxgvs", default=6, help="Max # of GVs")
parser.add_argument("-nb", "--numbkg", default=100, help="Max # of bkg tracks")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=3000, help="Evt # to end with")
parser.add_argument("-t", "--test", default=False, action="store_true", help="Create test dataset")

args = parser.parse_args()

num_gvs = int(args.maxgvs)
trk_features = ['trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
pos_features = ['trk_eta', 'trk_phi']
truth_features = ['flag', 'ind']

datafile  = "files/"+args.data
labelfile = "files/"+args.label

datatree = None
labeltree = None

evt_graphs = None

print("Loading files...")
with uproot.open(datafile) as f:
    demo = f['demo']
    datatree = demo['tree']

with uproot.open(labelfile) as f:
    labeltree = f['tree']

if(int(args.end) > len(datatree['trk_p'].array())):
    print("Not enough events, check data file")


def create_dataobj(datatree, labeltree, trk_features, k=4, nevts=3):

    evt_object = []

    for evt in range(int(args.start), int(args.end)):
#        if(evt>nevts): break
        print("Processing event", evt)

        features = {}
        pos = {}
        labels = np.zeros((len((datatree['trk_p'].array())[evt]), num_gvs))

        for feature in trk_features:
            features[feature] = (datatree[feature].array())[evt]

        for feature in pos_features:
            pos[feature] = (datatree[feature].array())[evt]
        
        print("Creating labels")

        sig_inds = []
        bkg_inds = []

        for trk in range(len((datatree['trk_p'].array())[evt])):
            if trk in (labeltree['ind'].array())[evt]:
                ind = (labeltree['ind'].array())[evt].tolist().index(trk)
                flag = (labeltree['flag'].array())[evt][ind]
                if flag <= num_gvs - 1:
                    labels[trk, flag] = 1
                sig_inds.append(trk)
            else:
                bkg_inds.append(trk)

        feature_matrix = np.stack([features[f] for f in trk_features], axis=1)
        pos_matrix = np.stack([pos[f] for f in pos_features], axis=1)

        if(not args.test):
            print("Downsampling background...")
            
            if(len(bkg_inds) > int(args.numbkg)):
                bkg_inds = random.sample(bkg_inds, int(args.numbkg))

            sel_inds = sig_inds+bkg_inds

            feature_matrix = feature_matrix[sel_inds]
            labels = labels[sel_inds]
            pos_matrix = pos_matrix[sel_inds]

        print("Removing NaNs....")
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

print("Creating data objects...")

evt_graphs = create_dataobj(datatree, labeltree, trk_features)
print(f"Saving evt_graphs to {args.save_graphs}...")
with open(args.save_graphs, 'wb') as f:
    pickle.dump(evt_graphs, f)
