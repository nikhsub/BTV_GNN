import argparse
import uproot
import numpy as np
import torch
import pickle
import torch_geometric
from torch_geometric.data import Data, DataLoader
import random

parser = argparse.ArgumentParser("Creating labels and seeds")

parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-l", "--label", default="", help="Label file")
parser.add_argument("-sg", "--save_graphs", default="", help="Save evt_graphs to a file")
parser.add_argument("-gv", "--maxgvs", default=10, help="Max # of GVs")
parser.add_argument("-nb", "--numbkg", default=-1, help="Max # of bkg tracks")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=4000, help="Evt # to end with")

args = parser.parse_args()

num_gvs = int(args.maxgvs)
trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']
knn_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d']
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
        kfeatures = {}
        labels = np.zeros((len((datatree['trk_p'].array())[evt]), num_gvs))
        seeds  = np.zeros((len((datatree['trk_p'].array())[evt]), 1))

        for feature in trk_features:
            features[feature] = (datatree[feature].array())[evt]

        for feature in knn_features:
            kfeatures[feature] = (datatree[feature].array())[evt]

        print("Creating labels")

        sig_inds = []
        bkg_inds = []

        trk_pt_array = datatree['trk_pt'].array()
        trk_ip3d_array = datatree['trk_ip3d'].array()
        trk_ip2dsig_array = datatree['trk_ip2dsig'].array()
        ind_array = labeltree['ind'].array()
        flag_array = labeltree['flag'].array()

        for trk in range(len((datatree['trk_p'].array())[evt])):
            if( trk_pt_array[evt][trk] > 0.8 and abs(trk_ip3d_array[evt][trk]) > 0.005 and abs(trk_ip2dsig_array[evt][trk]) > 1.2):
                seeds[trk] = 1

            if trk in ind_array[evt]:
                ind = ind_array[evt].tolist().index(trk)
                flag = flag_array[evt][ind]
                if flag <= num_gvs - 1:
                    labels[trk, flag] = 1
                sig_inds.append(trk)
            else:
                bkg_inds.append(trk)

        feature_matrix = np.stack([features[f] for f in trk_features], axis=1)
        kfeature_matrix = np.stack([kfeatures[f] for f in knn_features], axis=1)

        if(int(args.numbkg)>0):    
            if(len(bkg_inds) > int(args.numbkg)):
                bkg_inds = random.sample(bkg_inds, int(args.numbkg))

        sel_inds = sig_inds+bkg_inds

        feature_matrix = feature_matrix[sel_inds]
        kfeature_matrix = kfeature_matrix[sel_inds]
        labels = labels[sel_inds]
        seeds = seeds[sel_inds]

        print("Removing NaNs....")
        feature_matrix = np.asarray(feature_matrix)
        nan_mask = ~np.isnan(feature_matrix).any(axis=1)
        shape_i = feature_matrix.shape[0]
        feature_matrix = feature_matrix[nan_mask]
        kfeature_matrix = kfeature_matrix[nan_mask]
        labels = labels[nan_mask]
        seeds = seeds[nan_mask]
        shape_f = feature_matrix.shape[0]
        print("NaN rows dropped:", shape_i-shape_f)

        print(f"Creating data for {shape_f} tracks")
        data = Data(
            x=torch.tensor(feature_matrix, dtype=torch.float),
            knn_x = torch.tensor(kfeature_matrix, dtype=torch.float),
            seeds = torch.tensor(seeds, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.float)
        )
        evt_object.append(data)

    return evt_object

print("Creating data objects...")

evt_graphs = create_dataobj(datatree, labeltree, trk_features)
print(f"Saving evt_graphs to {args.save_graphs}...")
with open(args.save_graphs, 'wb') as f:
    pickle.dump(evt_graphs, f)
