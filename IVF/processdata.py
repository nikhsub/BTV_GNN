import warnings
warnings.filterwarnings("ignore")

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
parser.add_argument("-st", "--save_tag", default="", help="Save tag for data")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=4000, help="Evt # to end with")

args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

datafile  = args.data
labelfile = args.label

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

trk_pt_array = datatree['trk_pt'].array()
trk_ip3d_array = datatree['trk_ip3d'].array()
trk_ip2dsig_array = datatree['trk_ip2dsig'].array()
sig_ind_array = labeltree['sig_ind'].array()
sig_flag_array = labeltree['sig_flag'].array()
bkg_flag_array = labeltree['bkg_flag'].array()
bkg_ind_array = labeltree['bkg_ind'].array()
seed_array = labeltree['seed_ind'].array()

def create_dataobj(datatree, labeltree, trk_features, nevts=3):

    evt_objects = []
    had_objects = []

    for evt in range(int(args.start), int(args.end)):
        #if(evt>nevts): break
        print("Processing event", evt)
        evt_features = {}
        seeds = seed_array[evt]

        for feature in trk_features:
            evt_features[feature] = (datatree[feature].array())[evt]

        fullfeatmat = np.stack([evt_features[f] for f in trk_features], axis=1)
        fullfeatmat = np.asarray(fullfeatmat)
        evtsiginds = sig_ind_array[evt]
        evtbkginds = bkg_ind_array[evt]
        evtbkginds = [ind for ind in evtbkginds if ind not in evtsiginds]

        nan_mask = ~np.isnan(fullfeatmat).any(axis=1)
        fullfeatmat = fullfeatmat[nan_mask]

        valid_indices = np.where(nan_mask)[0]

        evtsiginds = [np.where(valid_indices == ind)[0][0] for ind in evtsiginds if ind in valid_indices]
        evtbkginds = [np.where(valid_indices == ind)[0][0] for ind in evtbkginds if ind in valid_indices]
        seeds = [np.where(valid_indices == ind)[0][0] for ind in seeds if ind in valid_indices]

        evt_data = Data(
                evt=evt,
                seeds=torch.tensor(seeds, dtype=torch.int16),
                x=torch.tensor(fullfeatmat, dtype=torch.float),
                siginds=torch.tensor(evtsiginds, dtype=torch.int16),
                bkginds=torch.tensor(evtbkginds, dtype=torch.int16)
                )

        evt_objects.append(evt_data)

        for had in np.unique(sig_flag_array[evt]):

            siginds = np.where(sig_flag_array[evt] == had)[0]
            bkginds = np.where(bkg_flag_array[evt] == had)[0]

            sig_inds = sig_ind_array[evt][siginds]
            bkg_inds = list(set(bkg_ind_array[evt][bkginds]))
            bkg_inds = [ind for ind in bkg_inds if ind not in sig_inds]

            comb_inds = list(sig_inds) + bkg_inds

            feature_matrix = np.vstack([np.array([evt_features[feature][int(ind)] for ind in comb_inds])
                            for feature in trk_features]).T


            had_nan_mask = ~np.isnan(feature_matrix).any(axis=1)
            feature_matrix = feature_matrix[had_nan_mask]

            val_comb_inds = np.array(comb_inds)[had_nan_mask]

            sig_inds = [i for i, ind in enumerate(val_comb_inds) if ind in sig_inds]
            bkg_inds = [i for i, ind in enumerate(val_comb_inds) if ind in bkg_inds]

            labels = np.zeros(len(sig_inds) + len(bkg_inds))
            labels[:len(sig_inds)] = 1

            shuffled_inds = np.random.permutation(len(labels))

            feature_matrix = feature_matrix[shuffled_inds]
            labels = labels[shuffled_inds]
            sig_inds = [shuffled_inds.tolist().index(i) for i in sig_inds]
            bkg_inds = [shuffled_inds.tolist().index(i) for i in bkg_inds]

            #print("LABELS", labels)
            #print("SIGINDS", sig_inds)
            #print("BKGINDS", bkg_inds)
            #print("X shape", len(feature_matrix))
            #print("Y shape", len(labels))

            had_data = Data(
                    evt=evt,
                    had=had,
                    seeds = torch.tensor(sig_inds, dtype=torch.int16), #For training, seeds are sigs
                    x=torch.tensor(feature_matrix, dtype=torch.float),
                    y=torch.tensor(labels, dtype=torch.float)
                    )

            had_objects.append(had_data)

    return evt_objects, had_objects

        
print("Creating data objects...")

evt_data, had_data = create_dataobj(datatree, labeltree, trk_features)

print(f"Saving evt_data to evtdata_{args.save_tag}.pkl...")
with open("evtdata_"+args.save_tag+".pkl", 'wb') as f:
    pickle.dump(evt_data, f)

print(f"Saving had_data to haddata_{args.save_tag}.pkl...")
with open("haddata_"+args.save_tag+".pkl", 'wb') as f:
    pickle.dump(had_data, f)
