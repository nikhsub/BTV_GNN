import warnings
warnings.filterwarnings("ignore")

import argparse
import uproot
import numpy as np
import torch
import pickle
from torch_geometric.data import Data
import random

parser = argparse.ArgumentParser("Creating labels and seeds")
parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-l", "--label", default="", help="Label file")
parser.add_argument("-st", "--save_tag", default="", help="Save tag for data")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=4000, help="Evt # to end with")
parser.add_argument("-t", "--train", default=False, action="store_true", help="Creating hadron level training data?")
args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 
                'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Load entire dataset for all features in one go
print("Loading files...")
with uproot.open(args.data) as f:
    demo = f['demo']
    datatree = demo['tree']
    
with uproot.open(args.label) as f:
    labeltree = f['tree']

# Preload arrays for all events for faster access in loops
trk_data = {feat: datatree[feat].array() for feat in trk_features}
sig_ind_array = labeltree['sig_ind'].array()
sig_flag_array = labeltree['sig_flag'].array()
bkg_flag_array = labeltree['bkg_flag'].array()
bkg_ind_array = labeltree['bkg_ind'].array()
seed_array = labeltree['seed_ind'].array()
SV_ind_array = labeltree['SVtrk_ind'].array()

def create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, 
                   seed_array, SV_ind_array, trk_features, nevts=3):

    evt_objects = []
    had_objects = []

    for evt in range(int(args.start), int(args.end)):
        print("Processing event", evt)
        evt_features = {f: trk_data[f][evt] for f in trk_features}
        seeds = seed_array[evt]

        if(not args.train):
            fullfeatmat = np.stack([evt_features[f] for f in trk_features], axis=1)
            nan_mask = ~np.isnan(fullfeatmat).any(axis=1)
            fullfeatmat = fullfeatmat[nan_mask]
            valid_indices = np.where(nan_mask)[0]

            evtsiginds = list(set(sig_ind_array[evt]))
            evtsigflags = [sig_flag_array[evt][sig_ind_array[evt].index(ind)] for ind in evtsiginds]
            evtbkginds = list(set(bkg_ind_array[evt]))
            evtbkginds = [ind for ind in evtbkginds if ind not in evtsiginds]
            evtsvinds = list(set(SV_ind_array[evt]))

            # Adjust valid indices for masking
            evtsiginds = [np.where(valid_indices == ind)[0][0] for ind in evtsiginds if ind in valid_indices]
            evtbkginds = [np.where(valid_indices == ind)[0][0] for ind in evtbkginds if ind in valid_indices]
            seeds      = [np.where(valid_indices == ind)[0][0] for ind in seeds if ind in valid_indices]
            evtsvinds  = [np.where(valid_indices == ind)[0][0] for ind in evtsvinds if ind in valid_indices]
            evtsigflags = [flag for ind, flag in zip(sig_ind_array[evt], evtsigflags) if ind in valid_indices]

            evt_data = Data(
                evt=evt,
                seeds=torch.tensor(seeds, dtype=torch.int16),
                x=torch.tensor(fullfeatmat, dtype=torch.float),
                siginds=torch.tensor(evtsiginds, dtype=torch.int16),
                sigflags=torch.tensor(evtsigflags, dtype=torch.int16),
                bkginds=torch.tensor(evtbkginds, dtype=torch.int16),
                svinds=torch.tensor(evtsvinds, dtype=torch.int16)
            )
            evt_objects.append(evt_data)

        if args.train:
            # Process hadrons within the event if in training mode
            for had in np.unique(sig_flag_array[evt]):
                sig_inds = sig_ind_array[evt][sig_flag_array[evt] == had]
                bkg_inds = list(set(bkg_ind_array[evt][bkg_flag_array[evt] == had]) - set(sig_inds))
                comb_inds = list(sig_inds) + bkg_inds
                feature_matrix = np.vstack([np.array([evt_features[f][int(ind)] for ind in comb_inds]) for f in trk_features]).T

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

                had_data = Data(
                    evt=evt,
                    had=had,
                    seeds=torch.tensor(sig_inds, dtype=torch.int16),
                    x=torch.tensor(feature_matrix, dtype=torch.float),
                    y=torch.tensor(labels, dtype=torch.float)
                )
                had_objects.append(had_data)

    return evt_objects, had_objects

print("Creating data objects...")
evt_data, had_data = create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array,
                                    seed_array, SV_ind_array, trk_features)

if not args.train:
    print(f"Saving evt_data to evtdata_{args.save_tag}.pkl...")
    with open("evtdata_" + args.save_tag + ".pkl", 'wb') as f:
        pickle.dump(evt_data, f)

if args.train:
    print(f"Saving had_data to haddata_{args.save_tag}.pkl...")
    with open("haddata_" + args.save_tag + ".pkl", 'wb') as f:
        pickle.dump(had_data, f)

