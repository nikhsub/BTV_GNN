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
parser.add_argument("-st", "--save_tag", default="", help="Save tag for data")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=3000, help="Evt # to end with")
parser.add_argument("-t", "--test", default=False, action="store_true", help="Creating test dataset?")
args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 
                'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Load entire dataset for all features in one go
print("Loading files...")
with uproot.open(args.data) as f:
    datatree = f['tree']

num_evts = datatree.num_entries
print("NUMEVTS", num_evts)

# Preload arrays for all events for faster access in loops
trk_data = {feat: datatree[feat].array() for feat in trk_features}
sig_ind_array = datatree['sig_ind'].array()
sig_flag_array = datatree['sig_flag'].array()
bkg_flag_array = datatree['bkg_flag'].array()
bkg_ind_array = datatree['bkg_ind'].array()
seed_array = datatree['seed_ind'].array()
SV_ind_array = datatree['SVtrk_ind'].array()
had_pt_array = datatree['had_pt'].array()

def create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, 
                   seed_array, SV_ind_array, had_pt_array, trk_features, nevts=3):

    evt_objects = []

    if (int(args.end) == -1): end = num_evts
    else: end = int(args.end)

    for evt in range(int(args.start), end):
        print(evt)
        evt_features = {f: trk_data[f][evt] for f in trk_features}
        #seeds = seed_array[evt]

        fullfeatmat = np.stack([evt_features[f] for f in trk_features], axis=1)
        fullfeatmat = np.array(fullfeatmat, dtype=np.float32)
        nan_mask = ~np.isnan(fullfeatmat).any(axis=1)
        fullfeatmat = fullfeatmat[nan_mask]
        valid_indices = np.where(nan_mask)[0]

        evtsiginds = list(set(sig_ind_array[evt]))
        evtsigflags = [sig_flag_array[evt][np.where(sig_ind_array[evt] == ind)[0][0]] for ind in evtsiginds]
        evtbkginds = list(set(bkg_ind_array[evt]))
        evtbkginds = [ind for ind in evtbkginds if ind not in evtsiginds]
        evtsvinds = list(set(SV_ind_array[evt]))

        # Adjust valid indices for masking
        evtsiginds = [np.where(valid_indices == ind)[0][0] for ind in evtsiginds if ind in valid_indices]

        if(not args.test):
            if(len(evtsiginds) < 3) : continue

        evtbkginds = [np.where(valid_indices == ind)[0][0] for ind in evtbkginds if ind in valid_indices]
        #seeds      = [np.where(valid_indices == ind)[0][0] for ind in seeds if ind in valid_indices]
        evtsvinds  = [np.where(valid_indices == ind)[0][0] for ind in evtsvinds if ind in valid_indices]
        evtsigflags = [flag for ind, flag in zip(sig_ind_array[evt], evtsigflags) if ind in valid_indices]

        evt_data = Data(
            evt=evt,
            #seeds=torch.tensor(seeds, dtype=torch.int16),
            x=torch.tensor(fullfeatmat, dtype=torch.float),
            siginds=torch.tensor(evtsiginds, dtype=torch.int16),
            sigflags=torch.tensor(evtsigflags, dtype=torch.int16),
            bkginds=torch.tensor(evtbkginds, dtype=torch.int16),
            svinds=torch.tensor(evtsvinds, dtype=torch.int16)
        )
        evt_objects.append(evt_data)

    return evt_objects

print("Creating data objects...")
evt_data = create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array,
                                    seed_array, SV_ind_array, had_pt_array, trk_features)

print(f"Saving evt_data to evtvaldata_{args.save_tag}.pkl...")
with open("evtvaldata_" + args.save_tag + ".pkl", 'wb') as f:
    pickle.dump(evt_data, f)
