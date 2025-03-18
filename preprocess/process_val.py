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
#seed_array = datatree['seed_ind'].array()
SV_ind_array = datatree['SVtrk_ind'].array()
had_pt_array = datatree['had_pt'].array()
trk_1_array  = datatree['trk_1'].array()
trk_2_array  = datatree['trk_2'].array()
deltaR_array  = datatree['deltaR'].array()
dca_array  = datatree['dca'].array()
rel_ip2d_array  = datatree['rel_ip2d'].array()
rel_ip3d_array  = datatree['rel_ip3d'].array()

def create_edge_index(trk_i, trk_j, dca, deltaR, rel_ip2d, rel_ip3d, val_comb_inds, dca_threshold=0.05):
    """
    Create edge index and edge features for tracks in val_comb_inds using DCA-based clustering.
    
    Edges are only formed if the DCA between two tracks is below a given threshold.
    """

    # Convert to numpy for fast indexing
    trk_i_np = trk_i.to_numpy()
    trk_j_np = trk_j.to_numpy()
    dca_np = dca.to_numpy()
    deltaR_np = deltaR.to_numpy()
    rel_ip2d_np = rel_ip2d.to_numpy()
    rel_ip3d_np = rel_ip3d.to_numpy()

    # Filter based on val_comb_inds (valid track indices)
    valid_edge_mask = np.isin(trk_i_np, val_comb_inds) & np.isin(trk_j_np, val_comb_inds)

    # Apply DCA threshold for clustering
    dca_mask = dca_np < dca_threshold  # Only keep edges with small DCA

    # Combine both masks
    final_mask = valid_edge_mask & dca_mask

    # Extract valid edges and their features
    edge_index = np.vstack([trk_i_np[final_mask], trk_j_np[final_mask]]).astype(np.int64)
    edge_features = np.vstack([dca_np[final_mask], deltaR_np[final_mask],
                               rel_ip2d_np[final_mask], rel_ip3d_np[final_mask]]).T

    return edge_index, edge_features

def get_dca_thres(trk_i, trk_j, dca, evtsiginds):
        sig_mask = np.isin(trk_i, evtsiginds) & np.isin(trk_j, evtsiginds)

        sig_dca_values = dca[sig_mask]

        if len(sig_dca_values) > 0:
            avg_sig_dca = np.mean(sig_dca_values)
        else:
            avg_sig_dca = 0.01  # No signal track pairs found, default val
        
        return avg_sig_dca

def create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, 
                   SV_ind_array, had_pt_array, trk_1_array, trk_2_array, deltaR_array,
                   dca_array, rel_ip2d_array, rel_ip3d_array, trk_features, nevts=3):

    evt_objects = []

    if (int(args.end) == -1): end = num_evts
    else: end = int(args.end)

    for evt in range(int(args.start), end):
        print(evt)
        evt_features = {f: trk_data[f][evt] for f in trk_features}
        fullfeatmat = np.stack([evt_features[f] for f in trk_features], axis=1)
        fullfeatmat = np.array(fullfeatmat, dtype=np.float32)
    
        nan_mask = ~np.isnan(fullfeatmat).any(axis=1)
        fullfeatmat = fullfeatmat[nan_mask]
        valid_indices = np.where(nan_mask)[0]
        
        val_inds_map = {ind: i for i, ind in enumerate(valid_indices)}

        evtsiginds = list(set(sig_ind_array[evt]))

        trk_i = trk_1_array[evt]
        trk_j = trk_2_array[evt]
        dca = dca_array[evt]
        deltaR = deltaR_array[evt]
        rel_ip2d = rel_ip2d_array[evt]
        rel_ip3d = rel_ip3d_array[evt]
        
        if(not args.test):
            dca_thres = get_dca_thres(trk_i, trk_j, dca, evtsiginds)
        else:
            dca_thres = 0.01*3 #For test events
            
        evtsigflags = [sig_flag_array[evt][np.where(sig_ind_array[evt] == ind)[0][0]] for ind in evtsiginds]
        evtbkginds = list(set(bkg_ind_array[evt]))
        evtbkginds = [ind for ind in evtbkginds if ind not in evtsiginds]
        evtsvinds = list(set(SV_ind_array[evt]))

        # Adjust valid indices for masking
        evtsiginds = [val_inds_map[ind] for ind in evtsiginds if ind in val_inds_map]

        if(not args.test):
            if(len(evtsiginds) < 3) : continue

        evtbkginds = [val_inds_map[ind] for ind in evtbkginds if ind in val_inds_map]
        evtsvinds  = [val_inds_map[ind] for ind in evtsvinds if ind in val_inds_map]
        evtsigflags = [flag for ind, flag in zip(sig_ind_array[evt], evtsigflags) if ind in valid_indices]
        
        trk_i = trk_1_array[evt]
        trk_j = trk_2_array[evt]
        dca = dca_array[evt]
        deltaR = deltaR_array[evt]
        rel_ip2d = rel_ip2d_array[evt]
        rel_ip3d = rel_ip3d_array[evt]
    
        
        edge_index, edge_features = create_edge_index(trk_i, trk_j, dca, deltaR, rel_ip2d, rel_ip3d, valid_indices, dca_threshold=dca_thres)
        if edge_index.shape[1] == 0:
            continue
        
        edge_index = np.vectorize(val_inds_map.get)(edge_index)


        evt_data = Data(
            evt=evt,
            #seeds=torch.tensor(seeds, dtype=torch.int16),
            x=torch.tensor(fullfeatmat, dtype=torch.float),
            siginds=torch.tensor(evtsiginds, dtype=torch.int16),
            sigflags=torch.tensor(evtsigflags, dtype=torch.int16),
            bkginds=torch.tensor(evtbkginds, dtype=torch.int16),
            svinds=torch.tensor(evtsvinds, dtype=torch.int16),
            edge_index=torch.tensor(edge_index, dtype=torch.int64),
            edge_attr=torch.tensor(edge_features, dtype=torch.float) 
        )
        evt_objects.append(evt_data)

    return evt_objects

print("Creating data objects...")
evt_data = create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array,
                                    SV_ind_array, had_pt_array, trk_1_array, trk_2_array, deltaR_array,
                                    dca_array, rel_ip2d_array, rel_ip3d_array, trk_features)

print(f"Saving evt_data to evtvaldata_{args.save_tag}.pkl...")
with open("evtvaldata_" + args.save_tag + ".pkl", 'wb') as f:
    pickle.dump(evt_data, f)
