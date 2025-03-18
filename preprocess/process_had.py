import warnings
warnings.filterwarnings("ignore")

import argparse
import uproot
import numpy as np
import torch
import pickle
from torch_geometric.data import Data
import random
from itertools import combinations

parser = argparse.ArgumentParser("Creating labels and seeds")
parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-st", "--save_tag", default="", help="Save tag for data")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=4000, help="Evt # to end with")
args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt', 
                'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Load entire dataset for all features in one go
print("Loading files...")
with uproot.open(args.data) as f:
    datatree = f['tree']

num_evts = datatree.num_entries

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

def create_edge_index(trk_i, trk_j, dca, deltaR, rel_ip2d, rel_ip3d, val_comb_inds):
    """
    Create edge index and edge features for tracks in val_comb_inds.
    """
    
    #Filter
    valid_edge_mask = np.isin(trk_i, val_comb_inds) & np.isin(trk_j, val_comb_inds)

    trk_i_np = trk_i.to_numpy()
    trk_j_np = trk_j.to_numpy()
    dca_np = dca.to_numpy()
    deltaR_np = deltaR.to_numpy()
    rel_ip2d_np = rel_ip2d.to_numpy()
    rel_ip3d_np = rel_ip3d.to_numpy()
    
    
    # Extract valid edges and their features
    edge_index = np.vstack([trk_i_np[valid_edge_mask], trk_j_np[valid_edge_mask]]).astype(np.int64)
    edge_features = np.vstack([dca_np[valid_edge_mask], deltaR_np[valid_edge_mask], 
                              rel_ip2d_np[valid_edge_mask], rel_ip3d_np[valid_edge_mask]]).T
    
    return edge_index, edge_features


def create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, 
                   SV_ind_array, had_pt_array, trk_1_array, trk_2_array, deltaR_array,
                   dca_array, rel_ip2d_array, rel_ip3d_array, trk_features, nevts=3):

    had_objects = []

    if (int(args.end) == -1): end = num_evts
    else: end = int(args.end)

    for evt in range(int(args.start), end):
        print(evt)
        evt_features = {f: trk_data[f][evt] for f in trk_features}

        #bins = [10, 20, 30, 40, 50]
        #had_weights = [4, 3, 2, 1] #10to20, 20to30, 30to40, 40to50

        for had in np.unique(sig_flag_array[evt]):
            sig_inds = sig_ind_array[evt][sig_flag_array[evt] == had]
            bkg_inds = list(set(bkg_ind_array[evt][bkg_flag_array[evt] == had]) - set(sig_inds))
            comb_inds = list(sig_inds) + bkg_inds
            feature_matrix = np.vstack([np.array([evt_features[f][int(ind)] for ind in comb_inds]) for f in trk_features]).T

            #hadron_pt = had_pt_array[evt][had]

            had_nan_mask = ~np.isnan(feature_matrix).any(axis=1)
            feature_matrix = feature_matrix[had_nan_mask]
            val_comb_inds = np.array(comb_inds)[had_nan_mask]

            val_comb_inds_map = {ind: i for i, ind in enumerate(val_comb_inds)}
            
            sig_inds = [val_comb_inds_map[ind] for ind in sig_inds if ind in val_comb_inds_map]
            if(len(sig_inds) < 3): continue

            bkg_inds = [val_comb_inds_map[ind] for ind in bkg_inds if ind in val_comb_inds_map] #REMAP 1
            
            labels = np.zeros(len(val_comb_inds))
            labels[:len(sig_inds)] = 1 #Sig_inds in the front

            shuffled_inds = np.random.permutation(len(labels))
            feature_matrix = feature_matrix[shuffled_inds]
            labels = labels[shuffled_inds]

            shuffled_map = {i: shuffled_inds[i] for i in range(len(shuffled_inds))}

            sig_inds = [shuffled_map[i] for i in sig_inds]
            bkg_inds = [shuffled_map[i] for i in bkg_inds] #REMAP 2

            trk_i = trk_1_array[evt]
            trk_j = trk_2_array[evt]
            dca = dca_array[evt]
            deltaR = deltaR_array[evt]
            rel_ip2d = rel_ip2d_array[evt]
            rel_ip3d = rel_ip3d_array[evt]

            edge_index, edge_features = create_edge_index(trk_i, trk_j, dca, deltaR, rel_ip2d, rel_ip3d, val_comb_inds)
            if edge_index.shape[1] == 0:
                continue
            #REMAP 1
            #edge_index = np.array([[val_comb_inds_map[edge_index[0, i]], val_comb_inds_map[edge_index[1, i]]] for i in range(edge_index.shape[1])]).T
            ##REMAP 2
            #edge_index = np.array([[shuffled_map[edge_index[0, i]], shuffled_map[edge_index[1, i]]] for i in range(edge_index.shape[1])]).T

            # First remap using val_comb_inds_map
            edge_index = np.vectorize(val_comb_inds_map.get)(edge_index)
            
            # Second remap using shuffled_map
            edge_index = np.vectorize(shuffled_map.get)(edge_index)
            

            hadron_weight = 1  # Default weight

            #for i, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
             #   if lower <= hadron_pt < upper:
             #       hadron_weight = had_weights[i]
              #      break

            had_data = Data(
                x=torch.tensor(feature_matrix, dtype=torch.float),
                y=torch.tensor(labels, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.int64),
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                had_weight=torch.tensor([hadron_weight], dtype=torch.float)
            )
            had_objects.append(had_data)

    return had_objects

print("Creating hadron training objects...")
had_data = create_dataobj(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array,
                                    SV_ind_array, had_pt_array, trk_1_array, trk_2_array, deltaR_array,
                                    dca_array, rel_ip2d_array, rel_ip3d_array, trk_features)
print(f"Saving had_data to haddata_{args.save_tag}.pkl...")
with open("haddata_" + args.save_tag + ".pkl", 'wb') as f:
    pickle.dump(had_data, f)

