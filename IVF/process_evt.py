import warnings
warnings.filterwarnings("ignore")

import argparse
import uproot
import numpy as np
import torch
import pickle
from torch_geometric.data import Data

parser = argparse.ArgumentParser("Creating event-level training samples")
parser.add_argument("-d", "--data", default="", help="Data file")
parser.add_argument("-st", "--save_tag", default="", help="Save tag for data")
parser.add_argument("-s", "--start", default=0, help="Evt # to start from")
parser.add_argument("-e", "--end", default=4000, help="Evt # to end with")
args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt',
                'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Load data
print("Loading files...")
with uproot.open(args.data) as f:
    datatree = f['tree']

num_evts = datatree.num_entries

trk_data = {feat: datatree[feat].array() for feat in trk_features}
sig_ind_array = datatree['sig_ind'].array()
sig_flag_array = datatree['sig_flag'].array()
bkg_flag_array = datatree['bkg_flag'].array()
bkg_ind_array = datatree['bkg_ind'].array()

def create_event_graphs(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, trk_features, nevts=3):
    evt_graphs = []

    if int(args.end) == -1:
        end = num_evts
    else:
        end = int(args.end)

    for evt in range(int(args.start), end):
        print(f"Processing event {evt}...")

        evt_features = {f: trk_data[f][evt] for f in trk_features}
        fullfeatmat = np.stack([evt_features[f] for f in trk_features], axis=1)
        fullfeatmat = np.array(fullfeatmat, dtype=np.float32)

        # Mask out NaNs
        nan_mask = ~np.isnan(fullfeatmat).any(axis=1)
        fullfeatmat = fullfeatmat[nan_mask]

        valid_indices = np.where(nan_mask)[0]

        # Process indices for valid tracks
        evtsiginds = [np.where(valid_indices == ind)[0][0] for ind in set(sig_ind_array[evt]) if ind in valid_indices]
        evtbkginds = [np.where(valid_indices == ind)[0][0] for ind in set(bkg_ind_array[evt]) if ind in valid_indices]

        labels = np.zeros(len(fullfeatmat), dtype=np.float32)
        labels[evtsiginds] = 1  # Label signal as 1

        # Generate random edges
        num_nodes = fullfeatmat.shape[0]
        if num_nodes < 2:  # Skip events with less than 2 nodes
            continue

        max_edges = num_nodes * (num_nodes - 1) // 2
        num_edges = max_edges // 50  # Use a fraction of possible edges

        source = np.random.randint(0, num_nodes, num_edges)
        target = np.random.randint(0, num_nodes, num_edges)

        edge_index = torch.tensor(np.vstack([source, target]), dtype=torch.int64)

        hadron_weight = 1

        # Create the graph
        evt_graph = Data(
            x=torch.tensor(fullfeatmat, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.float),
            edge_index=edge_index,
            had_weight=torch.tensor([hadron_weight], dtype=torch.float)
        )
        evt_graphs.append(evt_graph)

    return evt_graphs

print("Creating event graphs...")
event_graphs = create_event_graphs(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, trk_features)

print(f"Saving event graphs to eventdata_{args.save_tag}.pkl...")
with open(f"eventdata_{args.save_tag}.pkl", 'wb') as f:
    pickle.dump(event_graphs, f)

