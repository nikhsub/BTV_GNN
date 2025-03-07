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

def compute_deltaR_matrix(eta, phi):
    """Computes the ΔR matrix for a given event."""
    eta = np.array(eta)[:, np.newaxis]  # Shape (N, 1)
    phi = np.array(phi)[:, np.newaxis]  # Shape (N, 1)

    delta_eta = eta - eta.T  # Broadcast subtraction (N, N)
    delta_phi = np.arctan2(np.sin(phi - phi.T), np.cos(phi - phi.T))  # Handles periodicity
    deltaR_matrix = np.sqrt(delta_eta**2 + delta_phi**2)  # Element-wise sqrt

    np.fill_diagonal(deltaR_matrix, np.inf)  # Avoid self-connections

    return deltaR_matrix

def compute_deltaR_threshold(deltaR_matrix, signal_indices):

    signal_pairs = [(i, j) for i in signal_indices for j in signal_indices if i < j]
    signal_distances = [deltaR_matrix[i, j] for i, j in signal_pairs]

    return np.mean(signal_distances) if signal_distances else None

def create_edge_index(deltaR_matrix, threshold):
    """Returns edge index tensor for graph connectivity based on ΔR threshold."""
    row, col = np.where(deltaR_matrix < threshold)  # Get valid edges
    edge_index = np.vstack((row, col))  # Shape (2, E)
    
    return torch.tensor(edge_index, dtype=torch.int64)

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

        # Filter events with at least 3 signal indices
        if len(evtsiginds) < 3:
            continue  # Skip this event

        labels = np.zeros(len(fullfeatmat), dtype=np.float32)
        labels[evtsiginds] = 1  # Label signal as 1

        eta = evt_features['trk_eta'][nan_mask]
        phi = evt_features['trk_phi'][nan_mask]

        deltaR_matrix = compute_deltaR_matrix(eta, phi)

        deltaR_thres = compute_deltaR_threshold(deltaR_matrix, evtsiginds) 

        edge_index = create_edge_index(deltaR_matrix, deltaR_thres)

        # Generate random edges
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

print("Creating event training data...")
event_graphs = create_event_graphs(trk_data, sig_ind_array, sig_flag_array, bkg_flag_array, bkg_ind_array, trk_features)

print(f"Saving event training data to evttrain_{args.save_tag}.pkl...")
with open(f"evttraindata_{args.save_tag}.pkl", 'wb') as f:
    pickle.dump(event_graphs, f)

