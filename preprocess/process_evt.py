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
parser.add_argument("-ds", "--downsample_fraction", type=float, default=0.1, help="fraction of no hf events to downsample")
args = parser.parse_args()

trk_features = ['trk_eta', 'trk_phi', 'trk_ip2d', 'trk_ip3d', 'trk_dz', 'trk_dzsig', 'trk_ip2dsig', 'trk_ip3dsig', 'trk_p', 'trk_pt',
                'trk_nValid', 'trk_nValidPixel', 'trk_nValidStrip', 'trk_charge']

# Load data
print("Loading files...")
with uproot.open(args.data) as f:
    datatree = f['tree']

num_evts = datatree.num_entries
num_classes = 7

trk_data = {feat: datatree[feat].array() for feat in trk_features}
trk_1_array  = datatree['trk_1'].array()
trk_2_array  = datatree['trk_2'].array()
deltaR_array  = datatree['deltaR'].array()
dca_array  = datatree['dca'].array()
dca_sig_array  = datatree['dca_sig'].array()
cptopv_array  = datatree['cptopv'].array()
pvtoPCA_1_array  = datatree['pvtoPCA_1'].array()
pvtoPCA_2_array  = datatree['pvtoPCA_2'].array()
dotprod_1_array  = datatree['dotprod_1'].array()
dotprod_2_array  = datatree['dotprod_2'].array()
pair_mom_array  = datatree['pair_mom'].array()
pair_invmass_array  = datatree['pair_invmass'].array()

trk_label_array  = datatree["trk_label"].array()     # 0–6
trk_hadidx_array = datatree["trk_hadidx"].array()    # hadron index
trk_flav_array   = datatree["trk_flav"].array()      # -1, 1, 2, 3, 4, 5

def create_edge_index(trk_1, trk_2, dca, deltaR, dca_sig, cptopv, pvtoPCA_1, pvtoPCA_2, dotprod_1, dotprod_2, pair_mom, pair_invmass, trk_hadidx, trk_flav):
    """
    Create edge index and edge features for tracks in val_comb_inds using DCA-based clustering.

    Edges are only formed if the DCA between two tracks is below a given threshold.
    """

    # Convert to numpy for fast indexing
    trk_1_np = trk_1.to_numpy()
    trk_2_np = trk_2.to_numpy()
    dca_np = dca.to_numpy()
    deltaR_np = deltaR.to_numpy()
    dca_sig_np = dca_sig.to_numpy()
    cptopv_np = cptopv.to_numpy()
    pvtoPCA_1_np = pvtoPCA_1.to_numpy()
    pvtoPCA_2_np = pvtoPCA_2.to_numpy()
    dotprod_1_np = dotprod_1.to_numpy()
    dotprod_2_np = dotprod_2.to_numpy()
    pair_mom_np = pair_mom.to_numpy()
    pair_invmass_np = pair_invmass.to_numpy()


    feature_mask = (
        (cptopv_np < 20) &
        (pvtoPCA_1_np < 20) &
        (pvtoPCA_2_np < 20)
    )


    # Combine both masks
    final_mask = feature_mask

    # Extract valid edges and their features
    edge_index = np.vstack([trk_1_np[final_mask], trk_2_np[final_mask]]).astype(np.int64)
    edge_features = np.vstack([dca_np[final_mask], deltaR_np[final_mask],
                               dca_sig_np[final_mask], cptopv_np[final_mask],
                                pvtoPCA_1_np[final_mask], pvtoPCA_2_np[final_mask], dotprod_1_np[final_mask], dotprod_2_np[final_mask], pair_mom_np[final_mask], pair_invmass_np[final_mask]]).T

    edge_labels = (
        (trk_hadidx[trk_1_np[final_mask]] == trk_hadidx[trk_2_np[final_mask]])
        & (trk_flav[trk_1_np[final_mask]] == trk_flav[trk_2_np[final_mask]])
        & (trk_hadidx[trk_1_np[final_mask]] >= 0)
    ).astype(np.float32)

    return edge_index, edge_features, edge_labels

def create_edge_index_full(trk_1, trk_2, dca, deltaR, dca_sig, cptopv, pvtoPCA_1, pvtoPCA_2, dotprod_1, dotprod_2, pair_mom, pair_invmass, trk_hadidx, trk_flav):
    """
    Create edge index and edge features for fully connected graph

    """

    # Convert to numpy for fast indexing
    trk_1_np = trk_1.to_numpy()
    trk_2_np = trk_2.to_numpy()
    dca_np = dca.to_numpy()
    deltaR_np = deltaR.to_numpy()
    dca_sig_np = dca_sig.to_numpy()
    cptopv_np = cptopv.to_numpy()
    pvtoPCA_1_np = pvtoPCA_1.to_numpy()
    pvtoPCA_2_np = pvtoPCA_2.to_numpy()
    dotprod_1_np = dotprod_1.to_numpy()
    dotprod_2_np = dotprod_2.to_numpy()
    pair_mom_np = pair_mom.to_numpy()
    pair_invmass_np = pair_invmass.to_numpy()

    
    # Combine both masks
    final_mask = np.ones_like(trk_1_np, dtype=bool)

    # Extract valid edges and their features
    edge_index = np.vstack([trk_1_np[final_mask], trk_2_np[final_mask]]).astype(np.int64)

    edge_features = np.vstack([dca_np[final_mask], deltaR_np[final_mask],
                               dca_sig_np[final_mask], cptopv_np[final_mask],
                                pvtoPCA_1_np[final_mask], pvtoPCA_2_np[final_mask], dotprod_1_np[final_mask], dotprod_2_np[final_mask], pair_mom_np[final_mask], pair_invmass_np[final_mask]]).T

    edge_labels = (
        (trk_hadidx[trk_1_np[final_mask]] == trk_hadidx[trk_2_np[final_mask]])
        & (trk_flav[trk_1_np[final_mask]] == trk_flav[trk_2_np[final_mask]])
        & (trk_hadidx[trk_1_np[final_mask]] >= 0)
    ).astype(np.float32)

    return edge_index, edge_features, edge_labels


def create_event_graphs(trk_data, trk_label_array, trk_hadidx_array, trk_flav_array,
                   trk_1_array, trk_2_array, deltaR_array,
                   dca_array, dca_sig_array, cptopv_array, pvtoPCA_1_array, pvtoPCA_2_array,
                   dotprod_1_array, dotprod_2_array, pair_mom_array, pair_invmass_array, trk_features, nevts=3):

    evt_graphs = []

    if int(args.end) == -1:
        end = num_evts
    else:
        end = int(args.end)

    dummy_values = np.array([-999.0] * 10 + [-1.0] * 3 + [-3.0], dtype=np.float32)  # Customize per feature

    for evt in range(int(args.start), end):
        if(evt%1000==0): print(evt)

        
        # --- build the full feature matrix ---
        evt_features = np.stack([np.asarray(trk_data[f][evt]) for f in trk_features], axis=1).astype(np.float32)

        # --- replace non-finite values with dummy pattern ---
        mask = ~np.isfinite(evt_features)
        if mask.any():
            evt_features[mask] = np.broadcast_to(dummy_values, evt_features.shape)[mask]

        ntrk = len(evt_features)
        if ntrk == 0:
            continue
        
        trk_labels_evt = np.array(trk_label_array[evt], dtype=np.int64)
        trk_hadidx_evt = np.array(trk_hadidx_array[evt], dtype=np.int64)
        trk_flav_evt   = np.array(trk_flav_array[evt], dtype=np.int64)

        hf_mask = np.isin(trk_labels_evt, [2, 3, 4]) 
        num_hf = np.sum(hf_mask)
        if num_hf < 2:
            # keep only certain fraction of these events
            if np.random.rand() > args.downsample_fraction:
                continue

        trk_1 = trk_1_array[evt]
        trk_2 = trk_2_array[evt]
        dca = dca_array[evt]
        deltaR = deltaR_array[evt]
        dca_sig = dca_sig_array[evt]
        cptopv = cptopv_array[evt]
        pvtoPCA_1 = pvtoPCA_1_array[evt]
        pvtoPCA_2 = pvtoPCA_2_array[evt]
        dotprod_1 = dotprod_1_array[evt]
        dotprod_2 = dotprod_2_array[evt]
        pair_mom = pair_mom_array[evt]
        pair_invmass = pair_invmass_array[evt]
        
        edge_index, edge_features, edge_labels = create_edge_index(trk_1, trk_2, dca, deltaR, dca_sig, cptopv, pvtoPCA_1, pvtoPCA_2, dotprod_1, dotprod_2, pair_mom, pair_invmass, trk_hadidx_evt, trk_flav_evt)
        if edge_index.shape[1] == 0:
            continue

        y_onehot = np.eye(num_classes, dtype=np.float32)[trk_labels_evt]

        evt_graph = Data(
            x=torch.tensor(evt_features, dtype=torch.float),
            y=torch.tensor(y_onehot, dtype=torch.float),     # node labels
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            edge_y=torch.tensor(edge_labels, dtype=torch.float)    # edge labels
        )

        evt_graphs.append(evt_graph)

    return evt_graphs

print("Creating event training data...")
event_graphs = create_event_graphs(trk_data, trk_label_array, trk_hadidx_array, trk_flav_array, 
                                    trk_1_array, trk_2_array, 
                                    deltaR_array, dca_array, dca_sig_array, cptopv_array, 
                                    pvtoPCA_1_array, pvtoPCA_2_array, dotprod_1_array, dotprod_2_array, 
                                    pair_mom_array, pair_invmass_array, trk_features)

print(f"Saving event training data to evttrain_{args.save_tag}.pt...")
torch.save(event_graphs, f"evttraindata_{args.save_tag}.pt")

