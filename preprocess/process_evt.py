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
#bkg_ind_array = datatree['bkg_ind'].array()
SV_ind_array = datatree['SVtrk_ind'].array()
had_pt_array = datatree['had_pt'].array()
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

def create_edge_index(trk_1, trk_2, dca, deltaR, dca_sig, cptopv, pvtoPCA_1, pvtoPCA_2, dotprod_1, dotprod_2, pair_mom, pair_invmass, val_comb_inds):
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

    # Filter based on val_comb_inds (valid track indices)
    valid_edge_mask = np.isin(trk_1_np, val_comb_inds) & np.isin(trk_2_np, val_comb_inds)

    feature_mask = (
        (cptopv_np < 15) &
        (dca_np < 0.125) &
        (dca_sig_np < 30) &
        (pvtoPCA_1_np < 15) &
        (pvtoPCA_2_np < 15) &
        (np.abs(dotprod_1_np) > 0.75) &
        (np.abs(dotprod_2_np) > 0.75) &
        (pair_invmass_np < 5) &
        (pair_mom_np < 100)
    )


    # Combine both masks
    final_mask = valid_edge_mask & feature_mask

    # Extract valid edges and their features
    edge_index = np.vstack([trk_1_np[final_mask], trk_2_np[final_mask]]).astype(np.int64)
    edge_features = np.vstack([dca_np[final_mask], deltaR_np[final_mask],
                               dca_sig_np[final_mask], cptopv_np[final_mask],
                                pvtoPCA_1_np[final_mask], pvtoPCA_2_np[final_mask], dotprod_1_np[final_mask], dotprod_2_np[final_mask], pair_mom_np[final_mask], pair_invmass_np[final_mask]]).T

    return edge_index, edge_features

def create_event_graphs(trk_data, sig_ind_array, sig_flag_array,
                   SV_ind_array, had_pt_array, trk_1_array, trk_2_array, deltaR_array,
                   dca_array, dca_sig_array, cptopv_array, pvtoPCA_1_array, pvtoPCA_2_array,
                   dotprod_1_array, dotprod_2_array, pair_mom_array, pair_invmass_array, trk_features, nevts=3):

    evt_graphs = []

    if int(args.end) == -1:
        end = num_evts
    else:
        end = int(args.end)

    dummy_values = np.array([-999.0] * 8 + [-1.0] * 3 + [-3.0], dtype=np.float32)  # Customize per feature

    for evt in range(int(args.start), end):
        print(evt)

        evt_features = {f: trk_data[f][evt] for f in trk_features}
        fullfeatmat = np.stack([evt_features[f] for f in trk_features], axis=1)
        fullfeatmat = np.array(fullfeatmat, dtype=np.float32)


        # Where values are NaN or inf, replace with dummy
        mask = ~np.isfinite(fullfeatmat)
        fullfeatmat[mask] = np.broadcast_to(dummy_values, fullfeatmat.shape)[mask]

        #valid_indices = np.where(nan_mask)[0]
        #
        #val_inds_map = {ind: i for i, ind in enumerate(valid_indices)}
        valid_indices = np.arange(len(fullfeatmat))
        val_inds_map = {ind: ind for ind in valid_indices}  # Identity map
        
        evtsiginds = list(set(sig_ind_array[evt]))
        
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
    
        #evtbkginds = list(set(bkg_ind_array[evt]))
        #evtbkginds = [ind for ind in evtbkginds if ind not in evtsiginds]
    
        evtsiginds = [val_inds_map[ind] for ind in evtsiginds if ind in val_inds_map]
        if len(evtsiginds) < 2:
            continue  # Skip this event

        #evtbkginds = [val_inds_map[ind] for ind in evtbkginds if ind in val_inds_map]
        
        edge_index, edge_features = create_edge_index(trk_1, trk_2, dca, deltaR, dca_sig, cptopv, pvtoPCA_1, pvtoPCA_2, dotprod_1, dotprod_2, pair_mom, pair_invmass, valid_indices)
        if edge_index.shape[1] == 0:
            continue
        
        edge_index = np.vectorize(val_inds_map.get)(edge_index)

        labels = np.zeros(len(fullfeatmat), dtype=np.float32)
        labels[evtsiginds] = 1  # Label signal as 1

        hadron_weight = 1

        # Create the graph
        evt_graph = Data(
            x=torch.tensor(fullfeatmat, dtype=torch.float),
            y=torch.tensor(labels, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.int64),
            edge_attr=torch.tensor(edge_features, dtype=torch.float),
            had_weight=torch.tensor([hadron_weight], dtype=torch.float)
        )

        evt_graphs.append(evt_graph)

    return evt_graphs

print("Creating event training data...")
event_graphs = create_event_graphs(trk_data, sig_ind_array, sig_flag_array, 
                                    SV_ind_array, had_pt_array, trk_1_array, trk_2_array, deltaR_array,
                                    dca_array, dca_sig_array, cptopv_array, pvtoPCA_1_array, pvtoPCA_2_array,
                                    dotprod_1_array, dotprod_2_array, pair_mom_array, pair_invmass_array, trk_features)

print(f"Saving event training data to evttrain_{args.save_tag}.pkl...")
with open(f"evttraindata_{args.save_tag}.pkl", 'wb') as f:
    pickle.dump(event_graphs, f)

