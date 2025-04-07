import warnings
warnings.filterwarnings("ignore")

import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

edge_feature_names = ['dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1',
                      'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass']
NUM_EDGE_FEATURES = len(edge_feature_names)

parser = argparse.ArgumentParser("Model comparison")
parser.add_argument("-f", "--file", required=True, help="Testing data file")
args = parser.parse_args()

with open(args.file, "rb") as f:
    evt_data_list = pickle.load(f)  # could be list or dict of Data objects

# 📊 Organize edge features by connection type
edge_feature_data = {
    "SS": [[] for _ in range(NUM_EDGE_FEATURES)],
    "SB": [[] for _ in range(NUM_EDGE_FEATURES)],
    "BB": [[] for _ in range(NUM_EDGE_FEATURES)],
}

IDX_CPTOPV = 3
IDX_PVTOPCA1 = 4
IDX_PVTOPCA2 = 5
IDX_PAIR_MOM = 8
IDX_PAIR_INV = 9

# 🔄 Process each event
for evt_data in evt_data_list:
    edge_index = evt_data.edge_index
    edge_attr = evt_data.edge_attr
    siginds = evt_data.siginds.cpu().numpy()
    num_nodes = evt_data.x.shape[0]

    if edge_index.shape[1] == 0 or len(siginds) == 0:
        continue

    is_sig = np.zeros(num_nodes, dtype=bool)
    is_sig[siginds] = True

    for e_idx in range(edge_index.shape[1]):
        i = edge_index[0, e_idx].item()
        j = edge_index[1, e_idx].item()
        
        cptopv = edge_attr[e_idx, IDX_CPTOPV].item()
        pair_mom = edge_attr[e_idx, IDX_PAIR_MOM].item()
        pvtoPCA1 = edge_attr[e_idx, IDX_PVTOPCA1].item()
        pvtoPCA2 = edge_attr[e_idx, IDX_PVTOPCA2].item()
        pair_inv = edge_attr[e_idx, IDX_PAIR_INV].item()

        if cptopv > 400 or pair_mom > 500 or pair_inv > 10 or pvtoPCA1 > 400 or pvtoPCA2 > 400:
            continue  # skip this edge entirely

        if is_sig[i] and is_sig[j]:
            tag = "SS"
        elif is_sig[i] or is_sig[j]:
            tag = "SB"
        else:
            tag = "BB"

        for f in range(NUM_EDGE_FEATURES):
            val = edge_attr[e_idx, f].item()
            edge_feature_data[tag][f].append(val)

# 📈 Plot histograms for each edge feature
for f, feat_name in enumerate(edge_feature_names):
    plt.figure(figsize=(8, 5))
    for tag, color in zip(["SS", "SB", "BB"], ['red', 'green', 'orange']):
        values = edge_feature_data[tag][f]
        if values:  # non-empty
            plt.hist(values, bins=50, alpha=0.5, label=tag, color=color, density=False, log=True)
    plt.title(f"Edge Feature: {feat_name}")
    plt.xlabel(feat_name)
    plt.ylabel("# of Edges")
    plt.legend()
    plt.tight_layout()
    print(f"Saving {feat_name} hist")
    plt.savefig(f"{feat_name}_hist.png")
    plt.close()

