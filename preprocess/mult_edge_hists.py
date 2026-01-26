import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

edge_feature_names = [
    'dca', 'deltaR', 'dca_sig', 'cptopv', 'pvtoPCA_1',
    'pvtoPCA_2', 'dotprod_1', 'dotprod_2', 'pair_mom', 'pair_invmass'
]
NUM_EDGE_FEATURES = len(edge_feature_names)

parser = argparse.ArgumentParser("Edge feature distributions using edge_y labels (FAST)")
parser.add_argument("-f", "--file", required=True, help="Testing data file (.pt torch.load)")
parser.add_argument("--sig_value", type=float, default=1.0, help="edge_y value treated as signal")
parser.add_argument("--bins", type=int, default=50, help="Histogram bins (default: 50)")
parser.add_argument("--max_events", type=int, default=-1, help="Cap number of events (-1 = all)")
parser.add_argument("--start", type=int, default=0, help="Start index in event list")
args = parser.parse_args()

evt_data_all = torch.load(args.file)

start = max(0, args.start)
if args.max_events is not None and args.max_events > 0:
    stop = min(len(evt_data_all), start + args.max_events)
    evt_data_list = evt_data_all[start:stop]
else:
    evt_data_list = evt_data_all[start:]

print(f"Loaded events: {len(evt_data_all)} | Processing: {len(evt_data_list)} (start={start})")

# Collect as list-of-arrays first; concatenate once at the end (fast + less overhead)
edge_chunks = {
    "S": [[] for _ in range(NUM_EDGE_FEATURES)],
    "B": [[] for _ in range(NUM_EDGE_FEATURES)],
}
edge_type_counts = {"S": 0, "B": 0}

for ievt, evt_data in enumerate(evt_data_list):
    edge_index = getattr(evt_data, "edge_index", None)
    edge_attr  = getattr(evt_data, "edge_attr", None)
    edge_y     = getattr(evt_data, "edge_y", None)

    if edge_index is None or edge_attr is None or edge_y is None:
        continue

    E = int(edge_index.shape[1])
    if E == 0:
        continue

    # Move once to CPU numpy
    ea = edge_attr.detach().cpu().numpy()          # [E, F]
    ey = edge_y.detach().cpu().view(-1).numpy()    # [E]

    if ey.shape[0] != E:
        continue
    if ea.shape[1] < NUM_EDGE_FEATURES:
        raise ValueError(f"edge_attr has {ea.shape[1]} features, expected >= {NUM_EDGE_FEATURES}")

    sig_mask = (ey == args.sig_value)
    bkg_mask = ~sig_mask

    nS = int(sig_mask.sum())
    nB = E - nS
    edge_type_counts["S"] += nS
    edge_type_counts["B"] += nB

    # Slice all features at once, then store per-feature columns
    # These are views; we copy to float32 to reduce memory
    eaS = ea[sig_mask, :NUM_EDGE_FEATURES].astype(np.float32, copy=False)
    eaB = ea[bkg_mask, :NUM_EDGE_FEATURES].astype(np.float32, copy=False)

    for f in range(NUM_EDGE_FEATURES):
        if nS:
            edge_chunks["S"][f].append(eaS[:, f])
        if nB:
            edge_chunks["B"][f].append(eaB[:, f])

    # Optional progress print every N events
    if (ievt + 1) % 10 == 0:
        print(f"Processed {ievt+1}/{len(evt_data_list)} events | edges so far S={edge_type_counts['S']} B={edge_type_counts['B']}")

print("Edge counts by edge_y:", edge_type_counts)

# Final concatenate once
edge_feature_data = {"S": [], "B": []}
for tag in ("S", "B"):
    for f in range(NUM_EDGE_FEATURES):
        if len(edge_chunks[tag][f]) == 0:
            edge_feature_data[tag].append(np.array([], dtype=np.float32))
        else:
            edge_feature_data[tag].append(np.concatenate(edge_chunks[tag][f]))

# Plot
for f, feat_name in enumerate(edge_feature_names):
    plt.figure(figsize=(8, 5))
    for tag, color in zip(["S", "B"], ["red", "blue"]):
        values = edge_feature_data[tag][f]
        if values.size:
            plt.hist(values, bins=args.bins, alpha=0.5, label=tag, color=color, log=True)
            plt.xscale("log")
            plt.grid(True, which="both", axis="both", linestyle="--", alpha=0.4)

    plt.title(f"Edge Feature: {feat_name}")
    plt.xlabel(feat_name)
    plt.ylabel("# of Edges")
    plt.legend()
    plt.tight_layout()
    outname = f"{feat_name}_hist_edgey.png"
    print(f"Saving {outname}")
    plt.savefig(outname)
    plt.close()

