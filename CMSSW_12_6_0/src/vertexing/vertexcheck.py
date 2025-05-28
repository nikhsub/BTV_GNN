from ROOT import *
import sys
import numpy as np
import argparse
#import array
import math
import numpy as np
import random

parser = argparse.ArgumentParser("Compare vertex info")

parser.add_argument("-i", "--inp", default="test_ntuple.root", help="Input root file")

args = parser.parse_args()

infile = args.inp

Infile = TFile(infile, 'READ')
demo = Infile.Get('demo')
tree = demo.Get('tree')

match_threshold = 0.1

# Initialize counters
total_GVs = 0
matched_GVs = 0
total_SVs = 0
matched_SVs = 0
total_ivf = 0
matched_ivf = 0
matched_GVs_ivf = 0

for entry in tree:
    n_gv = entry.nGV[0]
    n_sv = entry.nSVs_reco[0]
    n_ivf = entry.nSVs[0]
    total_GVs += n_gv
    total_SVs += n_sv
    total_ivf += n_ivf

    GV_coords = np.array(list(zip(entry.Hadron_GVx, entry.Hadron_GVy, entry.Hadron_GVz)))
    SV_coords = np.array(list(zip(entry.SV_x_reco, entry.SV_y_reco, entry.SV_z_reco)))
    ivf_coords = np.array(list(zip(entry.SV_x, entry.SV_y, entry.SV_z)))

    used_sv_indices = set()
    matched_SVs_event = set()
    used_ivf_indices = set()
    matched_ivf_event = set()

    for gv in GV_coords:
        if len(SV_coords) == 0:
            continue  # No SVs to compare, skip this GV
        distances = np.linalg.norm(SV_coords - gv, axis=1)
        sv_idx = np.argmin(distances)
        #print(distances[sv_idx])
        if distances[sv_idx] < match_threshold and sv_idx not in used_sv_indices:
            matched_GVs += 1
            matched_SVs_event.add(sv_idx)
            used_sv_indices.add(sv_idx)

    matched_SVs += len(matched_SVs_event)  # count for this event

    for gv in GV_coords:
        if len(ivf_coords) == 0:
            continue  # No SVs to compare, skip this GV
        distances = np.linalg.norm(ivf_coords - gv, axis=1)
        ivf_idx = np.argmin(distances)
        #print(distances[sv_idx])
        if distances[ivf_idx] < match_threshold and ivf_idx not in used_ivf_indices:
            matched_GVs_ivf += 1
            matched_ivf_event.add(ivf_idx)
            used_ivf_indices.add(ivf_idx)

    matched_ivf += len(matched_ivf_event)  # count for this event

# Reporting
print(f"Total GVs: {total_GVs}")
print(f"Matched GVs: {matched_GVs}")
print(f"Total SVs: {total_SVs}")
print(f"Matched SVs: {matched_SVs}")
print("")
print(f"Total GVs: {total_GVs}")
print(f"Matched GVs IVF: {matched_GVs_ivf}")
print(f"Total IVF SVs: {total_ivf}")
print(f"Matched IVF SVs: {matched_ivf}")



    
