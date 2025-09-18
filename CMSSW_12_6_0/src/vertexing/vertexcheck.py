from ROOT import *
gErrorIgnoreLevel = 5000
import argparse
import numpy as np
import math
import random
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# ------------------------
# Parse arguments
# ------------------------
parser = argparse.ArgumentParser("Compare vertex info")
parser.add_argument("-i", "--inp", default="test_ntuple.root", help="Input root file")
args = parser.parse_args()

# ------------------------
# Open ROOT file
# ------------------------
infile = args.inp
Infile = TFile(infile, 'READ')
demo = Infile.Get('demo')
tree = demo.Get('tree')

# ------------------------
# Config
# ------------------------
match_threshold = 2.0

# Counters
total_GVs = 0
matched_GVs = 0
total_SVs = 0
matched_SVs = 0
total_ivf = 0
matched_ivf = 0
matched_GVs_ivf = 0

fake_SVs = 0
fake_IVFs = 0

# Min-distance storage
min_dists_gnn = []
min_dists_ivf = []

# ------------------------
# Loop over events
# ------------------------
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

    # -------- GV ↔ RECO SV matching --------
    if len(GV_coords) and len(SV_coords):
        cost_matrix = np.linalg.norm(GV_coords[:, None, :] - SV_coords[None, :, :], axis=2)
        gv_idx, sv_idx = linear_sum_assignment(cost_matrix)
        matched_SVs_event = set()
        for g, s in zip(gv_idx, sv_idx):
            dist = cost_matrix[g, s]
            min_dists_gnn.append(dist)
            if dist < match_threshold:
                matched_GVs += 1
                matched_SVs_event.add(s)
        matched_SVs += len(matched_SVs_event)
        fake_SVs += (n_sv - len(matched_SVs_event))

    # -------- GV ↔ IVF SV matching --------
    if len(GV_coords) and len(ivf_coords):
        cost_matrix = np.linalg.norm(GV_coords[:, None, :] - ivf_coords[None, :, :], axis=2)
        gv_idx, ivf_idx = linear_sum_assignment(cost_matrix)
        matched_IVF_event = set()
        for g, s in zip(gv_idx, ivf_idx):
            dist = cost_matrix[g, s]
            min_dists_ivf.append(dist)
            if dist < match_threshold:
                matched_GVs_ivf += 1
                matched_IVF_event.add(s)
        matched_ivf += len(matched_IVF_event)
        fake_IVFs += (n_ivf - len(matched_IVF_event))

# ------------------------
# Close ROOT file
# ------------------------
Infile.Close()

# ------------------------
# Efficiencies
# ------------------------
eff_GV_match     = matched_GVs     / total_GVs if total_GVs else 0
eff_SV_recovery  = matched_SVs     / total_SVs if total_SVs else 0
eff_ivf_GV_match = matched_GVs_ivf / total_GVs if total_GVs else 0
eff_ivf_recovery = matched_ivf     / total_ivf if total_ivf else 0

# Fake rates
fake_rate_SV  = fake_SVs  / total_SVs if total_SVs else 0
fake_rate_IVF = fake_IVFs / total_ivf if total_ivf else 0

# ------------------------
# Reporting
# ------------------------
print(f"GV matching efficiency (to RECO SV): {eff_GV_match:.3f}")
print(f"GV matching efficiency (to IVF SV):  {eff_ivf_GV_match:.3f}")
print()
print(f"Fake rate (RECO SVs not matched): {fake_rate_SV:.3f}")
print(f"Fake rate (IVF SVs not matched):  {fake_rate_IVF:.3f}")
print()
print("## RECO ##")
print(f"Total GVs: {total_GVs}")
print(f"Matched GVs: {matched_GVs}")
print(f"Total SVs: {total_SVs}")
print(f"Matched SVs: {matched_SVs}")
print(f"Fake SVs: {fake_SVs}")
print()
print("## IVF ##")
print(f"Total GVs: {total_GVs}")
print(f"Matched GVs IVF: {matched_GVs_ivf}")
print(f"Total IVF SVs: {total_ivf}")
print(f"Matched IVF SVs: {matched_ivf}")
print(f"Fake IVF SVs: {fake_IVFs}")

# ------------------------
# Plots: min distances
# ------------------------
plt.figure(figsize=(8, 5))
plt.hist(min_dists_gnn, bins=100, alpha=0.6, label="Genmatch", color='red', log=True)
plt.hist(min_dists_ivf, bins=100, alpha=0.6, label="IVF", color='blue', log=True)
plt.axvline(match_threshold, color='k', linestyle='--', label=f"Current threshold = {match_threshold}")
plt.xlabel("Minimum distance between GV and SV [cm]")
plt.ylabel("Number of GVs (log scale)")
plt.title("Minimum GV-SV Distance Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mindist_genmatch.png")

# ------------------------
# Threshold scan
# ------------------------
thresholds = np.linspace(0.0, 10.0, 500)
min_dists_gnn_np = np.array(min_dists_gnn)
min_dists_ivf_np = np.array(min_dists_ivf)

eff_gnn  = [np.sum(min_dists_gnn_np < thr) / total_GVs if total_GVs else 0 for thr in thresholds]
eff_ivf  = [np.sum(min_dists_ivf_np < thr) / total_GVs if total_GVs else 0 for thr in thresholds]
fake_gnn = [(total_SVs - np.sum(min_dists_gnn_np < thr)) / total_SVs if total_SVs else 0 for thr in thresholds]
fake_ivf = [(total_ivf - np.sum(min_dists_ivf_np < thr)) / total_ivf if total_ivf else 0 for thr in thresholds]

# --- Plot Efficiency vs Threshold
plt.figure(figsize=(8,5))
plt.plot(thresholds, eff_gnn, label="GV→RECO SV efficiency", color="red")
plt.plot(thresholds, eff_ivf, label="GV→IVF SV efficiency", color="blue")
plt.axvline(match_threshold, color='k', linestyle='--', label=f"Chosen thr = {match_threshold}")
plt.xlabel("Matching distance threshold [cm]")
plt.ylabel("Efficiency")
plt.title("GV→SV Matching Efficiency vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("efficiency_vs_threshold.png")

# --- Plot Fake Rate vs Threshold
plt.figure(figsize=(8,5))
plt.plot(thresholds, fake_gnn, label="RECO SV fake rate", color="red")
plt.plot(thresholds, fake_ivf, label="IVF SV fake rate", color="blue")
plt.axvline(match_threshold, color='k', linestyle='--', label=f"Chosen thr = {match_threshold}")
plt.xlabel("Matching distance threshold [cm]")
plt.ylabel("Fake Rate")
plt.title("SV Fake Rate vs Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fakerate_vs_threshold.png")

