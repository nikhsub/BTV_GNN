from ROOT import *
gErrorIgnoreLevel = 5000
import argparse
import numpy as np

# ------------------------
# Parse arguments
# ------------------------
parser = argparse.ArgumentParser("Compare vertex info with truth indices")
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
# Counters
# ------------------------
total_GVs   = 0
matched_GVs_reco = 0
matched_GVs_ivf  = 0

total_SVs_reco = 0
total_SVs_ivf  = 0

# ------------------------
# Loop over events
# ------------------------
for entry in tree:
    n_gv  = entry.nGV[0]
    n_sv_reco = entry.nSVs_reco[0]
    n_sv_ivf  = entry.nSVs[0]

    total_GVs       += n_gv
    total_SVs_reco  += n_sv_reco
    total_SVs_ivf   += n_sv_ivf

    # Reco matches: Hadron_SVRecoIdx
    if n_gv > 0:
        reco_idx = np.array(entry.Hadron_SVRecoIdx)
        matched_GVs_reco += np.sum(reco_idx >= 0)

    # IVF matches: Hadron_SVIdx
    if n_gv > 0:
        ivf_idx = np.array(entry.Hadron_SVIdx)
        matched_GVs_ivf += np.sum(ivf_idx >= 0)

# ------------------------
# Efficiencies
# ------------------------
eff_reco = matched_GVs_reco / total_GVs if total_GVs else 0
eff_ivf  = matched_GVs_ivf  / total_GVs if total_GVs else 0

# Fake rates: 1 - (# matched GVs / # SVs)
fake_reco = 1 - (matched_GVs_reco / total_SVs_reco) if total_SVs_reco else 0
fake_ivf  = 1 - (matched_GVs_ivf  / total_SVs_ivf) if total_SVs_ivf else 0

# ------------------------
# Reporting
# ------------------------
print("## RECO ##")
print(f"GV matching efficiency: {eff_reco:.4f}")
print(f"Fake rate: {fake_reco:.4f}")
print(f"Total GVs: {total_GVs}")
print(f"Matched GVs: {matched_GVs_reco}")
print(f"Total SVs: {total_SVs_reco}")
print()

print("## IVF ##")
print(f"GV matching efficiency: {eff_ivf:.4f}")
print(f"Fake rate: {fake_ivf:.4f}")
print(f"Total GVs: {total_GVs}")
print(f"Matched GVs: {matched_GVs_ivf}")
print(f"Total SVs: {total_SVs_ivf}")

