#!/usr/bin/env python3
from ROOT import *
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------
input_dir = "."  # directory containing ROOT files
base_script = "newvertexcheck.py"  # your vertex check script
root_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".root")])

# --- GEN efficiencies and fake rates (editable) ---
num_files = len(root_files)
GEN_eff  = [56.6] * num_files
GEN_fake = [22.0] * num_files

# -------------------------------------------------------------
# STORAGE ARRAYS
# -------------------------------------------------------------
cut_values = []
RECO_eff, IVF_eff = [], []
RECO_fake, IVF_fake = [], []

# -------------------------------------------------------------
# LOOP OVER FILES, RUN SCRIPT, EXTRACT VALUES
# -------------------------------------------------------------
for f in root_files:
    print(f"\n>>> Processing {f} ...")

    # Run your vertexcheck script
    result = subprocess.run(
        ["python3", base_script, "-i", os.path.join(input_dir, f)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    lines = result.stdout.splitlines()
    eff_r = eff_i = fake_r = fake_i = None

    for i, line in enumerate(lines):
        if "## RECO ##" in line:
            eff_r = float(lines[i+1].split(":")[1])*100
            fake_r = float(lines[i+2].split(":")[1])*100
        if "## IVF ##" in line:
            eff_i = float(lines[i+1].split(":")[1])*100
            fake_i = float(lines[i+2].split(":")[1])*100

    if None in (eff_r, eff_i, fake_r, fake_i):
        print(f"⚠️ Warning: Could not parse metrics from {f}")
        continue

    # Extract cut value from filename (e.g. mod0p5cut → 0.5)
    cut_str = f.split("mod")[1].split("cut")[0].replace("p", ".")
    cut_val = float(cut_str)
    cut_values.append(cut_val)

    # Append metrics
    RECO_eff.append(eff_r)
    IVF_eff.append(eff_i)
    RECO_fake.append(fake_r)
    IVF_fake.append(fake_i)
    
# -------------------------------------------------------------
# SORT BY CUT VALUE
# -------------------------------------------------------------

cut_values, RECO_eff, IVF_eff, GEN_eff, RECO_fake, IVF_fake, GEN_fake = zip(
    *sorted(zip(
        cut_values,
        RECO_eff,
        IVF_eff,
        GEN_eff,
        RECO_fake,
        IVF_fake,
        GEN_fake
    ))
)

# -------------------------------------------------------------
# PLOT 1: Efficiency vs Cut
# -------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(cut_values, GEN_eff,  '^-', color='tab:orange', label="GEN Efficiency",  linewidth=2, markersize=6)
plt.plot(cut_values, RECO_eff, 's-', color='tab:red',   label="RECO Efficiency", linewidth=2, markersize=6)
plt.plot(cut_values, IVF_eff,  'o--', color='tab:blue', label="IVF Efficiency", linewidth=2, markersize=6)

plt.title("GV Matching Efficiency vs Model Cut", fontsize=14)
plt.xlabel("Model Cut", fontsize=12)
plt.ylabel("Efficiency (%)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("efficiency_vs_cut.png")
plt.close()

# -------------------------------------------------------------
# PLOT 2: Fake Rate vs Cut
# -------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.plot(cut_values, GEN_fake,  '^-', color='tab:orange', label="GEN Fake Rate",  linewidth=2, markersize=6)
plt.plot(cut_values, RECO_fake, 's-', color='tab:red',   label="RECO Fake Rate", linewidth=2, markersize=6)
plt.plot(cut_values, IVF_fake,  'o--', color='tab:blue', label="IVF Fake Rate", linewidth=2, markersize=6)

plt.title("GV Fake Rate vs Model Cut", fontsize=14)
plt.xlabel("Model Cut", fontsize=12)
plt.ylabel("Fake Rate (%)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("fake_rate_vs_cut.png")
plt.close()

print("\n✅ Plots saved as 'efficiency_vs_cut.png' and 'fake_rate_vs_cut.png'")

