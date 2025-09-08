from ROOT import *
import sys
import numpy as np
import argparse
#import array
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser("Compare vertex info")

parser.add_argument("-i", "--inp", default="test_ntuple.root", help="Input root file")

args = parser.parse_args()

infile = args.inp

Infile = TFile(infile, 'READ')
demo = Infile.Get('demo')
tree = demo.Get('tree')

# Accumulate all GV and SV coordinates across events
all_gv_x, all_gv_y, all_gv_z = [], [], []
all_sv_x, all_sv_y, all_sv_z = [], [], []

for i, entry in enumerate(tree):
    if(i>10): break
    if len(entry.Hadron_GVx) > 0:
        all_gv_x.extend(entry.Hadron_GVx)
        all_gv_y.extend(entry.Hadron_GVy)
        all_gv_z.extend(entry.Hadron_GVz)
    
    if len(entry.SV_x_reco) > 0:
        all_sv_x.extend(entry.SV_x_reco)
        all_sv_y.extend(entry.SV_y_reco)
        all_sv_z.extend(entry.SV_z_reco)

# Convert to NumPy arrays for plotting
gv_x = np.array(all_gv_x)
gv_y = np.array(all_gv_y)
gv_z = np.array(all_gv_z)

sv_x = np.array(all_sv_x)
sv_y = np.array(all_sv_y)
sv_z = np.array(all_sv_z)

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot gen vertices
ax.scatter(gv_x, gv_y, gv_z, c='blue', marker='x', label='Gen Vertices (GV)', alpha=0.6)

# Plot reco vertices
ax.scatter(sv_x, sv_y, sv_z, c='red', marker='o', label='Reco Vertices (SV)', alpha=0.5)

ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')
ax.set_title('3D Plot of All Gen and Reco Vertices')
ax.legend()
plt.tight_layout()
plt.savefig("testplot.png")
