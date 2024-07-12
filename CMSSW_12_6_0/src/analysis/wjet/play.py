from ROOT import *
import matplotlib.pyplot as plt
import numpy as np

infile = "checkfile.root"

Infile = TFile(infile, 'READ')
#tree_dir = Infile.Get('btagana')

tree = Infile.Get('checktree')

delr = []

for evt in tree:
    for dr in evt.delr:
        delr.append(dr)

plt.hist(delr, bins=500, color='white', edgecolor='blue', histtype='stepfilled')

plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5, color='gray')

plt.xticks(ticks=np.arange(0, 5, 0.2), labels=[f"{x:.2f}" for x in np.arange(0, 5, 0.2)])

plt.yscale('log')
plt.show()
