from ROOT import *
import argparse
import numpy as np
import uproot# as uproot
import awkward# as awkward
from collections import defaultdict
import pandas as pd
import random


parser = argparse.ArgumentParser(description="Convert signal and bkg files to numpy arrays")
parser.add_argument("-s", "--signal", default="", help="Name of signal ROOT file")
parser.add_argument("-b", "--background", default="", help="Name of background ROOT file")
parser.add_argument("-o", "--out", default="", help="Name of output dataframe")
parser.add_argument("-n", "--numbkg", default=20, help="Number of bkg tracks per b-hadron")

args = parser.parse_args()

#sig_file = TFile(args.signal, 'READ')
#sigtree = sig_file.Get("sigtree")
#
#bkg_file = TFile(args.background, 'READ')
#bkgtree = bkg_file.Get("bkgtree")

bhad_branches = ["bhad_pt", "bhad_eta", "bhad_phi", "bhad_SVx", "bhad_SVy", "bhad_SVz", "bhad_evtnum"]
trk_branches = ["ip2d", "ip3d", "ip2dsig", "ip3dsig", "pt", "eta", "phi"]

branches = bhad_branches + trk_branches

def read_root(filepath, branches, treename):
    with uproot.open(filepath) as f:
        tree = f[treename]
        outputs = tree.arrays(branches)
    return outputs

def read_files(filepath, branches, treename):
    branches = list(branches)
    table = defaultdict(list)
    a = read_root(filepath, branches, treename)
    if a is not None:
        for name in branches:
            table[name].append(a[name])
    table = {name:_concat(arrs) for name, arrs in table.items()}
    return table

def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array([])
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)

print("Starting data reading from root files")

sigout = read_files(args.signal, branches, "sigtree")
bkgout = read_files(args.background, branches, "bkgtree") # Reading the signal and background files into arrays

print("Finished data reading from root files")


#Putting information into per-track format to put into a dataframe
print("Starting per-track data creation")
trks_sig = []
trks_bkg = []
for i in range(len(sigout['bhad_pt'])):
    if(len(sigout['pt'])==0):
        continue
    for j in range(len(sigout['pt'][i])):
        trks_sig.append([sigout['ip2d'][i][j], sigout['ip3d'][i][j],sigout['ip2dsig'][i][j], sigout['ip3dsig'][i][j],sigout['pt'][i][j],sigout['eta'][i][j],sigout['phi'][i][j],
                         sigout['bhad_pt'][i][0], sigout['bhad_eta'][i][0], sigout['bhad_phi'][i][0], sigout['bhad_SVx'][i][0], sigout['bhad_SVy'][i][0], sigout['bhad_SVz'][i][0], 1, i, sigout['bhad_evtnum'][i][0]])

print("Finished signal tracks")
for i in range(len(bkgout['bhad_pt'])):
    if len(bkgout['pt'][i]) > int(args.numbkg):
        sampled_indices = random.sample(range(len(bkgout['pt'][i])), int(args.numbkg))
    else:
        sampled_indices = range(len(bkgout['pt'][i]))

    for j in sampled_indices:
        trks_bkg.append([bkgout['ip2d'][i][j], bkgout['ip3d'][i][j],bkgout['ip2dsig'][i][j], bkgout['ip3dsig'][i][j],bkgout['pt'][i][j],bkgout['eta'][i][j],bkgout['phi'][i][j],
                     bkgout['bhad_pt'][i][0], bkgout['bhad_eta'][i][0], bkgout['bhad_phi'][i][0], bkgout['bhad_SVx'][i][0], bkgout['bhad_SVy'][i][0], bkgout['bhad_SVz'][i][0], 0, i, bkgout['bhad_evtnum'][i][0]])

print("Finished per-track data creation")


#Creating and saving dataframe
columns = ['trks_ip2d', 'trks_ip3d', 'trks_ip2dsig', 'trks_ip3dsig', 'trks_pt', 'trks_eta', 'trks_phi',
           'bhad_pt', 'bhad_eta', 'bhad_phi', 'bhad_SVx', 'bhad_SVy', 'bhad_SVz', 'is_signal', "bhad_num", 'evt_num']

combined_tracks = trks_sig + trks_bkg

print("Creating dataframe...")

df = pd.DataFrame(combined_tracks, columns=columns)

print("Saving dataframe")

df.to_csv(args.out, index=False)

