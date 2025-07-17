from ROOT import *
import sys
import numpy as np
import argparse
#import array
import math
import numpy as np
import random
import csv

parser = argparse.ArgumentParser("Create track information root file")

parser.add_argument("-i", "--inp", default="test_ntuple.root", help="Input root file")
parser.add_argument("-o", "--out", default="testfile", help="Name of output ROOT file")
parser.add_argument("-s", "--start", type=int, help="Start index for events")
parser.add_argument("-e", "--end", type=int, help="End index for events")
parser.add_argument("-lpt", "--lowpt", default=False, action="store_true", help="Apply low pt cut?")

args = parser.parse_args()

infile = args.inp
start_index = args.start
end_index = args.end

#Infile = TFile(infile, 'READ')
#demo = Infile.Get('demo')
#tree = demo.Get('tree')

filenames = args.inp.split(',')

tree = TChain("demo/tree")
for filename in filenames:
    tree.AddFile(filename)

Outfile = TFile(args.out+".root", "recreate")
outtree = TTree("tree", "tree")

sig_ind       = std.vector('int')()
sig_flag      = std.vector('int')()
sig_flav      = std.vector('int')()

run = std.vector('int')()
lumi = std.vector('int')()
event = std.vector('int')()

bkg_ind       = std.vector('int')()
bkg_flag      = std.vector('int')()

#delr     = std.vector('double')()
#ptrat     = std.vector('double')()

SVtrk_ind     = std.vector('int')()

trk_ip2d        = std.vector('double')()
trk_ip3d        = std.vector('double')()
trk_ip2dsig     = std.vector('double')()
trk_ip3dsig     = std.vector('double')()
trk_p           = std.vector('double')()
trk_pt          = std.vector('double')()
trk_eta         = std.vector('double')()
trk_phi         = std.vector('double')()
trk_nValid      = std.vector('double')()
trk_nValidPixel = std.vector('double')()
trk_nValidStrip = std.vector('double')()
trk_charge      = std.vector('double')()

had_pt          = std.vector('double')()
nhads           = std.vector('int')()

missed_sig      = std.vector('int')()
trk_1           = std.vector('int')()
trk_2           = std.vector('int')()
deltaR          = std.vector('double')()
dca             = std.vector('double')()
dca_sig         = std.vector('double')()
cptopv          = std.vector('double')()
pvtoPCA_1       = std.vector('double')()
pvtoPCA_2       = std.vector('double')()
dotprod_1       = std.vector('double')()
dotprod_2       = std.vector('double')()
pair_mom        = std.vector('double')()
pair_invmass    = std.vector('double')()
preds           = std.vector('double')()

branches = {

    "run": run, "lumi": lumi, "event": event, "had_pt": had_pt, "sig_flag": sig_flag, "sig_flav": sig_flav, "sig_ind": sig_ind, "bkg_ind": bkg_ind,
    "bkg_flag": bkg_flag, "SVtrk_ind": SVtrk_ind, "trk_ip2d": trk_ip2d,
    "trk_ip3d": trk_ip3d, "trk_ip2dsig": trk_ip2dsig, "trk_ip3dsig": trk_ip3dsig, "trk_p": trk_p,
    "trk_pt": trk_pt, "trk_eta": trk_eta, "trk_phi": trk_phi, "trk_nValid": trk_nValid,
    "trk_nValidPixel": trk_nValidPixel, "trk_nValidStrip": trk_nValidStrip, "trk_charge": trk_charge,
    "missed_sig": missed_sig, "trk_1": trk_1, "trk_2": trk_2, "deltaR": deltaR,
    "dca": dca, "dca_sig": dca_sig, "cptopv": cptopv, "pvtoPCA_1": pvtoPCA_1, "pvtoPCA_2": pvtoPCA_2,
    "dotprod_1": dotprod_1, "dotprod_2": dotprod_2, "pair_mom": pair_mom, "pair_invmass": pair_invmass,
    "preds": preds
}

for name, branch in branches.items():
    outtree.Branch(name, branch)

run_list = []   # or directly append run number per event
lumi_list = []
evt_list = []

sig_ind_list = []

#outtree.Branch("delr", delr)
#outtree.Branch("ptrat", ptrat)

def delta_R(eta1, phi1, eta2, phi2):
    """Efficiently compute ΔR using vectorized operations."""
    deta = eta1[:, None] - eta2  # Vectorized subtraction
    dphi = np.abs(phi1[:, None] - phi2)
    dphi[dphi > np.pi] -= 2 * np.pi  # Ensure dphi is in [-π, π]
    return np.sqrt(deta**2 + dphi**2)

for i, evt in enumerate(tree):
    if i < start_index:
        continue
    if i >= end_index:
        break

    if(i%10 ==0): 
        print("EVT", i) 
    
    for branch in branches.values():
        branch.clear()
    
    if(args.lowpt):
        high_pt = np.any(np.array(evt.Hadron_pt) > 20)
        if(high_pt): continue
    
    nHads = evt.nHadrons[0]
    had_pt.reserve(nHads)
    had_pt.assign(evt.Hadron_pt.begin(), evt.Hadron_pt.end());
    preds.assign(evt.preds.begin(), evt.preds.end())

    run.push_back(int(evt.run))
    lumi.push_back(int(evt.lumi))
    event.push_back(int(evt.evt))

    trk_1.assign(list(evt.trk_i))
    trk_2.assign(list(evt.trk_j))
    deltaR.assign(list(evt.deltaR))
    dca.assign(list(evt.dca))
    dca_sig.assign(list(evt.dca_sig))
    cptopv.assign(list(evt.cptopv))
    pvtoPCA_1.assign(list(evt.pvtoPCA_i))
    pvtoPCA_2.assign(list(evt.pvtoPCA_j))
    dotprod_1.assign(list(evt.dotprod_i))
    dotprod_2.assign(list(evt.dotprod_j))
    pair_mom.assign(list(evt.pair_mom))
    pair_invmass.assign(list(evt.pair_invmass))

    trk_ip2d.assign(list(evt.trk_ip2d))
    trk_ip3d.assign(list(evt.trk_ip3d))
    trk_ip2dsig.assign(list(evt.trk_ip2dsig))
    trk_ip3dsig.assign(list(evt.trk_ip3dsig))
    trk_p.assign(list(evt.trk_p))
    trk_pt.assign(list(evt.trk_pt))
    trk_eta.assign(list(evt.trk_eta))
    trk_phi.assign(list(evt.trk_phi))
    trk_nValid.assign(list(evt.trk_nValid))
    trk_nValidPixel.assign(list(evt.trk_nValidPixel))
    trk_nValidStrip.assign(list(evt.trk_nValidStrip))
    trk_charge.assign(list(evt.trk_charge))
    
    nTrks = evt.nTrks[0]

    #MATCHING SV TRKS TO TRKS
    if(sum(evt.SV_ntrks) > 0):
        alltrk_data = np.array([(evt.trk_pt[i], evt.trk_eta[i], evt.trk_phi[i]) for i in range(nTrks)])
        svtrk_data = np.array([(evt.SVtrk_pt[i], evt.SVtrk_eta[i], evt.SVtrk_phi[i]) for i in range(sum(evt.SV_ntrks))])
        
        pt_diff = np.abs(alltrk_data[:, 0][:, None] - svtrk_data[:, 0])
        eta_diff = np.abs(alltrk_data[:, 1][:, None] - svtrk_data[:, 1])
        phi_diff = np.abs(alltrk_data[:, 2][:, None] - svtrk_data[:, 2])
        
        best_indices = np.argmin(pt_diff + eta_diff + phi_diff, axis=0)
        svtrkinds = set(best_indices)
        
        for ind in svtrkinds:
            SVtrk_ind.push_back(int(ind))

    seltrk_ind =  np.array(evt.trk_i) 
    trk_pt_array = np.array(evt.trk_pt)
    d_pt_array = np.array(evt.Daughters_pt)
    d_flag_array = np.array(evt.Daughters_flag)
    d_flav_array = np.array(evt.Daughters_flav)
    trk_eta_array = np.array(evt.trk_eta)  # Direct slicing
    trk_phi_array = np.array(evt.trk_phi)
    d_eta_array = np.array(evt.Daughters_eta)
    d_phi_array = np.array(evt.Daughters_phi)
    delta_R_matrix = delta_R(trk_eta_array, trk_phi_array, d_eta_array, d_phi_array)
    valid_tracks = (trk_pt_array >= 0.5) & (np.abs(trk_eta_array) < 2.5)
    
    #bkg_mask1 = valid_tracks[:, None] & (delta_R_matrix < 0.02) & (0.6 <= (trk_pt_array[:, None] / d_pt_array)) & ((trk_pt_array[:, None] / d_pt_array) < 0.8)
    #bkg_mask2 = valid_tracks[:, None] & (0.02 < delta_R_matrix) & (delta_R_matrix < 0.1)
    #bkg_mask3 = valid_tracks[:, None] & (0.1 < delta_R_matrix) & (delta_R_matrix < 0.2)
    
    
    min_deltaR_indices = np.argmin(delta_R_matrix, axis=0)
    sig_mask = valid_tracks[min_deltaR_indices] & \
               (delta_R_matrix[min_deltaR_indices, np.arange(delta_R_matrix.shape[1])] < 0.02) & \
               (0.8 <= (trk_pt_array[min_deltaR_indices] / d_pt_array)) & \
               ((trk_pt_array[min_deltaR_indices] / d_pt_array) <= 1.2)
    sig_indices = min_deltaR_indices[sig_mask]
    miss_sig_count = np.sum(~np.isin(sig_indices, seltrk_ind))
    missed_sig.assign([int(miss_sig_count)])
    sig_ind.assign(sig_indices.tolist())
    sig_daughters = np.where(sig_mask)[0]  # Get indices of selected daughters
    sig_flag.assign(d_flag_array[sig_daughters].tolist())  
    sig_flav.assign(d_flav_array[sig_daughters].tolist())

    
    #bkg_indices, bkg_daughters = np.where(bkg_mask1 | bkg_mask2 | bkg_mask3)
    #if len(bkg_indices) > 20:
    #    sampled_indices = np.random.choice(len(bkg_indices), size=20, replace=False)
    #    bkg_indices = bkg_indices[sampled_indices]
    #    bkg_daughters = bkg_daughters[sampled_indices]
    #bkg_flag.assign(d_flag_array[bkg_daughters].tolist())
    #bkg_ind.assign(bkg_indices.tolist())
    
    outtree.Fill()

    run_list.append(int(evt.run))
    lumi_list.append(int(evt.lumi))
    evt_list.append(int(evt.evt))
    sig_ind_list.append(sig_indices.tolist())
    


Outfile.WriteTObject(outtree, "tree")
Outfile.Close()


with open("genmatch_info.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["run", "lumi", "event", "sig_ind"])

    for i in range(len(run_list)):
        sig_str = "[" + ",".join(str(x) for x in sig_ind_list[i]) + "]"
        writer.writerow([run_list[i], lumi_list[i], evt_list[i], sig_str])

