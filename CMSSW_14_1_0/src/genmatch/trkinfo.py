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
parser.add_argument("-csv", "--write_csv", default=False, action="store_true", help="Write genmatch info to file")

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

run = std.vector('int')()
lumi = std.vector('int')()
event = std.vector('ULong64_t')()

trk_delr     = std.vector('double')()
trk_ptrat     = std.vector('double')()

SVtrk_ind     = std.vector('int')()

trk_ip2d        = std.vector('double')()
trk_ip3d        = std.vector('double')()
trk_dz          = std.vector('double')()
trk_dzsig       = std.vector('double')()
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
trk_label       = std.vector('int')()
trk_hadidx      = std.vector('int')()
trk_flav        = std.vector('int')()

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

    "run": run, "lumi": lumi, "event": event, "had_pt": had_pt,
    "SVtrk_ind": SVtrk_ind, "trk_ip2d": trk_ip2d,
    "trk_ip3d": trk_ip3d, "trk_dz": trk_dz, "trk_dzsig": trk_dzsig, 
    "trk_ip2dsig": trk_ip2dsig, "trk_ip3dsig": trk_ip3dsig, "trk_p": trk_p,
    "trk_pt": trk_pt, "trk_eta": trk_eta, "trk_phi": trk_phi, "trk_nValid": trk_nValid,
    "trk_nValidPixel": trk_nValidPixel, "trk_nValidStrip": trk_nValidStrip, "trk_charge": trk_charge,
    "trk_label": trk_label, "trk_hadidx": trk_hadidx, "trk_flav": trk_flav,
    "missed_sig": missed_sig, "trk_1": trk_1, "trk_2": trk_2, "deltaR": deltaR,
    "dca": dca, "dca_sig": dca_sig, "cptopv": cptopv, "pvtoPCA_1": pvtoPCA_1, "pvtoPCA_2": pvtoPCA_2,
    "dotprod_1": dotprod_1, "dotprod_2": dotprod_2, "pair_mom": pair_mom, "pair_invmass": pair_invmass,
    "preds": preds, "trk_delr": trk_delr, "trk_ptrat": trk_ptrat
}

for name, branch in branches.items():
    outtree.Branch(name, branch)

run_list = []   # or directly append run number per event
lumi_list = []
evt_list = []

trk_label_list = []
trk_hadidx_list = []


#outtree.Branch("delr", delr)
#outtree.Branch("ptrat", ptrat)

def delta_R(eta1, phi1, eta2, phi2):
    """Efficiently compute ΔR using vectorized operations."""
    deta = eta1[:, None] - eta2  # Vectorized subtraction
    dphi = np.abs(phi1[:, None] - phi2)
    dphi[dphi > np.pi] -= 2 * np.pi  # Ensure dphi is in [-π, π]
    return np.sqrt(deta**2 + dphi**2)

def delta_phi(phi1, phi2):
    """Compute Δφ taking into account the periodicity of the angle."""
    dphi = phi1 - phi2
    return np.mod(dphi + np.pi, 2 * np.pi) - np.pi

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

    trk_ip2d.assign(np.abs(np.array(evt.trk_ip2d)).tolist())
    trk_ip3d.assign(np.abs(np.array(evt.trk_ip3d)).tolist())
    trk_dz.assign(list(evt.trk_dz))
    trk_dzsig.assign(list(evt.trk_dzsig))
    trk_ip2dsig.assign(np.abs(np.array(evt.trk_ip2dsig)).tolist())
    trk_ip3dsig.assign(np.abs(np.array(evt.trk_ip3dsig)).tolist())
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
        phi_diff = np.abs(delta_phi(alltrk_data[:, 2][:, None], svtrk_data[:, 2]))
        
        best_indices = np.argmin(pt_diff + eta_diff + phi_diff, axis=0)
        svtrkinds = set(best_indices)
        
        for ind in svtrkinds:
            SVtrk_ind.push_back(int(ind))

    seltrk_ind =  np.array(evt.trk_i) 
    trk_pt_array = np.array(evt.trk_pt)
    d_pt_array = np.array(evt.Daughters_pt)
    trk_eta_array = np.array(evt.trk_eta)  # Direct slicing
    trk_phi_array = np.array(evt.trk_phi)
    d_eta_array = np.array(evt.Daughters_eta)
    d_phi_array = np.array(evt.Daughters_phi)
    d_hadidx_array = np.array(evt.Daughters_hadidx)
    d_flav_array   = np.array(evt.Daughters_flav)
    d_label_array  = np.array(evt.Daughters_label)

    # --- Daughter ↔ Track matching ---
    if(nTrks > 0 and len(d_eta_array) > 0):
        delta_R_matrix = delta_R(trk_eta_array, trk_phi_array, d_eta_array, d_phi_array)
        all_pt_ratios  = trk_pt_array[:, None] / d_pt_array[None, :]
        
        # Apply quality cuts to all pairs
        pair_mask = (trk_pt_array[:, None] >= 0.5) & (np.abs(trk_eta_array)[:, None] < 2.5) & \
                    (delta_R_matrix < 0.02) & (all_pt_ratios >= 0.8) & (all_pt_ratios <= 1.2)
        
        delta_R_masked_T = np.where(pair_mask, delta_R_matrix, np.inf)
        best_daughter_per_track = np.argmin(delta_R_masked_T, axis=1)        # daughter index per track
        best_deltaR_per_track = delta_R_masked_T[np.arange(nTrks), best_daughter_per_track] 
        best_pt_ratio_per_track = trk_pt_array / d_pt_array[best_daughter_per_track]

        trk_label_np   = np.full(nTrks, 6, dtype=int)        # default label = fake
        trk_hadidx_np  = np.full(nTrks, -1, dtype=int)
        trk_flav_np    = np.full(nTrks, -1, dtype=int)
        trk_delr_np    = np.full(nTrks, np.inf, dtype=float)
        trk_ptrat_np   = np.full(nTrks, np.nan, dtype=float)

        matched_tracks = np.isfinite(best_deltaR_per_track)

        trk_label_np[matched_tracks] = d_label_array[best_daughter_per_track[matched_tracks]]
        trk_hadidx_np[matched_tracks] = d_hadidx_array[best_daughter_per_track[matched_tracks]]
        trk_flav_np[matched_tracks]   = d_flav_array[best_daughter_per_track[matched_tracks]]
        trk_delr_np[matched_tracks]   = best_deltaR_per_track[matched_tracks]
        trk_ptrat_np[matched_tracks]  = best_pt_ratio_per_track[matched_tracks]
    else:
        trk_label_np    = np.full(nTrks, 6, dtype=int)
        trk_hadidx_np  = np.full(nTrks, -1, dtype=int)
        trk_flav_np    = np.full(nTrks, -1, dtype=int)
        trk_delr_np    = np.full(nTrks, np.inf, dtype=float)
        trk_ptrat_np   = np.full(nTrks, np.nan, dtype=float)


    # --- Save results to branches ---
    trk_label.assign(trk_label_np.tolist())
    trk_hadidx.assign(trk_hadidx_np.tolist())
    trk_flav.assign(trk_flav_np.tolist())
    trk_delr.assign(trk_delr_np.tolist())
    trk_ptrat.assign(trk_ptrat_np.tolist())
    outtree.Fill()

    run_list.append(int(evt.run))
    lumi_list.append(int(evt.lumi))
    evt_list.append(int(evt.evt))
    trk_label_list.append(trk_label_np.tolist())
    trk_hadidx_list.append(trk_hadidx_np.tolist())
    

Outfile.WriteTObject(outtree, "tree")
Outfile.Close()

if(args.write_csv):
    with open("geninfo_"+args.out+".csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run", "lumi", "event", "trk_labels", "trk_hadidx"])
    
        for i in range(len(run_list)):
            label_str = "[" + ",".join(str(x) for x in trk_label_list[i]) + "]"
            hadidx_str = "[" + ",".join(str(x) for x in trk_hadidx_list[i]) + "]"
            writer.writerow([run_list[i], lumi_list[i], evt_list[i], label_str, hadidx_str])

