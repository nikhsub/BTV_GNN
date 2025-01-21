from ROOT import *
import sys
import numpy as np
import argparse
#import array
import math
import numpy as np

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
seed_ind      = std.vector('int')()
sig_flag      = std.vector('int')()
sig_flav      = std.vector('int')()

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

outtree.Branch("had_pt", had_pt)
outtree.Branch("sig_flag", sig_flag)
outtree.Branch("sig_flav", sig_flav)
outtree.Branch("sig_ind", sig_ind)
outtree.Branch("bkg_ind", bkg_ind)
outtree.Branch("bkg_flag", bkg_flag)
outtree.Branch("seed_ind", seed_ind)
outtree.Branch("SVtrk_ind", SVtrk_ind)

outtree.Branch("trk_ip2d", trk_ip2d)
outtree.Branch("trk_ip3d", trk_ip3d)
outtree.Branch("trk_ip2dsig", trk_ip2dsig)
outtree.Branch("trk_ip3dsig", trk_ip3dsig)
outtree.Branch("trk_p", trk_p)
outtree.Branch("trk_pt", trk_pt)
outtree.Branch("trk_eta", trk_eta)
outtree.Branch("trk_phi", trk_phi)
outtree.Branch("trk_nValid", trk_nValid)
outtree.Branch("trk_nValidPixel", trk_nValidPixel)
outtree.Branch("trk_nValidStrip", trk_nValidStrip)
outtree.Branch("trk_charge", trk_charge)

def delta_phi(phi1, phi2):
    """
    Calculate the difference in phi between two angles.
    """
    dphi = phi2 - phi1
    return (dphi + np.pi) % (2 * np.pi) - np.pi

def delta_eta(eta1, eta2):
    """
    Calculate the difference in eta.
    """
    return eta2 - eta1

def delta_R(eta1, phi1, eta2, phi2):
    """
    Calculate the distance in eta-phi space.
    """
    deta = delta_eta(eta1[:, None], eta2)
    dphi = delta_phi(phi1[:, None], phi2)
    return np.sqrt(deta**2 + dphi**2)


for i, evt in enumerate(tree):
    if i < start_index:
        continue
    if i >= end_index:
        break

    if(i%1000 ==0): 
        print("EVT", i) 
    
    nhads.clear()
    had_pt.clear()
    sig_ind.clear()
    bkg_ind.clear()
    seed_ind.clear()
    sig_flag.clear()
    bkg_flag.clear()
    sig_flav.clear()
    SVtrk_ind.clear()

    trk_ip2d.clear();
    trk_ip3d.clear();
    trk_ip2dsig.clear();
    trk_ip3dsig.clear();
    trk_p.clear();
    trk_pt.clear();
    trk_eta.clear();
    trk_phi.clear();
    trk_charge.clear();
    trk_nValid.clear();
    trk_nValidPixel.clear();
    trk_nValidStrip.clear();
    
    #low_pt = False
    #for had in range(evt.nHadrons[0]):
    #    print(evt.Hadron_pt[had])
    #    if(evt.Hadron_pt[had] < 50):
    #        low_pt = True
    #if(not low_pt): continue
    
    if(args.lowpt):
        high_pt = np.any(np.array(evt.Hadron_pt) > 20)
        if(high_pt): continue
    
    hads = 0
    for had in range(evt.nHadrons[0]):
        hads+=1
        had_pt.push_back(evt.Hadron_pt[had])

    nhads.push_back(hads)

    for trk in range(evt.nTrks[0]):
        trk_ip2d.push_back(evt.trk_ip2d[trk])
        trk_ip3d.push_back(evt.trk_ip3d[trk])
        trk_ip2dsig.push_back(evt.trk_ip2dsig[trk])
        trk_ip3dsig.push_back(evt.trk_ip3dsig[trk])
        trk_p.push_back(evt.trk_p[trk])
        trk_pt.push_back(evt.trk_pt[trk])
        trk_eta.push_back(evt.trk_eta[trk])
        trk_phi.push_back(evt.trk_phi[trk])
        trk_nValid.push_back(evt.trk_nValid[trk])
        trk_nValidPixel.push_back(evt.trk_nValidPixel[trk])
        trk_nValidStrip.push_back(evt.trk_nValidStrip[trk])
        trk_charge.push_back(evt.trk_charge[trk])

    nds = sum(evt.nDaughters)

    #MATCHING SV TRKS TO TRKS
    if(sum(evt.SV_ntrks) > 0):
        alltrk_data = np.array([(evt.trk_pt[i], evt.trk_eta[i], evt.trk_phi[i]) for i in range(evt.nTrks[0])])
        svtrk_data = np.array([(evt.SVtrk_pt[i], evt.SVtrk_eta[i], evt.SVtrk_phi[i]) for i in range(sum(evt.SV_ntrks))])
        
        pt_diff = np.abs(alltrk_data[:, 0][:, None] - svtrk_data[:, 0])
        eta_diff = np.abs(alltrk_data[:, 1][:, None] - svtrk_data[:, 1])
        phi_diff = np.abs(alltrk_data[:, 2][:, None] - svtrk_data[:, 2])
        
        best_indices = np.argmin(pt_diff + eta_diff + phi_diff, axis=0)
        svtrkinds = set(best_indices)
        
        for ind in svtrkinds:
            SVtrk_ind.push_back(int(ind))

    #tinds = []
     

    if(nds>0):
        trk_eta_array = np.array([evt.trk_eta[trk] for trk in range(evt.nTrks[0])])
        trk_phi_array = np.array([evt.trk_phi[trk] for trk in range(evt.nTrks[0])])
        d_eta_array = np.array([evt.Daughters_eta[d] for d in range(nds)])
        d_phi_array = np.array([evt.Daughters_phi[d] for d in range(nds)])

        delta_R_matrix = delta_R(trk_eta_array, trk_phi_array, d_eta_array, d_phi_array)


        for d in range(nds):
            trk_mindr = 1e6
            trk_flag = -1
            trk_flav = -1
            trk_ptrat = -1
            tind = -1
            bkgcount = 0
            for trk in range(evt.nTrks[0]):
                if(d==0):
                    if(evt.trk_pt[trk] > 0.8 and abs(evt.trk_ip3d[trk]) > 0.005 and abs(evt.trk_ip2dsig[trk]) > 1.2):
                        seed_ind.push_back(trk)

                #if(trk in tinds): continue
                if(evt.trk_charge[trk] != evt.Daughters_charge[d]): continue
                if(not (evt.trk_pt[trk] >= 0.5 and abs(evt.trk_eta[trk]) < 2.5)): continue
                delR = delta_R_matrix[trk, d]
                temp_ptrat = (evt.trk_pt[trk])/(evt.Daughters_pt[d])
                if (delR <= trk_mindr and delR< 0.02 and temp_ptrat > 0.8 and temp_ptrat < 1.2):
                    trk_mindr = delR
                    trk_ptrat = temp_ptrat
                    tind = trk

                elif (bkgcount < 15 and delR >= 0.02 and delR < 0.6):
                    bkg_ind.push_back(trk)
                    bkg_flag.push_back(evt.Daughters_flag[d])
                    bkgcount+=1;
            
            if(trk_ptrat > 0):
                trk_flag = evt.Daughters_flag[d] #Which hadron it comes from
                trk_flav = evt.Daughters_flav[d]
                sig_ind.push_back(tind)
                sig_flag.push_back(trk_flag)
                sig_flav.push_back(trk_flav)
                #tinds.append(tind)

    outtree.Fill()


Outfile.WriteTObject(outtree, "tree")
Outfile.Close()
