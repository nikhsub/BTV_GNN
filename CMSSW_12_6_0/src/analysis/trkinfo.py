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
parser.add_argument("-s", "--start", default=0, help="Start index")
parser.add_argument("-e", "--end", default=99, help="End index")


#parser.add_argument("-ob", "--out_bkg", default="testout_bkg", help="Name of output background ROOT file")

args = parser.parse_args()

infile = args.inp

Infile = TFile(infile, 'READ')
demo = Infile.Get('demo')
tree = demo.Get('tree')

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


outtree.Branch("sig_flag", sig_flag)
outtree.Branch("sig_flav", sig_flav)
outtree.Branch("sig_ind", sig_ind)
outtree.Branch("bkg_ind", bkg_ind)
outtree.Branch("bkg_flag", bkg_flag)
outtree.Branch("seed_ind", seed_ind)
outtree.Branch("SVtrk_ind", SVtrk_ind)



#outtree.Branch("chi2", chi2)
#outtree.Branch("nHitAll", nHitAll)
#outtree.Branch("nHitPixel", nHitPixel)
#outtree.Branch("nHitStrip", nHitStrip)  #Creating branches in signal tree
#outtree.Branch("nHitTIB", nHitTIB)
#outtree.Branch("nHitTID", nHitTID)
#outtree.Branch("nHitTOB", nHitTOB)
#outtree.Branch("nHitTEC", nHitTEC)
#outtree.Branch("nHitPXB", nHitPXB)
#outtree.Branch("nHitPXF", nHitPXF)
#outtree.Branch("isHitL1", isHitL1)
#outtree.Branch("nSiLayers", nSiLayers)
#outtree.Branch("nPxLayers", nSiLayers)


def delta_phi(phi1, phi2):
    """
    Calculate the difference in phi between two angles.
    """
    dphi = phi2 - phi1
    while dphi > math.pi:
        dphi -= 2 * math.pi
    while dphi < -math.pi:
        dphi += 2 * math.pi
    return dphi

def delta_eta(eta1, eta2):
    """
    Calculate the difference in eta.
    """
    return eta2 - eta1

def delta_R(eta1, phi1, eta2, phi2):
    """
    Calculate the distance in eta-phi space.
    """
    deta = delta_eta(eta1, eta2)
    dphi = delta_phi(phi1, phi2)
    return math.sqrt(deta**2 + dphi**2)

for i, evt in enumerate(tree):
    if(i >= int(args.start) and i <= int(args.end)-1):
        print("Processing event:", i)
        sig_ind.clear()
        bkg_ind.clear()
        seed_ind.clear()
        sig_flag.clear()
        bkg_flag.clear()
        sig_flav.clear()
        SVtrk_ind.clear()

        #chi2.clear()
        #charge.clear()
        #nHitAll.clear()
        #nHitPixel.clear()
        #nHitStrip.clear()
        #nHitTIB.clear()
        #nHitTID.clear()
        #nHitTOB.clear()
        #nHitTEC.clear()
        #nHitPXB.clear()
        #nHitPXF.clear()
        #isHitL1.clear()
        #nSiLayers.clear()
        #nPxLayers.clear()
       
        #ngvs = 0
        
        
        #for bd in range(len(evt.Hadron_GVx)):
        #    had_GVx.push_back(evt.Hadron_GVx[bd])
        #    had_GVy.push_back(evt.Hadron_GVy[bd])
        #    had_GVz.push_back(evt.Hadron_GVz[bd])
        #    GV_flag.push_back(evt.GV_flag[bd])
        #    ngvs+=1

        nds = sum(evt.nDaughters)

        #t_pt = evt.trk_pt
        #t_ip2d = evt.trk_ip2d
        #t_ip3d = evt.trk_ip3d

        #t_ip2d = [x if -20 < x < 20 else 0 for x in t_ip2d]

        #t_ip3d = [x if -40 < x < 40 else 0 for x in t_ip3d]

        #t_ip2dsig = evt.trk_ip2dsig
        #t_ip3dsig = evt.trk_ip3dsig

        #n = ngvs*3

        #top_indices = top_n_indices([t_ip2d,t_ip3d], n)
        #
        #top_ind.push_back(numpy_to_std_vector(top_indices))

        #svtrkinds = set()

        #for svtrk in range(sum(evt.SV_ntrks)):
        #    mindelpt    = 1e6
        #    mindeleta   = 1e6
        #    mindelphi   = 1e6
        #    bestind     = -1
        #    for alltrk in range(evt.nTrks[0]):
        #        alltrk_pt  = evt.trk_pt[alltrk]
        #        alltrk_eta = evt.trk_eta[alltrk]
        #        alltrk_phi = evt.trk_phi[alltrk]
        #        svtrk_pt   = evt.SVtrk_pt[svtrk]
        #        svtrk_eta  = evt.SVtrk_eta[svtrk]
        #        svtrk_phi  = evt.SVtrk_phi[svtrk]

        #        delpt = abs(alltrk_pt - svtrk_pt)
        #        deleta = abs(alltrk_eta - svtrk_eta)
        #        delphi = abs(alltrk_phi - svtrk_phi)

        #        if(delpt <= mindelpt and deleta <= mindeleta and delphi <= mindelphi):
        #            mindelpt  = delpt
        #            mindeleta = deleta
        #            mindelphi = delphi
        #            bestind   = alltrk

        #    if(bestind != -1): svtrkinds.add(bestind)

        #SVtrk_ind.push_back(list(svtrkinds)
        

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

        tinds = []
         

        if(nds>0):
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
                    delR = delta_R(evt.trk_eta[trk], evt.trk_phi[trk], evt.Daughters_eta[d], evt.Daughters_phi[d]) 
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
                    tinds.append(tind)

        outtree.Fill()


Outfile.WriteTObject(outtree, "tree")
Outfile.Close()
