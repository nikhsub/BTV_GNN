from ROOT import *
import sys
import numpy as np
import argparse
#import array
import math

parser = argparse.ArgumentParser("Create track information root file")

parser.add_argument("-i", "--inp", default="test_ntuple.root", help="Input root file")
parser.add_argument("-o", "--out", default="testout", help="Name of output ROOT file")
parser.add_argument("-s", "--start", default=0, help="Start index")
parser.add_argument("-e", "--end", default=99, help="End index")


#parser.add_argument("-ob", "--out_bkg", default="testout_bkg", help="Name of output background ROOT file")

args = parser.parse_args()

infile = args.inp

Infile = TFile(infile, 'READ')
demo = Infile.Get('demo')
tree = demo.Get('tree')

Outfile_sig = TFile(args.out+"_sig.root", "recreate")
outtree_sig = TTree("tree", "tree")

Outfile_bkg = TFile(args.out+"_bkg.root", "recreate")
outtree_bkg = TTree("tree", "tree")

bhad_conedist = 1.5
genmatch_deleta = 0.02
genmatch_delphi = 0.02
genmatch_delr = 0.03

ip2d_sig      = std.vector('double')()
ip3d_sig      = std.vector('double')()
ip2dsig_sig   = std.vector('double')()
ip3dsig_sig   = std.vector('double')()
p_sig         = std.vector('double')()
pt_sig        = std.vector('double')()
eta_sig       = std.vector('double')()
phi_sig       = std.vector('double')()
#chi2_sig      = std.vector('int')()
#charge_sig    = std.vector('int')()
#nHitAll_sig   = std.vector('int')()
#nHitPixel_sig = std.vector('int')()
#nHitStrip_sig = std.vector('int')() #Signal Track variables
#nHitTIB_sig     = std.vector('int')()
#nHitTID_sig     = std.vector('int')()
#nHitTOB_sig     = std.vector('int')()
#nHitTEC_sig     = std.vector('int')()
#nHitPXB_sig     = std.vector('int')()
#nHitPXF_sig     = std.vector('int')()
#isHitL1_sig     = std.vector('int')()
#nSiLayers_sig   = std.vector('int')()
#nPxLayers_sig   = std.vector('int')()

ip2d_bkg      = std.vector('double')()
ip3d_bkg      = std.vector('double')()
ip2dsig_bkg   = std.vector('double')()
ip3dsig_bkg   = std.vector('double')()
p_bkg         = std.vector('double')()
pt_bkg        = std.vector('double')()
eta_bkg       = std.vector('double')()
phi_bkg       = std.vector('double')()
#chi2_bkg      = std.vector('int')()
#charge_bkg    = std.vector('int')()
#nHitAll_bkg   = std.vector('int')()
#nHitPixel_bkg = std.vector('int')()
#nHitStrip_bkg = std.vector('int')() # Background Track variables
#nHitTIB_bkg     = std.vector('int')()
#nHitTID_bkg     = std.vector('int')()
#nHitTOB_bkg     = std.vector('int')()
#nHitTEC_bkg     = std.vector('int')()
#nHitPXB_bkg     = std.vector('int')()
#nHitPXF_bkg     = std.vector('int')()
#isHitL1_bkg     = std.vector('int')()
#nSiLayers_bkg   = std.vector('int')()
#nPxLayers_bkg   = std.vector('int')()

bhad_pt   = std.vector('double')()
bhad_eta  = std.vector('double')()
bhad_phi  = std.vector('double')()
#bhad_mass = std.vector('double')()
bhad_SVx  = std.vector('double')()
bhad_SVy  = std.vector('double')()
bhad_SVz  = std.vector('double')()
bhad_evtnum  = std.vector('int')()#BHadron variables

outtree_sig.Branch("bhad_pt", bhad_pt)
outtree_sig.Branch("bhad_eta", bhad_eta)
outtree_sig.Branch("bhad_phi", bhad_phi)
#outtree_sig.Branch("bhad_mass", bhad_mass)
outtree_sig.Branch("bhad_SVx", bhad_SVx)
outtree_sig.Branch("bhad_SVy", bhad_SVy)
outtree_sig.Branch("bhad_SVz", bhad_SVz)
outtree_sig.Branch("bhad_evtnum", bhad_evtnum)

outtree_sig.Branch("ip2d", ip2d_sig)
outtree_sig.Branch("ip3d", ip3d_sig)
outtree_sig.Branch("ip2dsig", ip2dsig_sig)
outtree_sig.Branch("ip3dsig", ip3dsig_sig)
#outtree_sig.Branch("p", p_sig)
outtree_sig.Branch("pt", pt_sig)
outtree_sig.Branch("eta", eta_sig)
outtree_sig.Branch("phi", phi_sig)
#outtree_sig.Branch("chi2", chi2_sig)
#outtree_sig.Branch("charge", charge_sig)
#outtree_sig.Branch("nHitAll", nHitAll_sig)
#outtree_sig.Branch("nHitPixel", nHitPixel_sig)
#outtree_sig.Branch("nHitStrip", nHitStrip_sig)  #Creating branches in signal tree
#outtree_sig.Branch("nHitTIB", nHitTIB_sig)
#outtree_sig.Branch("nHitTID", nHitTID_sig)
#outtree_sig.Branch("nHitTOB", nHitTOB_sig)
#outtree_sig.Branch("nHitTEC", nHitTEC_sig)
#outtree_sig.Branch("nHitPXB", nHitPXB_sig)
#outtree_sig.Branch("nHitPXF", nHitPXF_sig)
#outtree_sig.Branch("isHitL1", isHitL1_sig)
#outtree_sig.Branch("nSiLayers", nSiLayers_sig)
#outtree_sig.Branch("nPxLayers", nSiLayers_sig)

outtree_bkg.Branch("bhad_pt", bhad_pt)
outtree_bkg.Branch("bhad_eta", bhad_eta)
outtree_bkg.Branch("bhad_phi", bhad_phi)
#outtree_bkg.Branch("bhad_mass", bhad_mass)
outtree_bkg.Branch("bhad_SVx", bhad_SVx)
outtree_bkg.Branch("bhad_SVy", bhad_SVy)
outtree_bkg.Branch("bhad_SVz", bhad_SVz)
outtree_bkg.Branch("bhad_evtnum", bhad_evtnum)

outtree_bkg.Branch("ip2d", ip2d_bkg)
outtree_bkg.Branch("ip3d", ip3d_bkg)
outtree_bkg.Branch("ip2dsig", ip2dsig_bkg)
outtree_bkg.Branch("ip3dsig", ip3dsig_bkg)
#outtree_bkg.Branch("p", p_bkg)
outtree_bkg.Branch("pt", pt_bkg)
outtree_bkg.Branch("eta", eta_bkg)
outtree_bkg.Branch("phi", phi_bkg)
#outtree_bkg.Branch("chi2", chi2_bkg)
#outtree_bkg.Branch("charge", charge_bkg)
#outtree_bkg.Branch("nHitAll", nHitAll_bkg)
#outtree_bkg.Branch("nHitPixel", nHitPixel_bkg)
#outtree_bkg.Branch("nHitStrip", nHitStrip_bkg) #Creating branches in background tree
#outtree_bkg.Branch("nHitTIB", nHitTIB_bkg)
#outtree_bkg.Branch("nHitTID", nHitTID_bkg)
#outtree_bkg.Branch("nHitTOB", nHitTOB_bkg)
#outtree_bkg.Branch("nHitTEC", nHitTEC_bkg)
#outtree_bkg.Branch("nHitPXB", nHitPXB_bkg)
#outtree_bkg.Branch("nHitPXF", nHitPXF_bkg)
#outtree_bkg.Branch("isHitL1", isHitL1_bkg)
#outtree_bkg.Branch("nSiLayers", nSiLayers_bkg)
#outtree_bkg.Branch("nPxLayers", nPxLayers_bkg)


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
        bd_ind = 0
        numtrks = evt.nTrks[0]
        for bhad in range(evt.nBHadrons[0]):
            #print("Bhad",bhad)
            #print("Dind",bd_ind)
            #print("D2ind",bd2_ind)
            #print("NEXT")
            ip2d_sig.clear()
            ip3d_sig.clear()
            ip2dsig_sig.clear()
            ip3dsig_sig.clear()
            p_sig.clear()
            pt_sig.clear()
            eta_sig.clear()
            phi_sig.clear()
            #chi2_sig.clear()
            #charge_sig.clear()
            #nHitAll_sig.clear()
            #nHitPixel_sig.clear()
            #nHitStrip_sig.clear()
            #nHitTIB_sig.clear()
            #nHitTID_sig.clear()
            #nHitTOB_sig.clear()
            #nHitTEC_sig.clear()
            #nHitPXB_sig.clear()
            #nHitPXF_sig.clear()
            #isHitL1_sig.clear()
            #nSiLayers_sig.clear()
            #nPxLayers_sig.clear()
            ip2d_bkg.clear()
            ip3d_bkg.clear()
            ip2dsig_bkg.clear()
            ip3dsig_bkg.clear()
            p_bkg.clear()
            pt_bkg.clear()
            eta_bkg.clear()
            phi_bkg.clear()
            #chi2_bkg.clear()
            #charge_bkg.clear()
            #nHitAll_bkg.clear()
            #nHitPixel_bkg.clear()
            #nHitStrip_bkg.clear()
            #nHitTIB_bkg.clear()
            #nHitTID_bkg.clear()
            #nHitTOB_bkg.clear()
            #nHitTEC_bkg.clear()
            #nHitPXB_bkg.clear()
            #nHitPXF_bkg.clear()
            #isHitL1_bkg.clear()
            #nSiLayers_bkg.clear()
            #nPxLayers_bkg.clear()

            bhad_pt.clear()
            bhad_eta.clear()
            bhad_phi.clear()
            #bhad_mass.clear()
            bhad_SVx.clear()
            bhad_SVy.clear()
            bhad_SVz.clear()
            bhad_evtnum.clear()

            bhad_pt.push_back(evt.BHadron_pt[bhad])
            bhad_eta.push_back(evt.BHadron_eta[bhad])
            bhad_phi.push_back(evt.BHadron_phi[bhad])
            bhad_evtnum.push_back(i)
            #bhad_mass.push_back(evt.BHadron_mass[bhad])
            #bhad_SVx.push_back(evt.BHadron_SVx[bhad])
            #bhad_SVy.push_back(evt.BHadron_SVy[bhad])
            #bhad_SVz.push_back(evt.BHadron_SVz[bhad])

            start_bd = bd_ind
            end_bd = start_bd + evt.nBDaughters[bhad]
            bd_ind += evt.nBDaughters[bhad]

            for bd in range(start_bd, end_bd):
                bhad_SVx.push_back(evt.BHadron_SVx[bd])
                bhad_SVy.push_back(evt.BHadron_SVy[bd])
                bhad_SVz.push_back(evt.BHadron_SVz[bd])

            #bd2_ind = sum(evt.BHadron_nDaughters2[:bd_ind])

            for trk in range(numtrks):
                #bd_ind = 0
                #bd2_ind = 0
                #start_bd = bd_ind
                #end_bd = start_bd + evt.BHadron_nDaughters[bhad]
                #bd_ind += evt.BHadron_nDaughters[bhad]
                #print(start_bd, end_bd)
                for d in range(start_bd, end_bd):
                    try:
                        ptrat = (evt.trk_pt[trk])/(evt.BDaughters_pt[d])
                    except:
                        print("Divide by zero, continuing")
                        continue

                    if(delta_R(evt.BDaughters_eta[d], evt.BDaughters_phi[d], evt.trk_eta[trk], evt.trk_phi[trk]) <= genmatch_delr and
                            (ptrat >= 0.8 and ptrat <= 1.2)):
                        ip2d_sig.push_back(evt.trk_ip2d[trk])
                        ip3d_sig.push_back(evt.trk_ip3d[trk])
                        ip2dsig_sig.push_back(evt.trk_ip2dsig[trk])
                        ip3dsig_sig.push_back(evt.trk_ip3dsig[trk])
                        #p_sig.push_back(evt.trk_p[trk])
                        pt_sig.push_back(evt.trk_pt[trk])
                        eta_sig.push_back(evt.trk_eta[trk])
                        phi_sig.push_back(evt.trk_phi[trk])
                        break
                    else:
                        ip2d_bkg.push_back(evt.trk_ip2d[trk])
                        ip3d_bkg.push_back(evt.trk_ip3d[trk])
                        ip2dsig_bkg.push_back(evt.trk_ip2dsig[trk])
                        ip3dsig_bkg.push_back(evt.trk_ip3dsig[trk])
                        #p_bkg.push_back(evt.trk_p[trk])
                        pt_bkg.push_back(evt.trk_pt[trk])
                        eta_bkg.push_back(evt.trk_eta[trk])
                        phi_bkg.push_back(evt.trk_phi[trk])

            outtree_sig.Fill()
            outtree_bkg.Fill()


Outfile_sig.WriteTObject(outtree_sig, "sigtree")
Outfile_bkg.WriteTObject(outtree_bkg, "bkgtree")
Outfile_sig.Close()
Outfile_bkg.Close()
