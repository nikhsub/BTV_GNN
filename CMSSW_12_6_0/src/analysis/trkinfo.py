from ROOT import *
import sys
import numpy as np
import argparse
#import array
import math

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

Outfile = TFile(args.out+"_out.root", "recreate")
outtree = TTree("tree", "tree")


ip2d      = std.vector('double')()
ip3d      = std.vector('double')()
ip2dsig   = std.vector('double')()
ip3dsig   = std.vector('double')()
p         = std.vector('double')()
pt        = std.vector('double')()
eta       = std.vector('double')()
phi       = std.vector('double')()
charge    = std.vector('double')()

#chi2_bkg      = std.vector('int')()
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

had_GVx  = std.vector('double')()
had_GVy  = std.vector('double')()
had_GVz  = std.vector('double')()

outtree.Branch("had_GVx", had_GVx)
outtree.Branch("had_GVy", had_GVy)
outtree.Branch("had_GVz", had_GVz)

outtree.Branch("ip2d", ip2d)
outtree.Branch("ip3d", ip3d)
outtree.Branch("ip2dsig", ip2dsig)
outtree.Branch("ip3dsig", ip3dsig)
outtree.Branch("p", p)
outtree.Branch("pt", pt)
outtree.Branch("eta", eta)
outtree.Branch("phi", phi)
outtree.Branch("charge", charge)

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
        ip2d.clear()
        ip3d.clear()
        ip2dsig.clear()
        ip3dsig.clear()
        p.clear()
        pt.clear()
        eta.clear()
        phi.clear()
        charge.clear()
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
        had_GVx.clear()
        had_GVy.clear()
        had_GVz.clear()

        for bd in range(len(evt.Hadron_GVx)):
            had_GVx.push_back(evt.Hadron_GVx[bd])
            had_GVy.push_back(evt.Hadron_GVy[bd])
            had_GVz.push_back(evt.Hadron_GVz[bd])


        for trk in range(evt.nTrks[0]):
            ip2d.push_back(evt.trk_ip2d[trk])
            ip3d.push_back(evt.trk_ip3d[trk])
            ip2dsig.push_back(evt.trk_ip2dsig[trk])
            ip3dsig.push_back(evt.trk_ip3dsig[trk])
            #p.push_back(evt.trk_p[trk])
            pt.push_back(evt.trk_pt[trk])
            eta.push_back(evt.trk_eta[trk])
            phi.push_back(evt.trk_phi[trk])
            charge.push_back(evt.trk_charge[trk])

        outtree.Fill()


Outfile.WriteTObject(outtree, "tree")
Outfile.Close()
