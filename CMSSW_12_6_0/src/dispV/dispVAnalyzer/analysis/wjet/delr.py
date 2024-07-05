from ROOT import *
import sys
import numpy as np
import argparse
#import array
import math
import json

Infile = TFile("test_output.root", 'READ')
demo = Infile.Get('demo')
tree = demo.Get('tree')

checkfile = TFile("checkfilebkg.root", "recreate")
checktree = TTree("tree", "tree")

delr = std.vector('double')()
checktree.Branch("delr", delr)

trkflag_file = 'trkflags.json'

with open(trkflag_file, 'r') as f:
    data = json.load(f)

sigflags = data[0]
bkgflags = data[1]


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
    print("Processing event", i)
    numtrks = evt.nTrks[0]
    numjets = evt.nJets[0]
    delr.clear()
    for jet in range(numjets):
        mindr = 10000
        for trk in range(numtrks):
            if(trk in bkgflags[str(i)]):
                if(delta_R(evt.jet_eta[jet], evt.jet_phi[jet], evt.trk_eta[trk], evt.trk_phi[trk]) < mindr):
                    mindr = delta_R(evt.jet_eta[jet], evt.jet_phi[jet], evt.trk_eta[trk], evt.trk_phi[trk])
        if(not mindr==10000): delr.push_back(mindr)

    checktree.Fill()


checkfile.WriteTObject(checktree, "checktree")
checkfile.Close()
