from ROOT import *
import argparse
gStyle.SetOptStat(0)
gROOT.SetBatch(1)

infile = "testout_1507.root"
Infile = TFile(infile, 'READ')
demo = Infile.Get('demo')
tree = demo.Get('tree')

nGV = TH1F("nGV", "", 14, 0, 14)
nGV_B = TH1F("nGV_B", "", 14, 0, 14)
nGV_D = TH1F("nGV_D", "", 14, 0, 14)
nHadrons = TH1F("nHadrons", "", 14, 0, 14)
nD    = TH1F("nD", "", 14, 0, 14)
nD_B    = TH1F("nD_B", "", 14, 0, 14)
nD_D    = TH1F("nD_D", "", 14, 0, 14)


nentries = tree.GetEntries()

for event in tree:
    nGV.Fill(event.nGV[0])
    nGV_B.Fill(event.nGV_B[0])
    nGV_D.Fill(event.nGV_D[0])
    nHadrons.Fill(event.nHadrons[0])
    if(len(event.nDaughters) >0):
        for d in event.nDaughters:
            nD.Fill(d)

    if(len(event.nDaughters_B) >0):
        for d in event.nDaughters_B:
            if(d>0): nD_B.Fill(d)
    
    if(len(event.nDaughters_D) >0):
        for d in event.nDaughters_D:
            if(d>0): nD_D.Fill(d)


saveFile = TFile("hists_TTToHadronic_2018UL_merged.root", "RECREATE")
saveFile.WriteObject(nGV, "nGV")
saveFile.WriteObject(nGV_B, "nGV_B")
saveFile.WriteObject(nGV_D, "nGV_D")
saveFile.WriteObject(nHadrons, "nHadrons")
saveFile.WriteObject(nD, "nD")
saveFile.WriteObject(nD_B, "nD_B")
saveFile.WriteObject(nD_D, "nD_D")

