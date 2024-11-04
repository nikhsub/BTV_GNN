from ROOT import *

infile = "ttbar_had_6000_3010_training.root"

Infile = TFile(infile, 'READ')
#tree_dir = Infile.Get('btagana')
demo = Infile.Get('demo')

tree = demo.Get('tree')

print(tree.Show(10))

for evt in tree:
    print(evt)
    #print(evt.Hadron_SVx)
    #print(evt.Hadron_SVy)
    #print(evt.Hadron_SVz)
    #print("#GV", evt.nGV)
    #print("#Hads", evt.nHadrons)
    #print(evt.nDaughters)
    #print(evt.Daughters_flag)
    print("#SV", evt.nSVs)
    print("#trksperSV", evt.SV_ntrks)
    print("SVtrk_eta", len(evt.SVtrk_eta))
    #print(evt.Daughter2_pt)
    
    #break

