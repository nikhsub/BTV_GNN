from ROOT import *

infile = "ttbar_2000_2907.root"

Infile = TFile(infile, 'READ')
#tree_dir = Infile.Get('btagana')
demo = Infile.Get('demo')

tree = demo.Get('tree')

print(tree.Show(10))

for evt in tree:
    print(evt)
    print(evt.Hadron_SVx)
    #print(evt.Hadron_SVy)
    #print(evt.Hadron_SVz)
    print("#SV", evt.nSV)
    print("#Hads", evt.nHadrons)
    #print(evt.Daughter1_pt)
    #print(evt.Daughter2_pt)
    
    break;

