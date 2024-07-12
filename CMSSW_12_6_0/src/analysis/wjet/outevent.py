from ROOT import *

infile = "test_output.root"

Infile = TFile(infile, 'READ')
#tree_dir = Infile.Get('btagana')
demo = Infile.Get('demo')

tree = demo.Get('tree')
print(tree.GetEntries())

print(tree.Show(10))

for evt in tree:
    print(evt.nJets)
    print(evt.jet_eta)
    break;

