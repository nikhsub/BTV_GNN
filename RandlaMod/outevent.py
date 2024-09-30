import ROOT
from collections import Counter

infile = "train_2207.root"
Infile = ROOT.TFile(infile, 'READ')

tree = Infile.Get('tree')

for i, evt in enumerate(tree):
    print("EVENT: ", i)
    print("#GVs:", evt.nGV)
    
    trkGV = evt.flag
    counter = Counter(trkGV)

    count_dict = {i: counter[i] for i in range(evt.nGV[0])}

# Print the dictionary in a more readable format
    for category, count in count_dict.items():
        print(f"GV {category}: {count} tracks")

