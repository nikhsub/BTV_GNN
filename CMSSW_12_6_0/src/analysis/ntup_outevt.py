import ROOT
from collections import Counter

infile = "ntup_0908_dup.root"
Infile = ROOT.TFile(infile, 'READ')

tree = Infile.Get('tree')

for i, evt in enumerate(tree):
    print("EVENT: ", i)
    print("#GVs:", evt.nGV[0])
    print("#Trks:", len(evt.p))
    print("Flag", evt.flag)
    print("Flav", evt.flav)

    GVflag = evt.flag
    GV_hadcounter = Counter(GVflag)
    #
    #dflag = evt.Daughters_flag
    #dcounter = Counter(dflag)

    gvcount_dict = {i+1: GV_hadcounter[i] for i in range(evt.nGV[0])}
    #dcount_dict = {i+1: dcounter[i] for i in range(evt.nGV[0])}

    for category, count in gvcount_dict.items():
        print(f"GV {category}: {count} tracks")


    #for category, count in dcount_dict.items():
    #    print(f"GV {category}: {count} daughters")

    print("")

   
