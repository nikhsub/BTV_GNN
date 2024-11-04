import ROOT
from collections import Counter

infile = "testcondorlabel.root"
Infile = ROOT.TFile(infile, 'READ')

tree = Infile.Get('tree')

for i, evt in enumerate(tree):
    print("EVENT: ", i)
    #print("#GVs:", evt.nGV[0])
    #print("#Trks:", len(evt.p))
    print("Sig Ind", set(evt.sig_ind))
    print("SV trk ind", evt.SVtrk_ind)
#    print("Seed_ind", evt.seed_ind)
    #set_ind = set(evt.ind)
    #set_seed_ind = set(evt.seed_ind)

    # Find common elements
    #common_elements = set_ind.intersection(set_seed_ind)
    
    # Print common elements
    #print("Diff Common elements:", len(set(evt.ind))-len(common_elements))

    #GVflag = evt.flag
    #GV_hadcounter = Counter(GVflag)
    ##
    ##dflag = evt.Daughters_flag
    ##dcounter = Counter(dflag)

    #gvcount_dict = {i+1: GV_hadcounter[i] for i in range(evt.nGV[0])}
    ##dcount_dict = {i+1: dcounter[i] for i in range(evt.nGV[0])}

    #for category, count in gvcount_dict.items():
    #    print(f"GV {category}: {count} tracks")


    #for category, count in dcount_dict.items():
    #    print(f"GV {category}: {count} daughters")

    print("")

   
