import ROOT
from collections import Counter

infile = "ttbar_had_100_1612.root"
Infile = ROOT.TFile(infile, 'READ')

demo = Infile.Get('demo')
tree = demo.Get('tree')

for i, evt in enumerate(tree):
    print("EVENT: ", i)
    #print("#Hads:", evt.nHadrons) 
    #print("#GVs:", evt.nGV)
    #print("#daugs:", evt.nDaughters)
    #print("daug pt", evt.Daughters_pt)
    #print("daug flag", evt.Daughters_flag)
    #print("ntrks", evt.nTrks)
    #print("trk pt", evt.trk_pt)
    for j in range(len(evt.SV_x)):
        x_SV = evt.SV_x[j]
        y_SV = evt.SV_y[j]
        z_SV = evt.SV_z[j]

        print(f"SV x = {x_SV}, y = {y_SV}, z = {z_SV}")

    #GVflag = evt.GV_flag
    #GV_hadcounter = Counter(GVflag)
    ##
    ##dflag = evt.Daughters_flag
    ##dcounter = Counter(dflag)

    #gvcount_dict = {i+1: GV_hadcounter[i] for i in range(evt.nHadrons[0])}
    ##dcount_dict = {i+1: dcounter[i] for i in range(evt.nGV[0])}

    #for category, count in gvcount_dict.items():
    #    print(f"Hadron {category}: {count} GVs")


    ##for category, count in dcount_dict.items():
    ##    print(f"GV {category}: {count} daughters")

    #print("")

   
