import ROOT
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.lines import Line2D

infile1 = "data_2l_4000_1609.root"
Infile1 = ROOT.TFile(infile1, 'READ')
demo = Infile1.Get('demo')
tree1 = demo.Get('tree')

infile2 = "label_2l_1609_mindr_tinds.root"
Infile2 = ROOT.TFile(infile2, 'READ')
tree2 = Infile2.Get('tree')

max_sflag = 10  # Adjust this based on the expected number of unique sflag values

# Use a colormap for different sflag values
cmap = cm.get_cmap('tab10', max_sflag)  # 'tab10' is a colormap with 10 distinct colors


inds = []
flags = []
flavs = []
flavtohad = {1:'b', 2:'d'}
dflavtocolor = {1:'black', 2:'red'}


for j, evt2 in enumerate(tree2):
    index = evt2.ind
    flag = evt2.flag
    flav = evt2.flav
    index_c = index[:]
    flag_c = flag[:]
    flav_c = flav[:]
    inds.append(index_c)
    flags.append(flag_c)
    flavs.append(flav_c)


for i, evt in enumerate(tree1):
    if(i>49): break
    print("EVT:", i)
    sind = inds[i]
    sflag = flags[i]
    sflav = flavs[i]
    if(len(sind) == 0): continue
    deta = evt.Daughters_eta
    dphi = evt.Daughters_phi
    dflag = evt.Daughters_flav

    numd = len(deta)
    numt = len(sind)
    ngv = evt.nGV
    ngv_b = evt.nGV_B
    ngv_d = evt.nGV_D
    teta = evt.trk_eta
    tphi = evt.trk_phi
    seta = [teta[x] for x in sind]
    sphi = [tphi[x] for x in sind]

    unique_sflag = np.unique(sflag)
    sflag_to_color = {val: cmap(i) for i, val in enumerate(unique_sflag)}

    plt.figure(figsize=(10, 10))

    for j in range(numd):
        color = dflavtocolor.get(dflag[j], 'green')
        plt.scatter(deta[j], dphi[j], color=color, marker='X', s=100)
    #plt.scatter(teta, tphi, color='pink', label=f'Tracks', marker='x)

    for idx, s in enumerate(sflag):
        plt.scatter(seta[idx], sphi[idx], color=sflag_to_color[s], label=f'Matched track {idx} from {flavtohad[sflav[idx]]}', marker='o')


    daugb = Line2D([0], [0], color='black', marker='X', markersize=10, linestyle='', label='B Daughter')
    daugd = Line2D([0], [0], color='red', marker='X', markersize=10, linestyle='', label='D Daughter')

    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([daugb, daugd])
    labels.extend(['B Daughter', 'D Daughter'])


    plt.xlabel('Eta')
    plt.ylabel('Phi')
    plt.title(f'Event {i}: Eta-Phi Distribution,  (GV={ngv}, ngv_b={ngv_b}, ngv_d={ngv_d}, daug={numd}')
    plt.legend(handles=handles, labels=labels, loc='best')
    plt.grid(True)
    plt.savefig(f'event_{i}_eta_phi.png')
    plt.close()

