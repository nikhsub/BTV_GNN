from ROOT import *
import matplotlib
#matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt
import math

sigfile = "testingwevtnum_1k_2705_sig.root"
bkgfile = "testingwevtnum_1k_2705_bkg.root"



Sigfile = TFile(sigfile, 'READ')
#Bkgfile = TFile(bkgfile, 'READ')
#tree_dir = Infile.Get('btagana')
sigtree = Sigfile.Get('sigtree')
#bkgtree = Bkgfile.Get('bkgtree')

sig_ip2d = []
bkg_ip2d = []

sig_ip3d = []
bkg_ip3d = []

#print(sigtree.Show(10))
for evt in sigtree:
    print(evt.bhad_evtnum)
#    print(
#for evt in sigtree:
#    for ip2d in evt.ip2d:
#        sig_ip2d.append(ip2d)
#
#    for ip3d in evt.ip3d:
#        sig_ip3d.append(ip3d)
#
##for evt in bkgtree:
##    for ip2d in evt.ip2d:
##        bkg_ip2d.append(ip2d)
##
##    for ip3d in evt.ip3d:
##        bkg_ip3d.append(ip3d)
#
#plt.hist(sig_ip2d, bins=100, color="white",  edgecolor='red', histtype='stepfilled', density=True, label="signal")
##plt.hist(bkg_ip2d, bins=100, color="white", alpha=0.5, edgecolor='blue', histtype='stepfilled', density=True, label="bkg")
#
#plt.legend()
#
#plt.yscale('log')
#plt.show()
#plt.savefig('ip2d.png')
