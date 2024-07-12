from ROOT import *
gStyle.SetOptStat(0)
gROOT.SetBatch(1)

tfile = TFile("test_output.root", 'READ')
demo = tfile.Get('demo')
tree = demo.Get('tree')

nHad = TH1F("nHad", "nHad", 18, 0, 18)
GNNp_b = TH1F("GNNp_s", "GNNp_s", 400, 0, 1)
GNNd_s = TH1F("GNNd_s", "GNNd_s", 400, 0, 1)
GNNd_b = TH1F("GNNd_b", "GNNd_b", 400, 0, 1)
nsig = sig_tree.GetEntries()
nbkg = bkg_tree.GetEntries()

for event in sig_tree:
    if(event.evt%2==0): continue #Validation only on odd evt number
    GNNp_s.Fill(event.SVDNNp)
    GNNd_s.Fill(event.SVDNNd)

for event in bkg_tree:
    if(event.evt%2==0): continue #Validation only on odd evt number
    GNNp_b.Fill(event.SVDNNp)
    GNNd_b.Fill(event.SVDNNd)

gra_ROC_p = TGraph()
gra_ROC_p.GetXaxis().SetTitle("Signal Efficiency")
gra_ROC_p.GetYaxis().SetTitle("Background Efficiency")
gra_ROC_p.SetLineColor(kRed)
gra_ROC_p.GetXaxis().SetRangeUser(0, 1)
gra_ROC_p.GetYaxis().SetRangeUser(1e-6, 1)
gra_ROC_p.SetLineWidth(2)

gra_ROC_d = TGraph()
gra_ROC_d.GetXaxis().SetTitle("Signal Efficiency")
gra_ROC_d.GetYaxis().SetTitle("Background Efficiency")
gra_ROC_d.SetLineColor(kBlue)
gra_ROC_d.GetXaxis().SetRangeUser(0, 1)
gra_ROC_d.GetYaxis().SetRangeUser(1e-6, 1)
gra_ROC_d.SetLineWidth(2)


ip= 0
bkg_eff_p = 1.
sig_eff_p = 1.
bkg_eff_d = 1.
sig_eff_d = 1.

aucscore_p = 0
aucscore_d = 0
for ibin in range(401):
    gra_ROC_p.SetPoint(ip, sig_eff_p, bkg_eff_p)
    gra_ROC_d.SetPoint(ip, sig_eff_d, bkg_eff_d)
    bkg_eff_p -= float(GNNp_b.GetBinContent(ibin))/float(GNNp_b.Integral())
    sig_eff_p -= float(GNNp_s.GetBinContent(ibin))/float(GNNp_s.Integral())
    bkg_eff_d -= float(GNNd_b.GetBinContent(ibin))/float(GNNd_b.Integral())
    sig_eff_d -= float(GNNd_s.GetBinContent(ibin))/float(GNNd_s.Integral())
    aucscore_p += float(sig_eff_p)*float(GNNp_b.GetBinContent(ibin))/float(GNNp_b.Integral())
    aucscore_d += float(sig_eff_d)*float(GNNd_b.GetBinContent(ibin))/float(GNNd_b.Integral())
    ip+=1


c = TCanvas()
if(args.log): c.SetLogy()
x1=.15
y1=.7
x2=x1+.23
y2=y1+.175
leg = TLegend(x1,y1,x2,y2)
leg.AddEntry(gra_ROC_p, "Prompt Tagger", "l")
leg.AddEntry(gra_ROC_d, "Displaced Tagger", "l")

if(args.save):
        saveFile = TFile("saveROC_"+args.tag+".root", "RECREATE")
        saveFile.WriteObject(gra_ROC_p, "prompt")
        saveFile.WriteObject(gra_ROC_d, "disp")

gra_ROC_p.SetTitle("ROC curves "+str((args.tag).split("_")[0]))
gra_ROC_p.Draw()
gra_ROC_d.Draw("SAME")
leg.Draw()
c.SaveAs(args.outFile)
print("Prompt Tagger AUC Score: ", aucscore_p)
print("Displaced Tagger AUC Score: ", aucscore_d)
