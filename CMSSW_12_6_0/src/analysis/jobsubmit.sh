#!/bin/sh

INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/BTVH/ttbarhad_0711/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MC_ttbarhad_0711)/0000

python condor_analyze.py -i $INPATH -o 0811_ttbarhad -p 5
