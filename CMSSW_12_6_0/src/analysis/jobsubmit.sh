#!/bin/sh

INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/BTVH/ttbarhad_0911/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/MC_ttbarhad_0911)/0000

python condor_analyze.py -i $INPATH -o 1011_ttbarhad -ec 3000
