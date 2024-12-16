#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/BTVH/ttbarlep_0512/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MC_ttbarlep_0512)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 0512_ttbarlep -ec 3000
