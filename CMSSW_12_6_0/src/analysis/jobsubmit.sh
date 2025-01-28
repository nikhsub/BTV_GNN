#!/bin/sh


#INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/BTVH/ttbarlep_1812/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/MC_ttbarlep_1812)/0000

INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/ttbarlep_evttrain_2301/CRAB_UserFiles/MC_ttbarlep_2301/250123_150640/0000


python condor_analyze.py -i $INPATH -o 2301_ttbarlep_evttrain -ec 3000
