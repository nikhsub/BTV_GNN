#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/BTV/ttbarlep_120files_1103/CRAB_UserFiles/MC_ttbarlep_1103)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 1403_ttbar_lep_dca -ec 3000
