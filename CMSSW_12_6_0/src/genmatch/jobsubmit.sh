#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/user/nvenkata/BTV/ttbarlep_120files_0104/CRAB_UserFiles/MC_ttbarlep_0104)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 0204_ttbar_lep_edge -ec 3000
