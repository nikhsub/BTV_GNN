#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/user/nvenkata/BTV/ttbarhad_50files_0209/CRAB_UserFiles/MC_ttbarhad_0209)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 0209_ttbar_had -ec 3000
