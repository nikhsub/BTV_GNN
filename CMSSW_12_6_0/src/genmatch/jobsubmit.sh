#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/user/nvenkata/BTV/ttbarhad_50files_2406/CRAB_UserFiles/MC_ttbarhad_2406)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 2406_ttbar_had -ec 3000
