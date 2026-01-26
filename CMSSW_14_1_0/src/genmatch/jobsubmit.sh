#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/user/nvenkata/BTV/ttbarhad_70files_1101/CRAB_UserFiles/MC_ttbarhad_1101)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 1201_ttbar_had -ec 3000
