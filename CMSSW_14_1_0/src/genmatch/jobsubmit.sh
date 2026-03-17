#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/user/nvenkata/BTV/ttbarhad_70files_1603/CRAB_UserFiles/MC_ttbarhad_1603)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 1703_ttbar_had -ec 3000
