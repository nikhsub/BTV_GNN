#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/user/nvenkata/BTV/ttbarhad_50files_0911/CRAB_UserFiles/MC_ttbarhad_0911)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 1111_ttbar_had -ec 3000
