#!/bin/sh


INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/BTVH/ttbarlep_120files_0703/CRAB_UserFiles/MC_ttbarlep_0703)/0000

#INPATH=/eos/uscms/store/group/lpcljm/nvenkata/BTVH/files


python condor_analyze.py -i $INPATH -o 0703_ttbar_lep_120files -ec 3000
