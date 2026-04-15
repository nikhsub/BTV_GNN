#!/bin/sh

for era in 2016 2017 2018
do
	for fs in 4FS 5FS
	do

		INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/hplusb/hb_fortrain_${era}_HZZ4l_${fs}_3003/HPlusBottom_${fs}_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MC_hplusb${fs}_${era}_HZZ4l_3003)/0000
		
		python condor_analyze.py -i $INPATH -o 3103_hb_${era}_${fs} -ec 3000 -st ${era}_${fs}
	done

done
