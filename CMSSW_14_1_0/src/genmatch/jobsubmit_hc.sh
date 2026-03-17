#!/bin/sh

for era in 2016 2017 2018
do
	for fs in 3FS 4FS
	do

		INPATH=/eos/uscms$(xrdfsls /store/group/lpcljm/nvenkata/hplusc/hc_fortrain_${era}_HZZ4l_${fs}_0403/HPlusCharm_${fs}_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MC_hplusc${fs}_${era}_HZZ4l_0403)/0000
		
		python condor_analyze.py -i $INPATH -o 0503_hc_${era}_${fs} -ec 3000 -st ${era}_${fs}
	done

done
