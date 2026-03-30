#!/bin/sh
  

INPATH=/eos/uscms/store/user/nvenkata/BTV/ttbarhad_toproc_1703
MAX_FILES=1000

python condor_analyze.py -i $INPATH -o 1903_ttbarhadproc -st condor --max-files $MAX_FILES
