#!/bin/sh
  

INPATH=/eos/uscms/store/user/nvenkata/hplusb/toproc_3003
MAX_FILES=2000

python condor_analyze.py -i $INPATH -o 3103_hplusb -st condor --max-files $MAX_FILES
