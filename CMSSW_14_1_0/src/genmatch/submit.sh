#!/bin/bash
set -e

source /cvmfs/cms.cern.ch/cmsset_default.sh

export SCRAM_ARCH=el8_amd64_gcc10

scram p CMSSW CMSSW_12_6_0
cd CMSSW_12_6_0/src
eval "$(scramv1 runtime -sh)"

[ -n "$ROOTSYS" ] && [ -f "$ROOTSYS/bin/thisroot.sh" ] && source "$ROOTSYS/bin/thisroot.sh"

cd -

EOS_PREFIX="root://cmseos.fnal.gov"
EOS_OUT_DIR="/store/user/nvenkata/hplusb/toproc_3003"

OUTFILE="$3.root"

# run job
python3 $1 -i $2 -o $3 -s $4 -e $5

# copy to EOS
#xrdfs cmseos.fnal.gov mkdir -p $EOS_OUT_DIR
xrdcp -f "$OUTFILE" "${EOS_PREFIX}/${EOS_OUT_DIR}/$(basename "$OUTFILE")"
echo "Copied successfully to EOS."

# cleanup
rm -f "$OUTFILE"
echo "Removed local output file."
