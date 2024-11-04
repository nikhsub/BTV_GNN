#!/bin/sh

source /cvmfs/cms.cern.ch/cmsset_default.sh
export PATH=$ROOTSYS/bin:$PATH
current=$PWD
echo $current
export SCRAM_ARCH=el8_amd64_gcc10
scram p CMSSW CMSSW_12_6_0
cd CMSSW_12_6_0/src
eval `scramv1 runtime -sh` # cmsenv is an alias not on the workers
echo $CMSSW_BASE "is the CMSSW we created on the local worker node"
source $ROOTSYS/bin/thisroot.sh
cd $current

python3 $1 -i $2 -o $3 -s $4 -e $5


