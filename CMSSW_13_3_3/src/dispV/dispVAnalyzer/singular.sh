#!/bin/sh
export SCRAM_ARCH=slc7_amd64_gcc11
echo "SCRAM_ARCH is set to $SCRAM_ARCH"
cmsenv
source /cvmfs/cms.cern.ch/common/crab-setup.sh
voms-proxy-init --voms cms
