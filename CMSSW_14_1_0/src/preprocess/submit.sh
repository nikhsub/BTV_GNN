#!/bin/bash
set -e

source /cvmfs/cms.cern.ch/cmsset_default.sh

current="$PWD"
echo "Working directory: $current"

export SCRAM_ARCH=el8_amd64_gcc10

scram p CMSSW CMSSW_12_6_0
cd CMSSW_12_6_0/src
eval "$(scramv1 runtime -sh)"
echo "$CMSSW_BASE is the CMSSW we created on the local worker node"

if [ -n "${ROOTSYS:-}" ] && [ -f "$ROOTSYS/bin/thisroot.sh" ]; then
    source "$ROOTSYS/bin/thisroot.sh"
fi

cd "$current"

EOS_PREFIX="root://cmseos.fnal.gov"
EOS_OUT_DIR="/store/user/nvenkata/BTV/fortrain_ttbarhad_root_1903_redo/"

LOCAL_OUT=""
ARGS=("$@")

# Parse output argument
i=0
while [ $i -lt $# ]; do
    arg="${ARGS[$i]}"
    if [ "$arg" = "-o" ] || [ "$arg" = "--output" ]; then
        j=$((i + 1))
        LOCAL_OUT="${ARGS[$j]}"
        break
    fi
    i=$((i + 1))
done

if [ -z "$LOCAL_OUT" ]; then
    echo "ERROR: Could not determine output filename from arguments."
    echo "Arguments were: $*"
    exit 1
fi

echo "Output file: $LOCAL_OUT"
echo "Running process_evt_root.py..."

python3 process_evt_root.py "${ARGS[@]}"

if [ ! -f "$LOCAL_OUT" ]; then
    echo "ERROR: Expected output file '$LOCAL_OUT' was not created."
    exit 1
fi

echo "Processing finished. Output exists:"
ls -lh "$LOCAL_OUT"

# Ensure EOS destination exists
xrdfs cmseos.fnal.gov mkdir -p "$EOS_OUT_DIR"

# Copy output to EOS
echo "Copying $LOCAL_OUT to ${EOS_PREFIX}/${EOS_OUT_DIR}/"
xrdcp -f "$LOCAL_OUT" "${EOS_PREFIX}/${EOS_OUT_DIR}/"

echo "Copied successfully to EOS."

# Cleanup local output
rm -f "$LOCAL_OUT"
echo "Removed local output file."
