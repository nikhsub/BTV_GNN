#!/bin/bash

# Usage:
# ./move_eos_files.sh <SRC_DIR> <DEST_DIR> <NUM_FILES>

SRC_DIR="/store/user/nvenkata/BTV/proc_fortrain_ttbarhad_1311/"
DEST_DIR="/store/user/nvenkata/BTV/proc_fortrain_ttbarhad_1311/excess"
NUM="20"

#EOS_PREFIX="root://cmseos.fnal.gov/"
EOS_PREFIX=""

# Get a list of ROOT files in the source directory
files=$(xrdfsls $EOS_PREFIX"$SRC_DIR" | grep "\.pt$" | head -n "$NUM")

#echo $files

echo "Moving $NUM files from:"
echo "  $SRC_DIR"
echo "to:"
echo "  $DEST_DIR"
echo ""

for f in $files; do
    base=$(basename "$f")
    echo "Moving: $base"

    # mv file to new location
    eosmv "${EOS_PREFIX}${f}" "${EOS_PREFIX}${DEST_DIR}/${base}"


    echo "Done: $base"
    echo "-------------------------"
done

echo "Finished moving files!"

