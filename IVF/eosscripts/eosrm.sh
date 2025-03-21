#!/bin/bash

# Set input EOS directory (user provides this as an argument)
INPUT_DIR="/store/user/nvenkata/BTV/toproc/evttrain"

# EOS prefix
EOS_PREFIX="root://cmseos.fnal.gov"

# Particle number to filter (user provides this as an argument)
RMNUM=47

# Check if the input directory exists
if eos $EOS_PREFIX ls "$INPUT_DIR" >/dev/null 2>&1; then
    echo "Removing files matching 'output_${RMNUM}_chunk*.root' from $INPUT_DIR..."

    # Find matching files and remove them one by one
    for file in $(eos $EOS_PREFIX ls "$INPUT_DIR" | grep "^output_${RMNUM}_chunk.*\.root$"); do
        echo "Deleting $file..."
        eos $EOS_PREFIX rm "$INPUT_DIR/$file"
    done

    echo "All matching files deleted successfully."
else
    echo "Error: Input directory does not exist or is not accessible."
    exit 1
fi

