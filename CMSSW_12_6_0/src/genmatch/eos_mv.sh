#!/bin/bash

# Source and destination directories (Modify these)
SOURCE_DIR="/store/group/lpcljm/nvenkata/BTVH/ttbarlep_toproc_120files"
DEST_DIR="/store/user/nvenkata/ttbar_lep_toproc"

# EOS prefix
EOS_PREFIX="root://cmseos.fnal.gov"

# Check if source directory exists
if eos $EOS_PREFIX ls "$SOURCE_DIR" >/dev/null 2>&1; then
    echo "Moving files from $SOURCE_DIR to $DEST_DIR..."

    # Loop over each file in the source directory
    for file in $(eos $EOS_PREFIX ls "$SOURCE_DIR"); do
        echo "Moving $file..."
        eos $EOS_PREFIX mv "$SOURCE_DIR/$file" "$DEST_DIR/"
    done

    echo "All files moved successfully."
else
    echo "Error: Source directory does not exist or is not accessible."
    exit 1
fi

