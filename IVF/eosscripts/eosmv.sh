#!/bin/bash

# Set source and destination EOS directories
SOURCE_DIR="/store/user/nvenkata/BTV/toproc"
DEST_DIR="/store/user/nvenkata/BTV/toproc/evttrain"

# EOS prefix
EOS_PREFIX="root://cmseos.fnal.gov"


for NUM in 27 28 29 2;
do
# Check if source directory exists
	if eos $EOS_PREFIX ls "$SOURCE_DIR" >/dev/null 2>&1; then
	    echo "Moving files matching 'output_${NUM}_chunk*.root' from $SOURCE_DIR to $DEST_DIR..."
	
	    # Loop over matching files in the source directory
	    for file in $(eos $EOS_PREFIX ls "$SOURCE_DIR" | grep "^output_${NUM}_chunk.*\.root$"); do
	        echo "Moving $file..."
	        eos $EOS_PREFIX mv "$SOURCE_DIR/$file" "$DEST_DIR/"
	    done
	
	    echo "All matching files moved successfully."
	else
	    echo "Error: Source directory does not exist or is not accessible."
	    exit 1
	fi
done

