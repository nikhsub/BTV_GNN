#!/bin/bash

# Local directory containing the files to move
LOCAL_DIR="/uscms/home/nvenkata/nobackup/BTV/IVF/files/training/ttbar_lep_4M_1912"

# EOS directory where files will be moved
EOS_DIR="root://cmseos.fnal.gov//store/user/nvenkata"

# Check if the local directory exists
if [ -d "$LOCAL_DIR" ]; then
    echo "Moving all files from $LOCAL_DIR to $EOS_DIR..."

    # Iterate through all files in the local directory
    for file in "$LOCAL_DIR"/*; do
        if [ -f "$file" ]; then
            # Extract the base file name (without the directory)
            filename=$(basename "$file")
            
            # Copy the file to the EOS area
            xrdcp "$file" "$EOS_DIR/$filename"

            # Check if the copy was successful
            if [ $? -eq 0 ]; then
                echo "Successfully moved: $file"
                # Delete the local file after successful copy
                rm "$file"
            else
                echo "Failed to move: $file"
            fi
        fi
    done
    echo "All files have been processed."
else
    echo "Error: Local directory $LOCAL_DIR does not exist."
fi

