#!/bin/bash

# EOS directory to copy FROM
EOS_DIR="/store/user/nvenkata/BTV/fortrain_fullconn_ttbarhad_1201/"

# Local directory to copy TO
LOCAL_DIR="/uscmst1b_scratch/lpc1/3DayLifetime/nvenkata/btv"

# Full EOS root URL prefix
EOS_ROOT="root://cmseos.fnal.gov"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

echo "Copying all files FROM EOS: $EOS_ROOT/$EOS_DIR"
echo "TO local directory: $LOCAL_DIR"
echo

# List all files in EOS directory using xrdfs
FILE_LIST=$(xrdfs cmseos.fnal.gov ls "$EOS_DIR")

if [ -z "$FILE_LIST" ]; then
    echo "Error: No files found in EOS directory."
    exit 1
fi

# Loop through each file returned by xrdfs
for file in $FILE_LIST; do

    # Skip directories
    if [[ "$file" == */ ]]; then
        continue
    fi

    # Grab just the filename
    filename=$(basename "$file")

    # Local destination path
    dest="$LOCAL_DIR/$filename"

    echo "Copying: $filename"

    # Perform the download
    xrdcp "$EOS_ROOT/$file" "$dest"

    # Status check
    if [ $? -eq 0 ]; then
        echo "  ✓ Success: $filename"
    else
        echo "  ✗ Failed: $filename"
    fi

done

echo
echo "All files processed."


