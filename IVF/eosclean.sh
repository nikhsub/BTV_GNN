#!/bin/bash

# Set the EOS directory you want to clean
EOS_DIR="/store/user/nvenkata"

# Check if the directory exists
if eosls "$EOS_DIR" &> /dev/null; then
    echo "Removing all files from $EOS_DIR..."
    # List all files and remove them
    eosls "$EOS_DIR" | while read -r file; do
        eosrm -r "$EOS_DIR/$file"
    done
    echo "All files have been removed from $EOS_DIR."
else
    echo "Error: Directory $EOS_DIR does not exist or cannot be accessed."
fi

