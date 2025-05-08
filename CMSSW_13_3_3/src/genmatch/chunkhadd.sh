#!/bin/bash

mkdir -p comb_files

for base_name in $(ls output_*_chunk*.root | cut -d'_' -f1,2 | sort | uniq); do

    chunk_files=$(ls ${base_name}_chunk*.root 2>/dev/null)
    if [[ -z "$chunk_files" ]]; then
        echo "No chunk files found for $base_name. Skipping."
        continue
    fi

    combined_file="comb_files/${base_name}_combined.root"

    echo "Combining chunks for $base_name..."
    hadd "$combined_file" ${base_name}_chunk*.root

    if [[ $? -eq 0 ]]; then
        echo "Successfully combined $base_name into $combined_file"
    else
        echo "Error combining $base_name"
    fi
done

