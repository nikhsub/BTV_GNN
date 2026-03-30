#!/bin/bash

set -e pipefail

# -----------------------------
# EOS paths
# -----------------------------
EOS_PREFIX="root://cmseos.fnal.gov"

INPUT_DIR="/store/user/nvenkata/BTV/ttbarhad_toproc_1703"
OUTPUT_DIR="/store/user/nvenkata/BTV/fortrain_ttbarhad_root_1703"
DEST_DIR="/store/user/nvenkata/BTV/ttbarhad_toproc_1703/processed"

# -----------------------------
# Create destination directory if needed
# -----------------------------
#eos $EOS_PREFIX mkdir -p "$DEST_DIR"

echo "Scanning processed output files..."

# -----------------------------
# Loop over processed outputs
# -----------------------------
xrdfs cmseos.fnal.gov ls "$OUTPUT_DIR" | grep '\.root$' | while read -r out_file; do

    filename=$(basename "$out_file")

    # Convert:
    # evtrootdata_output_21_chunk3_None.root
    # → output_21_chunk3_None.root
    input_file="${filename#evtrootdata_}"

    src_path="${INPUT_DIR}/${input_file}"
    dest_path="${DEST_DIR}/${input_file}"

    # -----------------------------
    # Check if input exists before moving
    # -----------------------------
    if xrdfs cmseos.fnal.gov stat "$src_path" > /dev/null 2>&1; then
        echo "Moving: $input_file"

        eos $EOS_PREFIX mv "$src_path" "$DEST_DIR/"

    else
        echo "Skipping (not found in input): $input_file"
    fi

done

echo "Done moving processed input files."
