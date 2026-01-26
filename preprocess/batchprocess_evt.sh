#!/bin/bash

# -----------------------------
# Directory paths
# -----------------------------
INPUT_DIR="/store/user/nvenkata/BTV/toproc_1201/"
TMP_DIR="/uscms/home/nvenkata/nobackup/BTV/preprocess/tmp"
OUTPUT_DIR="/store/user/nvenkata/BTV/fortrain_fullconn_ttbarhad_1201"
EOS_PREFIX="root://cmseos.fnal.gov/"

mkdir -p "$TMP_DIR"

# Parameters for event processing
START_EVT=0
END_EVT=-1
DOWNSAMPLE=0.5

# -----------------------------
# Define the per-file job function
# -----------------------------
process_file() {
    local file_path="$1"
    local filename
    filename=$(basename "$file_path")
    local save_tag="${filename%.root}"

    local mod_file_path="${EOS_PREFIX}${file_path}"
    local local_out="${TMP_DIR}/evttraindata_${save_tag}.pt"
    local eos_out="${OUTPUT_DIR}/evttraindata_${save_tag}.pt"

    echo "Processing: $mod_file_path"
    python process_evt.py -d "$mod_file_path" -st "$save_tag" -s "$START_EVT" -e "$END_EVT" -ds "$DOWNSAMPLE"

    mv "evttraindata_${save_tag}.pt" "$TMP_DIR/" || exit 1

    echo "Copying to EOS..."
    xrdcp -f "$local_out" "${EOS_PREFIX}${eos_out}" && rm -f "$local_out"

    echo "Finished: $filename"
    echo "--------------------------------------"
}

export -f process_file
export TMP_DIR OUTPUT_DIR EOS_PREFIX START_EVT END_EVT DOWNSAMPLE

# -----------------------------
# Get file list and run in parallel
# -----------------------------
xrdfsls "$INPUT_DIR" | grep '\.root$' | \
    parallel -j 8 process_file {}

echo " All files processed and transferred to EOS: $OUTPUT_DIR"

