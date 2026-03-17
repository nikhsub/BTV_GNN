#!/bin/bash

set -euo pipefail

# -----------------------------
# Directory paths
# -----------------------------
INPUT_DIR="/store/group/lpcljm/nvenkata/hplusc/toproc_0503"
TMP_DIR="/uscms/home/nvenkata/nobackup/BTV/preprocess/tmp"
OUTPUT_DIR="/store/group/lpcljm/nvenkata/hplusc/fortrain_root_1603"
EOS_PREFIX="root://cmseos.fnal.gov/"

# -----------------------------
# Processing parameters (override via env)
# -----------------------------
START_EVT="${START_EVT:-0}"
END_EVT="${END_EVT:--1}"
DOWNSAMPLE="${DOWNSAMPLE:-0.5}"
WRITE_CHUNK_SIZE="${WRITE_CHUNK_SIZE:-1500}"
N_JOBS="${N_JOBS:-8}"
MAX_FILES="${MAX_FILES:-500}"  # 0 means process all files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESS_SCRIPT="${SCRIPT_DIR}/process_evt_root.py"

mkdir -p "$TMP_DIR"

# -----------------------------
# Define the per-file job function
# -----------------------------
process_file() {
    local file_path="$1"
    local filename
    filename=$(basename "$file_path")
    local save_tag="${filename%.root}"

    local mod_file_path="${EOS_PREFIX}${file_path}"
    local local_out="${TMP_DIR}/evtrootdata_${save_tag}.root"
    local eos_out="${OUTPUT_DIR}/evtrootdata_${save_tag}.root"

    echo "Processing: ${mod_file_path}"
    python3 "$PROCESS_SCRIPT" \
        -d "$mod_file_path" \
        -o "$local_out" \
        --in-tree tree \
        --out-tree tree \
        -s "$START_EVT" \
        -e "$END_EVT" \
        -ds "$DOWNSAMPLE" \
        --write-chunk-size "$WRITE_CHUNK_SIZE"

    echo "Copying to EOS..."
    xrdcp -f "$local_out" "${EOS_PREFIX}${eos_out}" && rm -f "$local_out"

    echo "Finished: ${filename}"
    echo "--------------------------------------"
}

export -f process_file
export PROCESS_SCRIPT TMP_DIR OUTPUT_DIR EOS_PREFIX START_EVT END_EVT DOWNSAMPLE WRITE_CHUNK_SIZE

# -----------------------------
# Get file list, optionally cap total files, run in parallel
# -----------------------------
if [[ "$MAX_FILES" -gt 0 ]]; then
    xrdfsls "$INPUT_DIR" | grep '\.root$' | head -n "$MAX_FILES" | \
        parallel -j "$N_JOBS" process_file {}
else
    xrdfsls "$INPUT_DIR" | grep '\.root$' | \
        parallel -j "$N_JOBS" process_file {}
fi

echo "All files processed and transferred to EOS: ${OUTPUT_DIR}"
