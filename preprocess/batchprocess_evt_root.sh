#!/bin/bash

set -euo pipefail

# -----------------------------
# Directory paths
# -----------------------------
INPUT_DIR="/store/user/nvenkata/BTV/ttbarhad_toproc_1703"
TMP_DIR="/uscms/home/nvenkata/nobackup/BTV/preprocess/tmp"
OUTPUT_DIR="/store/user/nvenkata/BTV/fortrain_ttbarhad_root_1703"
EOS_PREFIX="root://cmseos.fnal.gov/"

# -----------------------------
# Processing parameters (override via env)
# -----------------------------
START_EVT="${START_EVT:-0}"
END_EVT="${END_EVT:--1}"
DOWNSAMPLE="${DOWNSAMPLE:-0.5}"
WRITE_CHUNK_SIZE="${WRITE_CHUNK_SIZE:-1000}"
N_JOBS="${N_JOBS:-16}"
MAX_FILES="${MAX_FILES:-276}"  # 0 means process all files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROCESS_SCRIPT="${SCRIPT_DIR}/process_evt_root.py"

mkdir -p "$TMP_DIR"

FAILED_LOG="${TMP_DIR}/failed_files.txt"
XRDCP_FAILED_LOG="${TMP_DIR}/xrdcp_failed_files.txt"
MISSING_OUTPUT_LOG="${TMP_DIR}/missing_output_files.txt"
JOBLOG="${TMP_DIR}/parallel_joblog.txt"

# Start fresh each run
: > "$FAILED_LOG"
: > "$XRDCP_FAILED_LOG"
: > "$MISSING_OUTPUT_LOG"
: > "$JOBLOG"

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

    if xrdfs cmseos.fnal.gov stat "$eos_out" > /dev/null 2>&1; then
        echo "Skipping (already exists): ${filename}"
        echo "--------------------------------------"
        return 0
    fi

    echo "Processing: ${mod_file_path}"

    if ! python3 "$PROCESS_SCRIPT" \
        -d "$mod_file_path" \
        -o "$local_out" \
        --in-tree tree \
        --out-tree tree \
        -s "$START_EVT" \
        -e "$END_EVT" \
        -ds "$DOWNSAMPLE" \
        --write-chunk-size "$WRITE_CHUNK_SIZE"
    then
        echo "ERROR: Python processing failed for ${filename}"
        echo "$file_path" >> "$FAILED_LOG"
        echo "--------------------------------------"
        rm -f "$local_out"
        return 0
    fi

    if [[ ! -s "$local_out" ]]; then
        echo "ERROR: Output file missing or empty for ${filename}"
        echo "$file_path" >> "$MISSING_OUTPUT_LOG"
        echo "--------------------------------------"
        rm -f "$local_out"
        return 0
    fi

    echo "Copying to EOS..."
    if ! xrdcp -f "$local_out" "${EOS_PREFIX}${eos_out}"; then
        echo "ERROR: xrdcp failed for ${filename}"
        echo "$file_path" >> "$XRDCP_FAILED_LOG"
        echo "--------------------------------------"
        rm -f "$local_out"
        return 0
    fi

    rm -f "$local_out"
    echo "Finished: ${filename}"
    echo "--------------------------------------"
}

export -f process_file
export PROCESS_SCRIPT TMP_DIR OUTPUT_DIR EOS_PREFIX START_EVT END_EVT DOWNSAMPLE WRITE_CHUNK_SIZE
export FAILED_LOG XRDCP_FAILED_LOG MISSING_OUTPUT_LOG

# -----------------------------
# Get file list, optionally cap total files, run in parallel
# -----------------------------
if [[ "$MAX_FILES" -gt 0 ]]; then
    xrdfsls "$INPUT_DIR" | grep '\.root$' | head -n "$MAX_FILES" | \
        parallel --joblog "$JOBLOG" -j "$N_JOBS" process_file {}
else
    xrdfsls "$INPUT_DIR" | grep '\.root$' | \
        parallel --joblog "$JOBLOG" -j "$N_JOBS" process_file {}
fi

echo "All possible files processed and transferred to EOS: ${OUTPUT_DIR}"
echo "Python failures logged to: ${FAILED_LOG}"
echo "Missing/empty outputs logged to: ${MISSING_OUTPUT_LOG}"
echo "xrdcp failures logged to: ${XRDCP_FAILED_LOG}"
echo "Parallel job log: ${JOBLOG}"
