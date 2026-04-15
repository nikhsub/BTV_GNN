#!/bin/bash

EOS_DIR="/store/group/lpcljm/nvenkata/hplusb/fortrain_hplusb_root_3103"
EOS_HOST="cmseos.fnal.gov"
EOS_ROOT="root://cmseos.fnal.gov"
LOCAL_DIR="/uscmst1b_scratch/lpc1/3DayLifetime/nvenkata/hplusb"

JOBS=8
STREAMS=4
RETRIES=3

mkdir -p "$LOCAL_DIR"

FILE_LIST=$(xrdfs "$EOS_HOST" ls "$EOS_DIR")
if [ -z "$FILE_LIST" ]; then
    echo "Error: No files found in EOS directory."
    exit 1
fi

export EOS_ROOT LOCAL_DIR STREAMS RETRIES

printf '%s\n' "$FILE_LIST" | grep -v '/$' | xargs -P "$JOBS" -n 1 bash -c '
file="$1"
filename=$(basename "$file")
dest="$LOCAL_DIR/$filename"

if [ -s "$dest" ]; then
    echo "[SKIP] $filename already exists"
    exit 0
fi

for ((attempt=1; attempt<=RETRIES; attempt++)); do
    echo "[COPY attempt $attempt/$RETRIES] $filename"
    echo "    src : $EOS_ROOT/$file"
    echo "    dest: $dest"

    xrdcp --streams "$STREAMS" "$EOS_ROOT/$file" "$dest"
    rc=$?

    if [ $rc -eq 0 ]; then
        echo "[OK] $filename"
        exit 0
    fi

    echo "[WARN] Failed attempt $attempt for $filename"
    sleep 2
done

echo "$filename" >> failed_xrdcp.txt
echo "[FAIL] $filename"
exit 1
' _

rc=$?

if [ $rc -eq 0 ]; then
    echo "All files processed successfully."
else
    echo "Completed with some failures. See failed_xrdcp.txt"
fi
