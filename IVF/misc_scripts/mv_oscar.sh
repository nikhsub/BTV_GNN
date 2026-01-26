#!/bin/bash

SRC_DIR="/uscmst1b_scratch/lpc1/3DayLifetime/nvenkata/btv"

# Remote (OSCAR) info
DEST_USER="nvenkat4"
DEST_HOST="ssh.ccv.brown.edu"
DEST_DIR="/users/nvenkat4/scratch/btv_trainfiles/ttbarhad_fullconn_1301"

# Create destination directory on OSCAR (first SSH hop; will prompt Duo once)
#ssh ${DEST_USER}@${DEST_HOST} "mkdir -p ${DEST_DIR}"

# Now sync all files in one SSH session (one Duo prompt)
rsync -avP "${SRC_DIR}/" ${DEST_USER}@${DEST_HOST}:"${DEST_DIR}/"

