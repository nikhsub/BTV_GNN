#!/bin/bash

SRC_DIR="/uscmst1b_scratch/lpc1/3DayLifetime/nvenkata/hc"
DEST_USER="nvenkat4"
DEST_HOST="ssh.ccv.brown.edu"
DEST_DIR="/oscar/scratch/nvenkat4/graphevt_trainfiles/hplusc"

# Now sync all files in one SSH session (one Duo prompt)
rsync -avP "${SRC_DIR}/" ${DEST_USER}@${DEST_HOST}:"${DEST_DIR}/"
