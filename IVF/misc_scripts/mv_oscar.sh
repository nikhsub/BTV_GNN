#!/bin/bash

SRC_DIR="/uscmst1b_scratch/lpc1/3DayLifetime/nvenkata/hplusb"
DEST_USER="nvenkat4"
DEST_HOST="ssh.ccv.brown.edu"
DEST_DIR="/users/nvenkat4/scratch/hplusb_trainfiles"

# Now sync all files in one SSH session (one Duo prompt)
rsync -avP "${SRC_DIR}/" ${DEST_USER}@${DEST_HOST}:"${DEST_DIR}/"
