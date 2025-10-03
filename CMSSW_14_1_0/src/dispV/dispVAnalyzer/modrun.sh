#!/bin/bash

for val in $(seq 0 0.1 0.8); do
    # format value (0.1 -> 0p1, 0.8 -> 0p8)
    tag=$(echo $val | sed 's/\./p/')
    outfile="ttbarhad_3k_mod${tag}cut.root"

    echo ">>> Running with TrackPredCut=$val, output=$outfile"

    cmsRun Events_cfg.py trackCut=$val outfile=$outfile
done

