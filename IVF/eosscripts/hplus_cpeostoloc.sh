#!/bin/bash


# EOS directory to copy FROM
#EOS_DIR="/store/user/nvenkata/BTV/hc_vertex_2018_HZZ4l_3FS_1505/HPlusCharm_3FS_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MC_vertex_hplusc3FS_2018_HZZ4l_1505/260515_205755/0000"

#for era in 2016 2017 2018
for era in 2018
do

	#for fs in 4FS 5FS
	for fs in 3FS
	do

		#EOS_DIR=$(xrdfsls /store/user/nvenkata/BTV/hb_vertex_${era}_HZZ4l_${fs}_1206/HPlusBottom_${fs}_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MC_vertex_hplusb${fs}_${era}_HZZ4l_1206)/0000
		EOS_DIR=$(xrdfsls /store/user/nvenkata/BTV/hc_${era}_HZZ4l_${fs}_2406/HPlusCharm_${fs}_MuRFScaleDynX0p50_HToZZTo4L_M125_TuneCP5_13TeV_amcatnlo_JHUGenV7011_pythia8/MC_hplusc${fs}_${era}_HZZ4l_2406)/0001
		
		
		# Local directory to copy TO
		LOCAL_DIR="/uscmst1b_scratch/lpc1/3DayLifetime/nvenkata/hc/"
		
		# Full EOS root URL prefix
		EOS_ROOT="root://cmseos.fnal.gov"
		
		# Create local directory if it doesn't exist
		mkdir -p "$LOCAL_DIR"
		
		echo "Copying all files FROM EOS: $EOS_ROOT/$EOS_DIR"
		echo "TO local directory: $LOCAL_DIR"
		echo
		
		# List all files in EOS directory using xrdfs
		FILE_LIST=$(xrdfs cmseos.fnal.gov ls "$EOS_DIR")
		
		if [ -z "$FILE_LIST" ]; then
		    echo "Error: No files found in EOS directory."
		    exit 1
		fi
		
		# Loop through each file returned by xrdfs
		for file in $FILE_LIST; do
		
		    # Skip directories
		    if [[ "$file" == */ ]]; then
		        continue
		    fi
		
		    # Grab just the filename
		    filename=$(basename "$file")
		
		    # Local destination path
		    dest="$LOCAL_DIR/$filename"
		
		    echo "Copying: $filename"
		
		    # Perform the download
		    xrdcp "$EOS_ROOT/$file" "$dest"
		
		    # Status check
		    if [ $? -eq 0 ]; then
		        echo "  ✓ Success: $filename"
		    else
		        echo "  ✗ Failed: $filename"
		    fi
		
		done
		
		echo
		echo "All files processed."
	done
done


