#!/bin/bash

# Directory paths
INPUT_DIR="ttbar_lep_lowpt50_toproc"    # Set your input directory containing .root files
OUTPUT_DIR="files/training/ttbar_lep_lowpt50_0512"  # Set your output directory for .pkl files

mkdir -p "$OUTPUT_DIR"

# Parameters for event processing
START_EVT=0
END_EVT=-1

# Loop through all ROOT files in the input directory
for file_path in "$INPUT_DIR"/*.root; do
  # Extract file name without extension and set it as the save tag
  filename=$(basename "$file_path")
  save_tag="${filename%.root}"

  # Run the processing script with the required arguments
  python optprocess.py -d "$file_path" -st "$save_tag" -s "$START_EVT" -e "$END_EVT" -t

  # Move the generated .pkl file to the output directory
  mv "haddata_${save_tag}.pkl" "$OUTPUT_DIR/"
done

echo "Processing complete. All .pkl files have been moved to $OUTPUT_DIR."

