import os
import ROOT

# Directory containing the combined files
combined_dir = "comb_files"

# Loop through each .root file in the combined files directory
for filename in os.listdir(combined_dir):
    # Check if the file is a .root file
    if filename.endswith(".root"):
        file_path = os.path.join(combined_dir, filename)
        
        # Open the ROOT file
        root_file = ROOT.TFile.Open(file_path)
        
        # Check if the file was opened successfully
        if not root_file or root_file.IsZombie():
            print(f"Error: Could not open {filename}")
            continue
        
        # Loop through each tree in the file
        for key in root_file.GetListOfKeys():
            obj = key.ReadObj()
            # Check if the object is a TTree
            if isinstance(obj, ROOT.TTree):
                tree_name = obj.GetName()
                num_entries = obj.GetEntries()
                print(f"File: {filename}, Tree: {tree_name}, Number of Events: {num_entries}")
        
        # Close the ROOT file
        root_file.Close()

