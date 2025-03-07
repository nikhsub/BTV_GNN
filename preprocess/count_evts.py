import os
import ROOT

tot_count = 0  # Initialize the total event count

combined_dir = "toproc"

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

        # Loop through each key in the file
        for key in root_file.GetListOfKeys():
            obj = key.ReadObj()

            # Check if the object is a TTree
            if isinstance(obj, ROOT.TTree):
                # Ensure we only count the latest cycle
                if key.GetCycle() !=2:
                    continue

                tree_name = obj.GetName()
                num_entries = obj.GetEntries()
                tot_count += num_entries
                print(f"File: {filename}, Tree: {tree_name}, Number of Events: {num_entries}")

        # Close the ROOT file
        root_file.Close()

print(f"Total Number of Events: {tot_count}")

