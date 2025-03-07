import sys
import os
import shutil
import glob
import argparse
from ROOT import TFile, TTree

def parse_arguments():
    parser = argparse.ArgumentParser(description='Submit condor jobs')
    parser.add_argument('-i', "--input", default="", help="The input directory where the analyzer output trees are")
    parser.add_argument('-o', "--output", default="", help="The main output directory for log files")
    parser.add_argument('-ec', "--evtchunks", default=3000, type=int, help="Events per chunk")
    return parser.parse_args()

def get_total_events(filename):
    """
    Get the total number of events in the ROOT file.
    """
    file = TFile.Open(filename)
    tree = file.Get("demo/tree")  # Adjust the path if needed
    num_events = tree.GetEntries()
    file.Close()
    return num_events

def get_root_files(basefolder):
    return glob.glob(f"{basefolder}/*.root")

def create_condor_submit_file(folder, key, pyscript, bashjob, filename, current, start, end):
    condor_filename = os.path.join(folder, f"analyze_condor_{key}")
    with open(condor_filename, "w") as fcondor:
        fcondor.write(f"Executable = {bashjob}\n")
        fcondor.write("Universe = vanilla\n")
        fcondor.write(f"transfer_input_files = {pyscript}\n")
        fcondor.write("should_transfer_files = YES\n")
        fcondor.write(f"Output = {folder}/run_{key}.out\n")
        fcondor.write(f"Error  = {folder}/run_{key}.err\n")
        fcondor.write(f"Log    = {folder}/run_{key}.log\n")
        fcondor.write("request_memory = 8000\n")
        fcondor.write("request_cpus = 4\n")

        output = f"output_{key}"
        fcondor.write(f"Arguments = {pyscript} {filename} {output} {start} {end}\n")
        fcondor.write("Queue\n")
    return condor_filename

def main():
    args = parse_arguments()
    current = os.getcwd()

    # Ensure the main output directory exists
    main_output_dir = os.path.join(current, args.output)
    os.makedirs(main_output_dir, exist_ok=True)

    root_files = get_root_files(args.input)
    events_per_chunk = args.evtchunks

    for file_index, filename in enumerate(root_files, start=1):  # Start file index from 1
        # Adjust filename path for Condor
        filename = filename[filename.find("/store"):]
        filename = f"root://cmseos.fnal.gov/{filename}" #UNCOMMENT FOR EOS AREA

        # Calculate number of chunks
        total_events = get_total_events(filename)
        num_chunks = (total_events + events_per_chunk - 1) // events_per_chunk

        for chunk in range(1, num_chunks + 1):  # Start chunk count from 1
            start = (chunk - 1) * events_per_chunk
            end = min(start + events_per_chunk, total_events)

            # Create a unique folder for each file chunk within the main output directory
            chunk_folder = os.path.join(main_output_dir, f"file{file_index}_chunk{chunk}")
            os.makedirs(chunk_folder, exist_ok=True)

            # Copy necessary scripts to the chunk folder
            shutil.copyfile(os.path.join(current, "trkinfo.py"), os.path.join(chunk_folder, "trkinfo.py"))
            shutil.copyfile(os.path.join(current, "submit.sh"), os.path.join(chunk_folder, "submit.sh"))

            # Create Condor submit file in the chunk folder
            condor_file = create_condor_submit_file(
                chunk_folder, f"{file_index}_chunk{chunk}", "trkinfo.py", "submit.sh", filename, current, start, end
            )

            # Set permissions and submit the job
            os.system(f"chmod +x {chunk_folder}/submit.sh {chunk_folder}/trkinfo.py {condor_file}")
            os.system(f"condor_submit {condor_file}")

        os.chdir(current)

if __name__ == "__main__":
    main()

