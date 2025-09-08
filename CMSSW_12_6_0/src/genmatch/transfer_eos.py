import os
import time
import subprocess

# Configuration
watch_directory = "/uscms/home/nvenkata/nobackup/BTV/CMSSW_12_6_0/src/genmatch"  # Directory to monitor
#eos_directory = "/store/group/lpcljm/nvenkata/BTVH/ttbarlep_toproc_120files"   # EOS directory
eos_directory = "/store/user/nvenkata/BTV/toproc_0209"
polling_interval = 60

def transfer_and_remove(file_path, eos_directory):
    """
    Transfers a file to the EOS directory using xrdcp and removes it upon success.
    """
    try:
        # Construct the EOS destination path
        eos_path = f"root://cmseos.fnal.gov/{eos_directory}"
        
        # Execute the xrdcp command
        print(f"Transferring {file_path} to {eos_path}")
        subprocess.check_call(["xrdcp", file_path, eos_path])
        
        # Remove the file if transfer is successful
        os.remove(file_path)
        print(f"Successfully transferred and removed: {file_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to transfer {file_path}: {e}")
    except Exception as e:
        print(f"Error: {e}")

def monitor_directory(directory, eos_directory):
    """
    Monitors a directory for files matching output_*.root and processes them.
    """
    print(f"Monitoring directory: {directory}")
    while True:
        # List all files in the directory
        files = os.listdir(directory)
        
        # Process files matching the pattern output_*.root
        for file_name in files:
            if file_name.startswith("output_") and file_name.endswith(".root"):
                file_path = os.path.join(directory, file_name)
                transfer_and_remove(file_path, eos_directory)
        
        # Wait before checking again
        time.sleep(polling_interval)

if __name__ == "__main__":
    monitor_directory(watch_directory, eos_directory)
