import sys
import os
import shutil
import glob
import argparse
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description='Submit condor jobs')
    parser.add_argument('-i', "--input", default="", help="The input directory where the analyzer output trees are")
    parser.add_argument('-o', "--output", default="", help="The output directory for flat trees")
    parser.add_argument('-p', "--perjob", default=10, type=int, help="Files per job")
    return parser.parse_args()

def get_root_files(basefolder):
    return glob.glob("{0}/*.root".format(basefolder))

def group_files(root_files, files_per_job):
    filegroups = {}
    currgroup = ""
    for icount, rootfile in enumerate(root_files, start=1):
        rootfile=rootfile[rootfile.find("/store"): ]
        rootfile="root://cmseos.fnal.gov/"+rootfile
        if currgroup:
            currgroup += "," + rootfile
        else:
            currgroup += rootfile

        if icount % files_per_job == 0 or icount == len(root_files):
            key = (icount -1) // files_per_job+1
            filegroups[key] = currgroup
            currgroup = ""
    return filegroups

def create_output_folders(output_dir, filegroups):
    for key in filegroups.keys():
        folder0 = "output_" + str(key)
        folder = os.path.join(output_dir, folder0)
        os.makedirs(folder)
    return list(filegroups.keys())


def create_condor_submit_file(folder, key, pyscript, bashjob, filegroup, current):
    condor_filename = "analyze_condor_%s" % key
    with open(condor_filename, "w") as fcondor:
        fcondor.write("Executable = %s\n" % bashjob)
        fcondor.write("Universe = vanilla\n")
        fcondor.write("transfer_input_files = %s\n" % pyscript)
        fcondor.write("should_transfer_files = YES\n")
        fcondor.write("Output = %s/%s/run_%s.out\n" % (current, folder, key))
        fcondor.write("Error  = %s/%s/run_%s.err\n" % (current, folder, key))
        fcondor.write("Log    = %s/%s/run_%s.log\n" % (current, folder, key))
        #fcondor.write('+DesiredOS = "SL7"\n')
        #fcondor.write("request_memory = 8000\n")


        output = "output_"+str(key)

        fcondor.write("Arguments = %s %s %s\n" % (pyscript, filegroup, output))

        fcondor.write("Queue\n")
    return condor_filename

def main():
    args = parse_arguments()
    current = os.getcwd()

    root_files = get_root_files(args.input)
    print(root_files)
    filegroups = group_files(root_files, args.perjob)
    print(filegroups)
    keys = create_output_folders(args.output, filegroups)

    for key in keys:
        folder = os.path.join(args.output, "output_%d" % key)
        os.chdir(folder)

        output = "output_"+str(key)

        shutil.copyfile(os.path.join(current, "trkinfo.py"), "trkinfo.py")
        shutil.copyfile(os.path.join(current, "submit.sh"), "submit.sh")

        condor_file = create_condor_submit_file(
            folder, key, "trkinfo.py", "submit.sh", filegroups[key], current
        )

        os.system("chmod +x submit.sh trkinfo.py %s" % condor_file)
        os.system("condor_submit %s" % condor_file)
        os.chdir(current)


if __name__ == "__main__":
    main()
