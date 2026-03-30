#!/usr/bin/env python3
import os
import shlex
import shutil
import argparse
import subprocess

PROCESS_SCRIPT = "process_evt_root.py"
SUBMIT_SCRIPT = "submit.sh"
EOS_PREFIX = "root://cmseos.fnal.gov"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Submit one condor job per ROOT file")
    parser.add_argument(
        "-i", "--input", required=True,
        help="EOS directory containing input ROOT files, e.g. /store/user/nvenkata/BTV/ttbarhad_toproc_1703"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Main local output directory for condor job folders/logs"
    )
    parser.add_argument(
        "-st", "--savetag", default="",
        help="Optional tag to append to job names/output names"
    )
    parser.add_argument(
    "--max-files",
    type=int,
    default=-1,
    help="Maximum number of files to process (-1 = all)"
    )
    return parser.parse_args()


def normalize_eos_dir(eos_dir):
    if eos_dir.startswith(EOS_PREFIX):
        eos_dir = eos_dir[len(EOS_PREFIX):]
    return eos_dir.rstrip("/")


def get_root_files(eos_dir):
    eos_dir = normalize_eos_dir(eos_dir)
    cmd = ["xrdfs", "cmseos.fnal.gov", "ls", eos_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    files = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.endswith(".root"):
            files.append(f"{EOS_PREFIX}/{line}")

    return sorted(files)


def build_processing_args(
    input_file,
    output_file,
    in_tree,
    out_tree,
    start,
    end,
    downsample_fraction,
    fully_connected,
    seed,
    pt_branch,
    eta_branch,
    write_chunk_size,
    read_chunk_size,
):
    args = [
        "-d", input_file,
        "-o", output_file,
        "--in-tree", in_tree,
        "--out-tree", out_tree,
        "-s", str(start),
        "-e", str(end),
        "-ds", str(downsample_fraction),
        "--seed", str(seed),
        "--pt-branch", pt_branch,
        "--eta-branch", eta_branch,
        "--write-chunk-size", str(write_chunk_size),
        "--read-chunk-size", str(read_chunk_size),
    ]

    if fully_connected:
        args.append("--fully-connected")

    return args


def create_condor_submit_file(folder, key, bashjob, processing_args):
    condor_filename = os.path.join(folder, f"analyze_condor_{key}")

    quoted_args = " ".join(shlex.quote(x) for x in processing_args)

    with open(condor_filename, "w") as fcondor:
        fcondor.write(f"Executable = {bashjob}\n")
        fcondor.write("Universe = vanilla\n")
        fcondor.write(f"transfer_input_files = {PROCESS_SCRIPT}\n")
        fcondor.write("should_transfer_files = YES\n")
        fcondor.write("when_to_transfer_output = ON_EXIT\n")
        fcondor.write('transfer_output_files = ""\n')
        fcondor.write(f"Output = {folder}/run_{key}.out\n")
        fcondor.write(f"Error  = {folder}/run_{key}.err\n")
        fcondor.write(f"Log    = {folder}/run_{key}.log\n")
        fcondor.write("request_memory = 8000\n")
        fcondor.write("request_cpus = 1\n")
        fcondor.write(f"Arguments = {quoted_args}\n")
        fcondor.write("Queue\n")

    return condor_filename


def main():
    args = parse_arguments()
    current = os.getcwd()

    main_output_dir = os.path.join(current, args.output)
    os.makedirs(main_output_dir, exist_ok=True)

    # -----------------------------
    # TOGGLES FOR PROCESSING SCRIPT
    # -----------------------------
    IN_TREE = "tree"
    OUT_TREE = "tree"

    START = 0
    END = -1

    DOWNSAMPLE_FRACTION = 0.5
    FULLY_CONNECTED = True
    SEED = 12345

    PT_BRANCH = "jet_pt"
    ETA_BRANCH = "jet_eta"

    WRITE_CHUNK_SIZE = 500
    READ_CHUNK_SIZE = 1000

    try:
        root_files = get_root_files(args.input)
        if args.max_files > 0:
            root_files = root_files[:args.max_files]
    except subprocess.CalledProcessError as e:
        print(f"Failed to list EOS directory: {args.input}")
        print(e.stderr)
        raise

    if not root_files:
        print(f"No ROOT files found in {args.input}")
        return

    tag = args.savetag.strip()

    for file_index, input_file in enumerate(root_files, start=1):
        base = os.path.basename(input_file).replace(".root", "")
        key = f"{file_index}"
        if tag:
            key = f"{key}_{tag}"

        job_folder = os.path.join(main_output_dir, f"file{file_index}")
        os.makedirs(job_folder, exist_ok=True)

        local_py = os.path.join(job_folder, PROCESS_SCRIPT)
        local_sh = os.path.join(job_folder, SUBMIT_SCRIPT)

        shutil.copyfile(os.path.join(current, PROCESS_SCRIPT), local_py)
        shutil.copyfile(os.path.join(current, SUBMIT_SCRIPT), local_sh)

        output_name = f"evtrootdata_{base}"
        if tag:
            output_name = f"{output_name}_{tag}"
        output_name += ".root"

        processing_args = build_processing_args(
            input_file=input_file,
            output_file=output_name,
            in_tree=IN_TREE,
            out_tree=OUT_TREE,
            start=START,
            end=END,
            downsample_fraction=DOWNSAMPLE_FRACTION,
            fully_connected=FULLY_CONNECTED,
            seed=SEED,
            pt_branch=PT_BRANCH,
            eta_branch=ETA_BRANCH,
            write_chunk_size=WRITE_CHUNK_SIZE,
            read_chunk_size=READ_CHUNK_SIZE,
        )

        condor_file = create_condor_submit_file(
            folder=job_folder,
            key=key,
            bashjob=SUBMIT_SCRIPT,
            processing_args=processing_args,
        )

        os.system(f"chmod +x {local_sh} {local_py} {condor_file}")
        print(f"Submitting job for {input_file}")
        os.system(f"condor_submit {condor_file}")


if __name__ == "__main__":
    main()
