#!/bin/bash

# Enhanced batch processing script with improved error handling and parallel processing
# Usage: ./batchprocess_improved.sh [OPTIONS]

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Default configuration
INPUT_DIR="/store/user/nvenkata/BTV/toproc/"
OUTPUT_DIR="/uscms/home/nvenkata/nobackup/BTV/IVF/files/training/evt_ttbar_had_2406"
EOS_PREFIX="root://cmseos.fnal.gov/"
SCRIPT_TYPE="evt"  # evt, had, or val
START_EVT=0
END_EVT=-1
MAX_PARALLEL_JOBS=4
CONFIG_FILE=""
VALIDATION=false
DRY_RUN=false

# Logging setup
LOG_DIR="${OUTPUT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_processing_${TIMESTAMP}.log"

# Help function
show_help() {
    cat << EOF
Enhanced Batch Processing Script for Preprocessing

Usage: $0 [OPTIONS]

OPTIONS:
    -i, --input-dir DIR         Input directory containing ROOT files (default: $INPUT_DIR)
    -o, --output-dir DIR        Output directory for processed files (default: $OUTPUT_DIR)
    -t, --type TYPE             Processing type: evt, had, val (default: $SCRIPT_TYPE)
    -s, --start NUM             Starting event number (default: $START_EVT)
    -e, --end NUM               Ending event number, -1 for all events (default: $END_EVT)
    -j, --jobs NUM              Maximum parallel jobs (default: $MAX_PARALLEL_JOBS)
    -c, --config FILE           Configuration file path (optional)
    -p, --eos-prefix PREFIX     EOS prefix (default: $EOS_PREFIX)
    -v, --validate              Run validation after processing
    -n, --dry-run               Show commands without executing
    -h, --help                  Show this help message

EXAMPLES:
    # Basic processing with default settings
    $0 -t evt

    # Process with custom parameters and validation
    $0 -t evt -s 0 -e 1000 -j 8 -v

    # Dry run to see what would be executed
    $0 -t had -n

    # Use custom configuration
    $0 -t val -c config.yaml --validate
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -i|--input-dir)
                INPUT_DIR="$2"
                shift 2
                ;;
            -o|--output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            -t|--type)
                SCRIPT_TYPE="$2"
                shift 2
                ;;
            -s|--start)
                START_EVT="$2"
                shift 2
                ;;
            -e|--end)
                END_EVT="$2"
                shift 2
                ;;
            -j|--jobs)
                MAX_PARALLEL_JOBS="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -p|--eos-prefix)
                EOS_PREFIX="$2"
                shift 2
                ;;
            -v|--validate)
                VALIDATION=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                show_help
                exit 1
                ;;
        esac
    done
}

# Logging functions
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

# Setup function
setup() {
    # Create necessary directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$LOG_DIR"
    
    # Validate script type
    if [[ ! "$SCRIPT_TYPE" =~ ^(evt|had|val)$ ]]; then
        log_error "Invalid script type: $SCRIPT_TYPE. Must be one of: evt, had, val"
        exit 1
    fi
    
    # Check if processing script exists
    local script_name="process_${SCRIPT_TYPE}.py"
    if [[ ! -f "$script_name" ]]; then
        log_error "Processing script not found: $script_name"
        exit 1
    fi
    
    # Check if improved script exists and use it if available
    local improved_script="process_${SCRIPT_TYPE}_improved.py"
    if [[ -f "$improved_script" ]]; then
        script_name="$improved_script"
        log_info "Using improved script: $script_name"
    fi
    
    log_info "Setup completed successfully"
    log_info "Input directory: $INPUT_DIR"
    log_info "Output directory: $OUTPUT_DIR"
    log_info "Script type: $SCRIPT_TYPE"
    log_info "Parallel jobs: $MAX_PARALLEL_JOBS"
    log_info "Event range: $START_EVT to $END_EVT"
}

# Get list of ROOT files
get_file_list() {
    local file_list=()
    
    if [[ "$INPUT_DIR" == /store/* ]]; then
        # EOS directory
        log_info "Scanning EOS directory: $INPUT_DIR"
        while IFS= read -r file_path; do
            if [[ "$file_path" == *.root ]]; then
                file_list+=("$file_path")
            fi
        done < <(xrdfsls "$INPUT_DIR" 2>/dev/null || true)
    else
        # Local directory
        log_info "Scanning local directory: $INPUT_DIR"
        while IFS= read -r -d '' file_path; do
            file_list+=("$file_path")
        done < <(find "$INPUT_DIR" -name "*.root" -print0 2>/dev/null || true)
    fi
    
    if [[ ${#file_list[@]} -eq 0 ]]; then
        log_error "No ROOT files found in $INPUT_DIR"
        exit 1
    fi
    
    log_info "Found ${#file_list[@]} ROOT files to process"
    printf '%s\n' "${file_list[@]}"
}

# Process a single file
process_file() {
    local file_path="$1"
    local filename=$(basename "$file_path")
    local save_tag="${filename%.root}"
    local mod_file_path="${EOS_PREFIX}${file_path}"
    
    # Select appropriate script
    local script_name="process_${SCRIPT_TYPE}.py"
    if [[ -f "process_${SCRIPT_TYPE}_improved.py" ]]; then
        script_name="process_${SCRIPT_TYPE}_improved.py"
    fi
    
    # Build command
    local cmd="python $script_name -d \"$mod_file_path\" -st \"$save_tag\" -s $START_EVT -e $END_EVT"
    
    # Add configuration if provided
    if [[ -n "$CONFIG_FILE" ]]; then
        cmd="$cmd -c \"$CONFIG_FILE\""
    fi
    
    # Add output directory if improved script supports it
    if [[ -f "process_${SCRIPT_TYPE}_improved.py" ]]; then
        cmd="$cmd --output_dir \"$OUTPUT_DIR\""
    fi
    
    log_info "Processing: $filename"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN: $cmd"
        return 0
    fi
    
    # Execute command with error handling
    local start_time=$(date +%s)
    local output_file=""
    
    if eval "$cmd" 2>&1 | tee -a "$LOG_FILE.${save_tag}"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Determine output file name based on script type
        case "$SCRIPT_TYPE" in
            evt) output_file="evttraindata_${save_tag}.pkl" ;;
            had) output_file="haddata_${save_tag}.pkl" ;;
            val) output_file="evtvaldata_${save_tag}.pkl" ;;
        esac
        
        # Move output file if it exists and we're using the original script
        if [[ -f "$output_file" && "$script_name" != *"_improved.py" ]]; then
            mv "$output_file" "$OUTPUT_DIR/"
            log_info "Moved $output_file to $OUTPUT_DIR/"
        fi
        
        log_info "Successfully processed $filename in ${duration}s"
        return 0
    else
        log_error "Failed to process $filename"
        return 1
    fi
}

# Process files with job control
process_files_parallel() {
    local file_list=("$@")
    local total_files=${#file_list[@]}
    local completed=0
    local failed=0
    local active_jobs=0
    local job_pids=()
    
    log_info "Starting parallel processing of $total_files files with max $MAX_PARALLEL_JOBS jobs"
    
    # Function to wait for job completion
    wait_for_jobs() {
        local max_wait=${1:-$MAX_PARALLEL_JOBS}
        while [[ $active_jobs -ge $max_wait ]]; do
            for i in "${!job_pids[@]}"; do
                local pid=${job_pids[i]}
                if ! kill -0 "$pid" 2>/dev/null; then
                    wait "$pid"
                    local exit_code=$?
                    unset 'job_pids[i]'
                    ((active_jobs--))
                    
                    if [[ $exit_code -eq 0 ]]; then
                        ((completed++))
                    else
                        ((failed++))
                    fi
                    
                    log_info "Progress: $completed completed, $failed failed, $((total_files - completed - failed)) remaining"
                    break
                fi
            done
            sleep 1
        done
    }
    
    # Process files
    for file_path in "${file_list[@]}"; do
        # Wait for available slot
        wait_for_jobs
        
        # Start new job
        process_file "$file_path" &
        local job_pid=$!
        job_pids+=("$job_pid")
        ((active_jobs++))
        
        log_info "Started job $job_pid for $(basename "$file_path") (active jobs: $active_jobs)"
    done
    
    # Wait for all remaining jobs
    log_info "Waiting for remaining jobs to complete..."
    wait_for_jobs 1
    
    log_info "Processing completed: $completed successful, $failed failed out of $total_files total"
    
    if [[ $failed -gt 0 ]]; then
        log_warn "$failed files failed to process"
        return 1
    fi
    
    return 0
}

# Validation function
run_validation() {
    if [[ "$VALIDATION" != true ]]; then
        return 0
    fi
    
    log_info "Running validation on processed files..."
    
    local pickle_files=()
    while IFS= read -r -d '' file; do
        pickle_files+=("$file")
    done < <(find "$OUTPUT_DIR" -name "*.pkl" -print0 2>/dev/null || true)
    
    if [[ ${#pickle_files[@]} -eq 0 ]]; then
        log_warn "No pickle files found for validation"
        return 0
    fi
    
    # Run validation if script exists
    if [[ -f "data_validation.py" ]]; then
        local validation_cmd="python data_validation.py -f ${pickle_files[*]} -o \"${OUTPUT_DIR}/validation\""
        
        if [[ "$DRY_RUN" == true ]]; then
            echo "DRY RUN: $validation_cmd"
        else
            log_info "Running validation command: $validation_cmd"
            if eval "$validation_cmd"; then
                log_info "Validation completed successfully"
            else
                log_warn "Validation failed or had issues"
            fi
        fi
    else
        log_warn "Validation script not found: data_validation.py"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove individual log files if main log exists
    if [[ -f "$LOG_FILE" ]]; then
        find "$LOG_DIR" -name "batch_processing_${TIMESTAMP}.log.*" -delete 2>/dev/null || true
    fi
    
    log_info "Cleanup completed"
}

# Main execution
main() {
    # Set up error handling
    trap cleanup EXIT
    trap 'log_error "Script interrupted"; exit 130' INT TERM
    
    # Parse arguments
    parse_args "$@"
    
    # Setup
    setup
    
    # Get file list
    local file_list
    readarray -t file_list < <(get_file_list)
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would process ${#file_list[@]} files"
        for file in "${file_list[@]}"; do
            process_file "$file"
        done
        run_validation
        return 0
    fi
    
    # Process files
    if process_files_parallel "${file_list[@]}"; then
        log_info "All files processed successfully"
    else
        log_error "Some files failed to process"
        exit 1
    fi
    
    # Run validation
    run_validation
    
    log_info "Batch processing completed successfully!"
}

# Run main function with all arguments
main "$@"