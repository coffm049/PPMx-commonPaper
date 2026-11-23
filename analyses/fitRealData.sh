#!/bin/bash
#SBATCH --job-name=fitRealData
#SBATCH --output=logs/fitRealData-%j.out
#SBATCH --error=logs/fitRealData-%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu

set -euo pipefail

ANALYSIS_DIR="$HOME/papers/ABCD-adhd/analyses_v3-dl-centered"
SIMULATIONS_DIR="$HOME/software/ProductPartitionModels.jl/simulations"

cd "$ANALYSIS_DIR"
mkdir -p logs output_v3-dl-centered

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"

if [ "$(git -C ~/software/ProductPartitionModels.jl rev-parse --abbrev-ref HEAD)" != "v2.0" ]; then
    echo "Switching ProductPartitionModels.jl to v2.0 branch..."
    git -C ~/software/ProductPartitionModels.jl checkout v2.0
fi

test -f "$ANALYSIS_DIR/output_v3-dl-centered/sampledDF.csv" || {
    echo "Missing required input: $ANALYSIS_DIR/output_v3-dl-centered/sampledDF.csv"
    exit 1
}

wait_for_job() {
    local jobid="$1" label="$2" states state

    while true; do
        states=$(sacct -j "$jobid" --format=State --noheader -P 2>/dev/null | sed '/^$/d' | awk '{print $1}' | sort -u | tr '\n' ' ' | sed 's/ $//')

        if [ -z "$states" ]; then
            echo "$label (job $jobid): waiting for job to appear in sacct..."
            sleep 300
            continue
        fi

        echo "$label (job $jobid) states: $states"

        if echo "$states" | grep -Eq '(^|[[:space:]])COMPLETED($|[[:space:]])'; then
            echo "$label complete."
            return 0
        fi

        if echo "$states" | grep -Eiq '(^|[[:space:]])(FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)($|[[:space:]])'; then
            echo "ERROR: $label (job $jobid) did not complete cleanly: $states"
            return 1
        fi

        sleep 300
    done
}

echo "=== Submitting full-data refit ==="
JOB_FULLDATA=$(sbatch --parsable "$ANALYSIS_DIR/01fullDat-ppmxTot.SLURM")
echo "Submitted full-data refit with job ID $JOB_FULLDATA"

echo "=== Submitting split-half refit ==="
JOB_SUB=$(sbatch --parsable "$ANALYSIS_DIR/01sub-ppmxTotFulla.SLURM")
echo "Submitted split-half refit with job ID $JOB_SUB"

wait_for_job "$JOB_FULLDATA" "full-data refit"
wait_for_job "$JOB_SUB" "split-half refit"

echo "=== REAL DATA FITS COMPLETE ==="
