#!/bin/bash
#SBATCH --job-name=realDataPipe
#SBATCH --output=logs/runAllRealData-%j.out
#SBATCH --error=logs/runAllRealData-%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
# Master script to run all real-data ABCD-ADHD analyses.
# Refits run as separate jobs and complete before post-analysis starts.
# 09 then runs as its own job; posterior, baselines, metrics, R, and Quarto run
# within this allocation after the profiles job finishes.
#
# Usage on HPC (either works; #SBATCH lines are comments when run directly):
#   cd ~/papers/ABCD-adhd/analyses_v3-dl-centered && sbatch runAllRealData.sh
#   or ./runAllRealData.sh
#
# Pipeline order:
#   1. Refit jobs: full data, split halves, and standard PPMx
#   2. Subject profiles (09, ~4h)
#   3. Posterior analysis and comparison baselines
#   4. Real-data metric collection
#   5. R figures (covariateDist-ABCD.R)
#   6. Render Quarto presentation

set -e

# ── Fix working directory (sbatch stages script under /var/spool/slurmd/jobID/) ──
cd ~/papers/ABCD-adhd/analyses_v3-dl-centered

# ── Environment ──────────────────────────────────────────────
export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
export R_HOME="/projects/standard/gdc/public/envs/r-salso/lib/R"
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
export PATH="$R_HOME/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# Ensure v2.0 branch on simulations repo
if [ "$(git -C ~/software/ProductPartitionModels.jl rev-parse --abbrev-ref HEAD)" != "v2.0" ]; then
    echo "Switching ProductPartitionModels.jl to v2.0 branch..."
    git -C ~/software/ProductPartitionModels.jl checkout v2.0
fi

# Wait on a submitted job and abort the pipeline if it did not complete
# cleanly (polling squeue alone cannot tell success from failure).
wait_for_job() {
    local jobid="$1" label="$2" st
    while squeue -u "$USER" --job-id "$jobid" 2>/dev/null | grep -q .; do
        sleep 120
    done
    st=$(sacct -j "$jobid" --format=State --noheader -P 2>/dev/null | sort -u | tr '\n' ' ')
    echo "$label sacct states: $st"
    if echo "$st" | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED'; then
        echo "ERROR: $label (job $jobid) did not complete cleanly. Check its logs; aborting pipeline."
        exit 1
    fi
    echo "$label complete."
}

# ── Step 1: Refits ─────────────────────────────────────────
echo "=== Submitting full-data refit ==="
JOB_FULLDATA=$(sbatch --parsable "$HOME/papers/ABCD-adhd/analyses_v3-dl-centered/01fullDat-ppmxTot.SLURM")
echo "Submitted full-data refit with job ID $JOB_FULLDATA"
echo "=== Submitting split-half refit ==="
JOB_SUB=$(sbatch --parsable "$HOME/papers/ABCD-adhd/analyses_v3-dl-centered/01sub-ppmxTotFulla.SLURM")
echo "Submitted split-half refit with job ID $JOB_SUB"
echo "=== Submitting standard PPMx refit ==="
JOB_STD=$(sbatch --parsable "$HOME/papers/ABCD-adhd/analyses_v3-dl-centered/05-ppmxStd-ABCD.sh")
echo "Submitted standard PPMx refit with job ID $JOB_STD"
wait_for_job "$JOB_FULLDATA" "full-data refit"
wait_for_job "$JOB_SUB" "split-half refit"
wait_for_job "$JOB_STD" "standard PPMx refit"

# ── Step 2: Subject Profiles (~4h) ────────────────────────
echo "=== Submitting 09-subjectProfiles ==="
JOB_PROFILES=$(sbatch --parsable "$HOME/papers/ABCD-adhd/analyses_v3-dl-centered/09-subjectProfiles.SLURM")
echo "Submitted 09-subjectProfiles with job ID $JOB_PROFILES"
wait_for_job "$JOB_PROFILES" "09-subjectProfiles"

echo "=== Submitting 06-baselines ==="
JOB_BASELINES=$(sbatch --parsable "$HOME/papers/ABCD-adhd/analyses_v3-dl-centered/06-baselines-ABCD.sh")
echo "Submitted 06-baselines with job ID $JOB_BASELINES"
wait_for_job "$JOB_BASELINES" "06-baselines"

# ── Step 3: Posterior Analysis (~1-2h each) ────────────────
echo "=== Running posterior analyses ==="
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" --threads=2 02-posteriorFull.jl
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" --threads=2 02-posteriorSub.jl
if [ -f "output_v3-dl-centered/subTotalStableCombined.jld2" ] && [ -f "output_v3-dl-centered/subTotal2StableCombined.jld2" ] && [ -f "output_v3-dl-centered/subTotalCombined.jld2" ] && [ -f "output_v3-dl-centered/subTotal2Combined.jld2" ]; then
    julia --project="$HOME/software/ProductPartitionModels.jl/simulations" --threads=2 03subTotStableInference.jl
else
    echo "WARNING: subTotal*.jld2 inputs not found. Skipping 03subTotStableInference."
fi
if [ -f "output_v3-dl-centered/stdPPmxTot.jld2" ]; then
    julia --project="$HOME/software/ProductPartitionModels.jl/simulations" --threads=2 07-subjectDeviance-ABCD.jl
else
    echo "WARNING: output_v3-dl-centered/stdPPmxTot.jld2 not found (produced by 05-ppmxStd-ABCD.sh, ~24h refit). Skipping 07-subjectDeviance."
fi
echo "Posterior analyses complete."

echo "=== Collecting real-data metrics ==="
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" collect_real_metrics.jl

# ── Step 4: R Figures ──────────────────────────────────────
echo "=== Generating R figures ==="
Rscript covariateDist-ABCD.R
echo "R figures complete."

# ── Step 5: Render Quarto Presentation ──────────
if ! command -v quarto &>/dev/null; then
    echo "WARNING: quarto not found. Install with: conda install -c conda-forge quarto"
else
    echo "=== Rendering Quarto presentation ==="
    quarto render vizReal.qmd --to revealjs --output-dir output_v3-dl-centered/presentations
    echo "Quarto presentation rendered to output_v3-dl-centered/presentations/vizReal.html"
fi

echo "=== ALL REAL DATA ANALYSES COMPLETE ==="
