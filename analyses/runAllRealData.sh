#!/bin/bash
#SBATCH --job-name=realDataPipe
#SBATCH --output=logs/runAllRealData-%j.out
#SBATCH --error=logs/runAllRealData-%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu
# Master script to run all real-data ABCD-ADHD post-analysis steps.
# Submit 09 as its own job and wait; posterior/R/quarto steps run on this
# allocation (julia --threads=2), so this job needs the CPUs/memory above.
#
# Usage on HPC (either works; #SBATCH lines are comments when run directly):
#   cd ~/papers/ABCD-adhd/analyses && sbatch runAllRealData.sh
#   or ./runAllRealData.sh
#
# Pipeline order (reads existing MCMC chain files, does NOT refit):
#   1. Subject profiles (09, ~4h)
#   2. Posterior analysis (02, 03, 07, ~1-2h each)
#   3. R figures (covariateDist-ABCD.R)
#   4. Render Quarto presentation

set -e

# ── Fix working directory (sbatch stages script under /var/spool/slurmd/jobID/) ──
cd ~/papers/ABCD-adhd/analyses

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

# ── Step 1: Subject Profiles (~4h) ────────────────────────
echo "=== Submitting 09-subjectProfiles ==="
JOB_PROFILES=$(sbatch --parsable "$HOME/papers/ABCD-adhd/analyses/09-subjectProfiles.SLURM")
echo "Submitted 09-subjectProfiles with job ID $JOB_PROFILES"
wait_for_job "$JOB_PROFILES" "09-subjectProfiles"

# ── Step 2: Posterior Analysis (~1-2h each) ────────────────
echo "=== Running posterior analyses ==="
julia --project="../../simulations/" --threads=2 02-posteriorFull.jl
julia --project="../../simulations/" --threads=2 02-posteriorSub.jl
if [ -f "output/subTotalStableCombined.jld2" ] && [ -f "output/subTotal2StableCombined.jld2" ] && [ -f "output/subTotalCombined.jld2" ] && [ -f "output/subTotal2Combined.jld2" ]; then
    julia --project="../../simulations/" --threads=2 03subTotStableInference.jl
else
    echo "WARNING: subTotal*.jld2 inputs not found. Skipping 03subTotStableInference."
fi
if [ -f "output/stdPPmxTot.jld2" ]; then
    julia --project="../../simulations/" --threads=2 07-subjectDeviance-ABCD.jl
else
    echo "WARNING: output/stdPPmxTot.jld2 not found (produced by 05-ppmxStd-ABCD.sh, ~24h refit). Skipping 07-subjectDeviance."
fi
echo "Posterior analyses complete."

# ── Step 3: R Figures ──────────────────────────────────────
echo "=== Generating R figures ==="
Rscript covariateDist-ABCD.R
echo "R figures complete."

# ── Step 4: Render Quarto Presentation ──────────────────────
# vizReal.qmd degrades gracefully when optional inputs are absent
# (CV slide needs output/baselines/combined5foldCV.csv from 08;
# baseline table needs output/baselines/*.csv from 06).
if [ ! -f "output/baselines/combined5foldCV.csv" ]; then
    echo "NOTE: output/baselines/combined5foldCV.csv not found (produced by 08-cvCombined-ABCD.sh, ~48h refit). Deck will render without the CV slide."
fi
if ! command -v quarto &>/dev/null; then
    echo "WARNING: quarto not found. Install with: conda install -c conda-forge quarto"
else
    echo "=== Rendering Quarto presentation ==="
    quarto render vizReal.qmd --to revealjs --output-dir output/presentations
    echo "Quarto presentation rendered to output/presentations/vizReal.html"
fi

echo "=== ALL REAL DATA ANALYSES COMPLETE ==="
