#!/bin/bash
#SBATCH --job-name=analyzeRealData
#SBATCH --output=logs/analyzeRealData-%j.out
#SBATCH --error=logs/analyzeRealData-%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu

set -euo pipefail

ANALYSIS_DIR="$HOME/papers/ABCD-adhd/analyses_v3-dl-centered"
SIMULATIONS_DIR="$HOME/software/ProductPartitionModels.jl/simulations"

cd "$ANALYSIS_DIR"
mkdir -p logs output_v3-dl-centered

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
export R_HOME="/projects/standard/gdc/public/envs/r-salso/lib/R"
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
export PATH="$R_HOME/bin:$PATH"

for required in \
    "$ANALYSIS_DIR/output_v3-dl-centered/openTotalFullDatamcmc1-v2.jld2" \
    "$ANALYSIS_DIR/output_v3-dl-centered/openTotalFullDatamcmc2-v2.jld2" \
    "$ANALYSIS_DIR/output_v3-dl-centered/openTotalFull2-v2.jld2" \
    "$ANALYSIS_DIR/output_v3-dl-centered/openTotalFull3-v2.jld2"; do
    if [ ! -f "$required" ]; then
        echo "Missing required fit output: $required"
        echo "Run fitRealData.sh and wait for it to finish before analyzeRealData.sh."
        exit 1
    fi
done

if [ "$(git -C ~/software/ProductPartitionModels.jl rev-parse --abbrev-ref HEAD)" != "v2.0" ]; then
    echo "Switching ProductPartitionModels.jl to v2.0 branch..."
    git -C ~/software/ProductPartitionModels.jl checkout v2.0
fi

echo "=== Running subject profiles ==="
julia --threads=2 09-subjectProfiles-combinedShrunkK5.jl

echo "=== Running comparison baselines ==="
julia --project="$SIMULATIONS_DIR" 06-baselines-ABCD.jl

echo "=== Running posterior analysis ==="
julia --project="$SIMULATIONS_DIR" --threads=2 02-posteriorFull.jl
julia --project="$SIMULATIONS_DIR" --threads=2 02-posteriorSub.jl

if [ -f "$ANALYSIS_DIR/output_v3-dl-centered/subTotalStableCombined.jld2" ] && \
   [ -f "$ANALYSIS_DIR/output_v3-dl-centered/subTotal2StableCombined.jld2" ] && \
   [ -f "$ANALYSIS_DIR/output_v3-dl-centered/subTotalCombined.jld2" ] && \
   [ -f "$ANALYSIS_DIR/output_v3-dl-centered/subTotal2Combined.jld2" ]; then
    julia --project="$SIMULATIONS_DIR" --threads=2 03subTotStableInference.jl
else
    echo "WARNING: subTotal*.jld2 inputs not found. Skipping 03subTotStableInference."
fi

julia --project="$SIMULATIONS_DIR" --threads=2 07-subjectDeviance-ABCD.jl

echo "=== Collecting real-data metrics ==="
julia --project="$SIMULATIONS_DIR" collect_real_metrics.jl

test -f "$ANALYSIS_DIR/output_v3-dl-centered/baselines/frftotalComparison.csv" || {
    echo "Expected baseline output was not produced: output_v3-dl-centered/baselines/frftotalComparison.csv"
    exit 1
}

echo "=== Generating R figures ==="
Rscript covariateDist-ABCD.R

if ! command -v quarto &>/dev/null; then
    echo "WARNING: quarto not found. Install with: conda install -c conda-forge quarto"
else
    echo "=== Rendering Quarto presentation ==="
    quarto render vizReal.qmd --to revealjs --output-dir output_v3-dl-centered/presentations
fi

echo "=== REAL DATA ANALYSIS COMPLETE ==="
