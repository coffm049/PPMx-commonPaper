#!/bin/bash
#SBATCH --job-name=opentot1-v2
#SBATCH --output=logs/opentot-shrunk-v2.out
#SBATCH --error=logs/opentot1-shrunk-v2.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"

# v2.0 estimator: use the manually-pulled v2.0 checkout (pull manually on HPC
# before submitting; no auto fetch/pull here to avoid concurrent index.lock races).
# Chains go to output/openTotalFull{2,3}-v2.jld2 (see 01sub-ppmxTotFulla-shrunk.jl).
if [ "$(git -C ~/software/ProductPartitionModels.jl rev-parse --abbrev-ref HEAD)" != "v2.0" ]; then
  sleep $((RANDOM % 20))
  git -C ~/software/ProductPartitionModels.jl checkout v2.0
fi
cd ~/papers/ABCD-adhd/analyses

julia 01sub-ppmxTotFulla-shrunk.jl
