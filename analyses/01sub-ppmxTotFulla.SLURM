#!/bin/bash
#SBATCH --job-name=opentot1-v2
#SBATCH --output=logs/opentot-shrunk-v2.out
#SBATCH --error=logs/opentot1-shrunk-v2.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"

# v2.0 estimator: pin package version so refits never mix with previous fits.
# Chains go to output/openTotalFull{2,3}-v2.jld2 (see 01sub-ppmxTotFulla-shrunk.jl).
cd ~/software/ProductPartitionModels.jl
git fetch origin --quiet
git checkout v2.0
git pull --ff-only origin v2.0
cd ~/papers/ABCD-adhd/analyses

julia 01sub-ppmxTotFulla-shrunk.jl
