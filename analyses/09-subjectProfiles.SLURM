#!/bin/bash
#SBATCH --job-name=profilesK5-v2
#SBATCH --output=logs/profilesK5-v2.out
#SBATCH --error=logs/profilesK5-v2.err
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"

# optional: needed only for SALSO metrics (R 'salso' package)
module unload R 2>/dev/null || true
export R_HOME="/projects/standard/gdc/public/envs/r-salso/lib/R"
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
# R (conda) needs a newer libstdc++ (CXXABI_1.3.15) than Julia bundles; preload it so
# R's libicui18n resolves correctly, while keeping LD_LIBRARY_PATH limited to R_HOME/lib
# so Julia's own Pango still loads (conda's older libpangoft2 must NOT be exposed).
export LD_PRELOAD="/projects/standard/gdc/public/envs/r-salso/lib/libstdc++.so.6"
export PATH="$R_HOME/bin:$PATH"

# v2.0 estimator: pin package version so profiles never mix with previous fits.
# Reads output/openTotalFull{2,3}-v2.jld2, writes output/openTotal/subjectProfilesK5-v2/.
cd ~/software/ProductPartitionModels.jl
git fetch origin --quiet
git checkout v2.0
git pull --ff-only origin v2.0
cd ~/papers/ABCD-adhd/analyses

julia --threads=2 09-subjectProfiles-combinedShrunkK5.jl
