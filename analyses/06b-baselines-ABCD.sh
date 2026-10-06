#!/bin/bash
#SBATCH --job-name=baselines-nf
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm-baselines-nofemale-%j.out
#SBATCH --error=slurm-baselines-nofemale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"

# SALSO metrics (R 'salso' package via RCall)
module unload R 2>/dev/null || true
export R_HOME="/projects/standard/gdc/public/envs/r-salso/lib/R"
export LD_LIBRARY_PATH="$R_HOME/lib:$LD_LIBRARY_PATH"
export PATH="$R_HOME/bin:$PATH"

which julia

# requires output/stdPPmxTot.jld2 from 05-ppmxStd-ABCD.sh
cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 06b-baselines-ABCD.jl
