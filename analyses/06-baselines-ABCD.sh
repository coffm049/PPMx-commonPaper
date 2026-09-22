#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm-baselines-%j.out
#SBATCH --error=slurm-baselines-%j.err

# On HPC, Julia is managed via micromamba:
#   source <path-to-micromamba>/micromamba/etc/profile.d/conda.sh   # if needed
#   micromamba activate juliaup   # or whichever env exposes `julia`
export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

# First-time setup: get the DPM baseline package (DPMM) from the patched fork
#   julia -e 'import Pkg; Pkg.activate("../../simulations/"); Pkg.add(url="https://github.com/coffm049/DPMM.jl"); Pkg.resolve()'

# requires output/stdPPmxTot.jld2 from 05-ppmxStd-ABCD.sh
# sbatch stages a copy of this script under /var/spool/slurmd/jobID/, so
# "$(dirname $0")" would land in the spool dir; use the absolute path instead.
cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 06-baselines-ABCD.jl