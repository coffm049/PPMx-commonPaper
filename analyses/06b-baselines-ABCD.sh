#!/bin/bash
#SBATCH --job-name=baselines-nf
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-baselines-nofemale-%j.out
#SBATCH --error=slurm-baselines-nofemale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

# requires output/stdPPmxTot.jld2 from 05-ppmxStd-ABCD.sh
cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 06b-baselines-ABCD.jl
