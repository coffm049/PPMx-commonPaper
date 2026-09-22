#!/bin/bash
#SBATCH --job-name=ppmxStd-nf
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-ppmxStd-nofemale-%j.out
#SBATCH --error=slurm-ppmxStd-nofemale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 05b-ppmxStd-ABCD.jl
