#!/bin/bash
#SBATCH --job-name=ppmxCommon-nf
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-ppmxCommon-nofemale-%j.out
#SBATCH --error=slurm-ppmxCommon-nofemale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 01bfullDat-ppmxTot-shrunk.jl
