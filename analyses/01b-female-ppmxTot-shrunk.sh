#!/bin/bash
#SBATCH --job-name=ppmxFemale
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-ppmxFemale-%j.out
#SBATCH --error=slurm-ppmxFemale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 01b-female-ppmxTot-shrunk.jl
