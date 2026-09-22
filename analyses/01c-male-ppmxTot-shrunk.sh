#!/bin/bash
#SBATCH --job-name=ppmxMale
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-ppmxMale-%j.out
#SBATCH --error=slurm-ppmxMale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 01c-male-ppmxTot-shrunk.jl
