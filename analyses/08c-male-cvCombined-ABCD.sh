#!/bin/bash
#SBATCH --job-name=cvMale
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-cvMale-%j.out
#SBATCH --error=slurm-cvMale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 08c-male-cvCombined-ABCD.jl
