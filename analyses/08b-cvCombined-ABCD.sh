#!/bin/bash
#SBATCH --job-name=cvCombined-nf
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-cvCombined-nofemale-%j.out
#SBATCH --error=slurm-cvCombined-nofemale-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" 08b-cvCombined-ABCD.jl
