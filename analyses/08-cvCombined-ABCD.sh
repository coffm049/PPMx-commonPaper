#!/bin/bash
#SBATCH --job-name=cvCombined
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-cvCombined-%j.out
#SBATCH --error=slurm-cvCombined-%j.err

export PATH="$HOME/software/julia-1.11.7/bin:$PATH"
which julia

cd "$(dirname "$0")"
julia --project="../../simulations/" 08-cvCombined-ABCD.jl
