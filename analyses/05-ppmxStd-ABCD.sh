#!/bin/bash
#SBATCH --job-name=ppmxStd
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-ppmxStd-%j.out
#SBATCH --error=slurm-ppmxStd-%j.err

# On HPC, Julia is managed via micromamba:
#   source <path-to-micromamba>/micromamba/etc/profile.d/conda.sh   # if needed
#   micromamba activate juliaup   # or whichever env exposes `julia`
which julia

# First-time setup: get the DPM baseline package (DPMM) from the patched fork
#   julia -e 'import Pkg; Pkg.activate("../../simulations/"); Pkg.add(url="https://github.com/coffm049/DPMM.jl"); Pkg.resolve()'

julia --project="../../simulations/" analyses/05-ppmxStd-ABCD.jl