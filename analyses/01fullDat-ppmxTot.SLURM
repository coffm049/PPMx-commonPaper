#!/bin/bash
#SBATCH --job-name=opentotFullData
#SBATCH --output=logs/opentotFullData-shrunk.out
#SBATCH --error=logs/opentotFullData-shrunk.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu

cd ~/software/ProductPartitionModels.jl/ABCD-adhd/analyses
julia 01fullDat-ppmxTot-shrunk.jl
