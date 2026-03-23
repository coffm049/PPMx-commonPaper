#!/bin/bash
#SBATCH --job-name=opentot1
#SBATCH --output=logs/opentot-shrunk.out
#SBATCH --error=logs/opentot1-shrunk.err
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu


cd ~/software/ProductPartitionModels.jl/ABCD-adhd/analyses

julia 01sub-ppmxTotFulla-shrunk.jl
