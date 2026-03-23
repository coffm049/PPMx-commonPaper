#!/bin/bash
#SBATCH --job-name=opentot1
#SBATCH --output=logs/opentot1.out
#SBATCH --error=logs/opentot1.err
#SBATCH --time=4:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu


cd ~/software/ProductPartitionModels.jl/ABCD-adhd/analyses

julia 01sub-ppmxTota.jl
