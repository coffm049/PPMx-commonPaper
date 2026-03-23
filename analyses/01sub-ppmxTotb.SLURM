#!/bin/bash
#SBATCH --job-name=opentotb
#SBATCH --output=logs/opentotb.out
#SBATCH --error=logs/opentotb.err
#SBATCH --time=4:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=coffm049@umn.edu


cd ~/software/ProductPartitionModels.jl/ABCD-adhd/analyses

julia 01sub-ppmxTotb.jl
