#!/bin/bash
#SBATCH --job-name=collect_real
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-collect_real-%j.out
#SBATCH --error=slurm-collect_real-%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate julia

cd ~/papers/ABCD-adhd/analyses
julia --project="$HOME/software/ProductPartitionModels.jl/simulations" collect_real_metrics.jl

echo "Real-world metrics collected"
ls -la paper_real_*.csv