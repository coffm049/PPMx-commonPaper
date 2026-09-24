#!/bin/bash
#SBATCH --job-name=sexMfit
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --output=slurm-sexMfit-%j.out
#SBATCH --error=slurm-sexMfit-%j.err

module load R/4.4.0-openblas-rocky8

cd ~/papers/ABCD-adhd/analyses
Rscript compareSexStratified-Mfit.R