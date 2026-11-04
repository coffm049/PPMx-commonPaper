#!/usr/bin/env bash
# Read-only search for the split-half chains/outputs that produced modal K=8.
# Run on the HPC login node, ideally from the analyses/ directory:
#   bash find_k8.sh
#
# Nothing is written or modified. Adjust ROOTS if your HPC layout differs.

set -u
export LC_ALL=C

ROOTS=(
  "$HOME/papers"
  "$HOME/software"
  "$HOME/projects"
  "$HOME/scratch"
  "$HOME/PPMx-commonPaper"
  "$HOME/ABCD-adhd"
  # "$HOME"   # uncomment for a full home scan (slower)
)

EXISTING=()
for d in "${ROOTS[@]}"; do
  [ -d "$d" ] && EXISTING+=("$d")
done

if [ ${#EXISTING[@]} -eq 0 ]; then
  echo "No search roots exist. Edit ROOTS in this script." >&2
  exit 1
fi

echo "Search roots:"
printf '  %s\n' "${EXISTING[@]}"
echo

echo "=== 1. Candidate JLD2 chain files ==="
find "${EXISTING[@]}" -type f \( \
    -iname '*openTotal*.jld2' -o \
    -iname '*mcmc*.jld2' -o \
    -iname '*split*.jld2' -o \
    -iname '*ARM*.jld2' \
  \) -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort -u

echo
echo "=== 2. Candidate scripts / SLURM files ==="
find "${EXISTING[@]}" -type f \( \
    -name '01sub-ppmxTota.jl' -o \
    -name '01sub-ppmxTotb.jl' -o \
    -iname '*ppmxTot*.jl' -o \
    -iname '*split*.jl' -o \
    -iname '*ARM*.jl' -o \
    -iname '*ppmxTot*.SLURM' -o \
    -iname '*ppmxTot*.sh' \
  \) -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort -u

echo
echo "=== 3. @save / output lines in those scripts ==="
while IFS= read -r f; do
  echo "--- $f"
  grep -nE '@save|openTotal|openTotalm|mcmc|output/' "$f" 2>/dev/null | head -n 30
done < <(find "${EXISTING[@]}" -type f \( -name '*.jl' -o -name '*.SLURM' -o -name '*.sh' \) 2>/dev/null | grep -Ei 'ppmxTot|split|ARM|openTotal' | sort -u)

echo
echo "=== 4. Text evidence of K=8 / modal K / cluster-count tables ==="
find "${EXISTING[@]}" -type f \( \
    -name '*.tex' -o -name '*.md' -o -name '*.txt' -o \
    -name '*.csv' -o -name '*.log' -o -name '*.out' -o -name '*.err' -o \
    -name '*.jl' -o -name '*.R' \
  \) -size -20M -print0 2>/dev/null | \
  xargs -0 -r grep -nIE \
    'CbestRind: 8|K_?n? ?= ?8|K ?= ?8|modal[^.\n]{0,40}8|8[^.\n]{0,40}split|split[^.\n]{0,40}8|NumberofClusters|posterior mode' \
    2>/dev/null | head -n 400

echo
echo "=== 5. Cluster-count images / fit tables ==="
find "${EXISTING[@]}" -type f \( \
    -iname '*NumberofClusters*.png' -o \
    -iname '*SizeofClusters*.png' -o \
    -iname '*ppmFit*.tex' -o \
    -iname '*GroupCovarDifferences*.tex' \
  \) -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' 2>/dev/null | sort -u

echo
echo "Done. To inspect modal K in the JLD2 files, run:"
echo "  julia --project=~/software/ProductPartitionModels.jl/simulations inspect_jld2_k.jl FILE..."
