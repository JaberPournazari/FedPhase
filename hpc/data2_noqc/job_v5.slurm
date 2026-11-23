#!/bin/bash
#SBATCH --job-name=v5_ana
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Runs one fedphase analysis script inside fedphase.  Env: ARGS = "<script.py> <args>"
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg
set -- $ARGS
python "$@"
