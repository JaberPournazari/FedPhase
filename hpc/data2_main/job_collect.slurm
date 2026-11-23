#!/bin/bash
#SBATCH --job-name=collect
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
GROUP=${GROUP:-main}
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg
python collect_results.py --dataset "$DATASET" --group "$GROUP" --protocol "${PROTOCOL:-final}"
