#!/bin/bash
#SBATCH --job-name=paper
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Re-collect every group that exists, then dataset section + paper tables/figures.
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg
for g in $(ls client_logs/$DATASET 2>/dev/null); do
  case $g in tune*|smoke) continue ;; esac
  python collect_results.py --dataset $DATASET --group $g --protocol final > logs_collect_$g.txt 2>&1 \
    && echo "collected $g" || echo "collect FAILED: $g (see fedphase/logs_collect_$g.txt)"
done
[ -d client_logs/$DATASET/main ] && python collect_results.py --dataset $DATASET --group main --protocol bestval > /dev/null
python dataset_overview.py --dataset $DATASET --seq-len $SEQ --train-frac $TRAIN_FRAC --val-frac $VAL_FRAC
python paper_results.py --dataset $DATASET --boot ${BOOT:-5000}
echo "-> $CODE_DIR/results/$DATASET/paper/"
