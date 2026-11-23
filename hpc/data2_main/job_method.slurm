#!/bin/bash
#SBATCH --job-name=cmp
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Local / Centralized, one seed per array task. Env: METHOD=local|centralized GROUP
METHOD=${METHOD:?set METHOD}; GROUP=${GROUP:-main}; SEED=${SLURM_ARRAY_TASK_ID:-1}
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TAG="$DATASET/$GROUP/$METHOD"
rm -rf "client_logs/$TAG/seed_$SEED"
[ "$METHOD" = centralized ] && export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python baselines.py --mode $METHOD --seed $SEED --num-partitions $PARTS --num-rounds $ROUNDS \
    --local-epochs $EPOCHS --lr $LR $DATA_ARGS --workers ${SLURM_CPUS_PER_TASK:-0} --run-tag "$TAG"
