#!/bin/bash
#SBATCH --job-name=d1
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
# One method x one seed (SEED = array index) on ONE node: server + 15 clients (data1 is small).
# Env: METHOD (required) GROUP VARIANT HP_FILE EXTRA_ARGS + everything in common.sh
METHOD=${METHOD:?set METHOD}; GROUP=${GROUP:-main}; VARIANT=${VARIANT:-$METHOD}
SEED=${SLURM_ARRAY_TASK_ID:-1}
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
TAG="$DATASET/$GROUP/$VARIANT"
rm -rf "client_logs/$TAG/seed_$SEED" "server_logs/$TAG/seed_$SEED"
echo "tag=$TAG seed=$SEED method=$METHOD host=$(hostname)"
case $METHOD in
  local|centralized)
    [ "$METHOD" = centralized ] && export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
    python baselines.py --mode $METHOD --seed $SEED --num-partitions $PARTS --num-rounds $ROUNDS \
        --local-epochs $EPOCHS --lr $LR $DATA_ARGS --workers ${SLURM_CPUS_PER_TASK:-16} --run-tag "$TAG" ;;
  *)
    echo "server hp: $SRV_HP"
    python run_seeds.py --method $METHOD --seeds $SEED --num-partitions $PARTS --num-rounds $ROUNDS \
        --local-epochs $EPOCHS --port 0 --run-tag "$TAG" $DATA_ARGS --quiet -- $SRV_HP ;;
esac
RC=$?
echo "done rc=$RC: client_logs/$TAG/seed_$SEED ($(ls client_logs/$TAG/seed_$SEED/error_client*.csv 2>/dev/null | wc -l)/$PARTS clients)"
exit $RC
