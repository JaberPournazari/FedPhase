#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Quick end-to-end check of every method on one node: 8 cells, 3 rounds.
PARTS=${PARTS:-8}; ROUNDS=${ROUNDS:-3}
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg OMP_NUM_THREADS=1
mkdir -p logs
python -c "import flwr, torch; print('flwr', flwr.__version__, 'torch', torch.__version__)"
rm -rf client_logs/$DATASET/smoke server_logs/$DATASET/smoke results/$DATASET/smoke
PORT=$((30000 + SLURM_JOB_ID % 20000)); FAIL=""
for M in fedavg fedprox fedper ditto ifca fedphase; do
  PORT=$((PORT+1))
  python run_seeds.py --method $M --seeds 1 --num-partitions $PARTS --num-rounds $ROUNDS \
     --local-epochs 1 --port $PORT --run-tag $DATASET/smoke/$M $DATA_ARGS -- $SRV_HP \
     > logs/smoke_$M.log 2>&1 && echo "OK   $M" || { echo "FAIL $M (logs/smoke_$M.log)"; FAIL="$FAIL $M"; }
done
for M in local centralized; do
  python baselines.py --mode $M --seed 1 --num-partitions $PARTS --num-rounds $ROUNDS --local-epochs 1 \
     $DATA_ARGS --workers 8 --run-tag $DATASET/smoke/$M > logs/smoke_$M.log 2>&1 \
     && echo "OK   $M" || { echo "FAIL $M"; FAIL="$FAIL $M"; }
done
python collect_results.py --dataset $DATASET --group smoke
[ -z "$FAIL" ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED:$FAIL"
