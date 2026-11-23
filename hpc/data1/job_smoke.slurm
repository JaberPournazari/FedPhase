#!/bin/bash
#SBATCH --job-name=d1_smoke
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Every method on all 15 data1 cells, 3 rounds, 1 epoch (+ positional inputs, + dataset overview).
ROUNDS=${ROUNDS:-3}; EPOCHS=1
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg OMP_NUM_THREADS=1
mkdir -p logs
python -c "import flwr, torch, task; print('flwr', flwr.__version__, 'torch', torch.__version__, '| format', task.FORMAT, '| inputs', task.N_FEATURES, '| cells', task.num_cells())"
rm -rf client_logs/$DATASET/smoke server_logs/$DATASET/smoke results/$DATASET/smoke*
FAIL=""
for M in fedavg fedprox fedper ditto ifca fedphase; do
  X=""; [ $M = fedphase ] && X="--lookahead 1.0 --shrink 0.75"
  python run_seeds.py --method $M --seeds 1 --num-partitions $PARTS --num-rounds $ROUNDS --local-epochs 1 \
     --port 0 --run-tag $DATASET/smoke/$M $DATA_ARGS -- $SRV_HP $X > logs/smoke_$M.log 2>&1 \
     && echo "OK   $M" || { echo "FAIL $M (logs/smoke_$M.log)"; FAIL="$FAIL $M"; }
done
for M in local centralized; do
  python baselines.py --mode $M --seed 1 --num-partitions $PARTS --num-rounds $ROUNDS --local-epochs 1 \
     $DATA_ARGS --workers 8 --run-tag $DATASET/smoke/$M > logs/smoke_$M.log 2>&1 \
     && echo "OK   $M" || { echo "FAIL $M (logs/smoke_$M.log)"; FAIL="$FAIL $M"; }
done
FEAT_MODE=positional python run_seeds.py --method fedavg --seeds 1 --num-partitions $PARTS --num-rounds $ROUNDS \
   --local-epochs 1 --port 0 --run-tag $DATASET/smoke/fedavg__positional $DATA_ARGS -- $SRV_HP > logs/smoke_positional.log 2>&1 \
   && echo "OK   fedavg (positional inputs)" || { echo "FAIL positional"; FAIL="$FAIL positional"; }
python collect_results.py --dataset $DATASET --group smoke || FAIL="$FAIL collect"
python dataset_overview.py --dataset $DATASET --seq-len $SEQ --train-frac $TRAIN_FRAC --val-frac $VAL_FRAC || FAIL="$FAIL overview"
[ -z "$FAIL" ] && echo "SMOKE TEST PASSED" || echo "SMOKE TEST FAILED:$FAIL"
