#!/bin/bash
#SBATCH --job-name=fl
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# One FL method x one seed (SEED = array index). node 0 = server, nodes 1..K = clients.
# Env: METHOD (required) GROUP VARIANT HP_FILE EXTRA_ARGS + everything in common.sh
METHOD=${METHOD:?set METHOD}; GROUP=${GROUP:-main}; VARIANT=${VARIANT:-$METHOD}
SEED=${SLURM_ARRAY_TASK_ID:-1}
THREADS=${THREADS:-1}
source "$(dirname "$0")/common.sh" 2>/dev/null || source "$SLURM_SUBMIT_DIR/common.sh"
export MPLBACKEND=Agg OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
TAG="$DATASET/$GROUP/$VARIANT"
PORT=$((20000 + SLURM_JOB_ID % 20000))
rm -rf "client_logs/$TAG/seed_$SEED" "server_logs/$TAG/seed_$SEED"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
SERVER=${NODES[0]}; CLIENT_NODES=("${NODES[@]:1}"); K=${#CLIENT_NODES[@]}
PER=$(( (PARTS + K - 1) / K )); CPN=${SLURM_CPUS_ON_NODE:-168}
echo "tag=$TAG seed=$SEED server=$SERVER clients=${CLIENT_NODES[*]} ($PER/node)"
echo "server hp: $SRV_HP"

srun -N1 -n1 -c "$CPN" -w "$SERVER" --exact python server_app.py --method $METHOD \
    --num-rounds $ROUNDS --local-epochs $EPOCHS --address 0.0.0.0:$PORT \
    --num-clients $PARTS --seed $SEED --run-tag "$TAG" $SRV_HP &
for i in $(seq 0 $((K-1))); do
  S=$((i*PER)); E=$(( S+PER < PARTS ? S+PER : PARTS )); [ $S -ge $PARTS ] && break
  srun -N1 -n1 -c "$CPN" -w "${CLIENT_NODES[$i]}" --exact python run_clients_node.py \
      --start $S --end $E --server-address $SERVER:$PORT --seed $SEED --run-tag "$TAG" --quiet \
      --method $METHOD $DATA_ARGS &
done
wait
echo "done: client_logs/$TAG/seed_$SEED ($(ls client_logs/$TAG/seed_$SEED/error_client*.csv 2>/dev/null | wc -l)/$PARTS clients)"
