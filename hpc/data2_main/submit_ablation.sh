#!/bin/bash
# FedPhase ablation (tuned hp, 5 seeds). Each variant changes ONE component.
cd "$(dirname "$0")"; mkdir -p logs
export DATASET=${DATASET:-data2} GROUP=ablation
export HP_FILE=results/$DATASET/tune/best_hparams.env
SEEDS=${SEEDS:-1-5}; N=$((${CLIENT_NODES:-4}+1)); IDS=()
declare -A V=(
  [full]=""
  [no_adaptive_mu]="--adaptive-mu 0"        # mu_i = mu for every client
  [feat_only]="--phase-set feat"            # phase from input signals only (no capacity curve)
  [label_only]="--phase-set label"          # phase from capacity curve only
  [no_prox]="--mu 0"                        # phase-personalised aggregation only
  [uniform_kernel]="--tau-scale 1e6"        # no phase information  (== FedProx)
  [knn10]="--knn 10"                        # sparse: 10 nearest phases
)
for k in "${!V[@]}"; do
  ID=$(METHOD=fedphase VARIANT=$k EXTRA_ARGS="${V[$k]}" sbatch --parsable --array=$SEEDS \
       --job-name=abl_$k --nodes=$N job_fl_multinode.slurm); echo "$k -> $ID"; IDS+=($ID)
done
C=$(GROUP=ablation sbatch --parsable --dependency=afterany:$(IFS=:; echo "${IDS[*]}") job_collect.slurm)
echo "collector -> $C"
