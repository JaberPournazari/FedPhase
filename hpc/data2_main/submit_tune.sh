#!/bin/bash
# Hyper-parameter tuning on VALIDATION (seed 0 only; the main runs use seeds 1-5).
# Stage 1: FedProx mu, Ditto lambda, IFCA K  -> collect -> best_hparams.env
# Stage 2: FedPhase tau, with FedPhase's mu FIXED to FedProx's tuned mu
#          (same tuning budget: 4 configs each) -> collect again.
cd "$(dirname "$0")"; mkdir -p logs
export DATASET=${DATASET:-data2} GROUP=tune
CODE_DIR=${CODE_DIR:-${ROOT:-$HOME/fedphase_root}/fedphase}
N=$((${CLIENT_NODES:-4}+1)); IDS=()
sub() {  # METHOD VARIANT EXTRA_ARGS [dependency]
  local dep=${4:+--dependency=afterok:$4}
  METHOD=$1 VARIANT=$2 EXTRA_ARGS="$3" HP_FILE=${HP_FILE:-} sbatch --parsable --array=0 \
     --job-name=tune_$2 --nodes=$N $dep job_fl_multinode.slurm
}
for v in 0.001 0.01 0.1 1.0; do IDS+=($(sub fedprox "fedprox__MU=$v"          "--mu $v")); done
for v in 0.01 0.1 0.5 1.0;   do IDS+=($(sub ditto   "ditto__DITTO_LAMBDA=$v"  "--ditto-lambda $v")); done
for v in 2 3 4 5;            do IDS+=($(sub ifca    "ifca__IFCA_K=$v"         "--ifca-k $v")); done
C1=$(GROUP=tune sbatch --parsable --dependency=afterany:$(IFS=:; echo "${IDS[*]}") job_collect.slurm)
echo "stage 1: ${#IDS[@]} jobs, collector $C1"
export HP_FILE=results/$DATASET/tune/best_hparams.env     # read at job start -> tuned MU
ID2=()
for v in 0.25 0.5 1.0 2.0; do ID2+=($(sub fedphase "fedphase__TAU=$v" "--tau-scale $v" $C1)); done
C2=$(GROUP=tune sbatch --parsable --dependency=afterany:$(IFS=:; echo "${ID2[*]}") job_collect.slurm)
echo "stage 2: ${#ID2[@]} fedphase jobs, collector $C2"
echo "-> $CODE_DIR/results/$DATASET/tune/{tune_table.csv,best_hparams.env}"
