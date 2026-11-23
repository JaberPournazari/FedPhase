#!/bin/bash
# STEP 1. FedPhase + shrinkage: tune lambda on VALIDATION (seed 0). tau/lookahead fixed from tune3, mu = FedProx's tuned mu.
cd "$(dirname "$0")"; mkdir -p logs
export DATASET=${DATASET:-data2} GROUP=tune4
CODE=${CODE_DIR:-${ROOT:-$HOME/fedphase_root}/fedphase}
cat $CODE/results/$DATASET/tune/best_hparams.env $CODE/results/$DATASET/tune3/best_hparams.env > $CODE/results/$DATASET/tune3/combined.env
export HP_FILE=results/$DATASET/tune3/combined.env
N=$((${CLIENT_NODES:-4}+1)); IDS=()
for l in 0.25 0.5 0.75; do
  IDS+=($(METHOD=fedphase VARIANT="fedphase4__SHRINK=$l" EXTRA_ARGS="--shrink $l" \
      sbatch --parsable --array=0 --job-name=tune4_$l --nodes=$N job_fl_multinode.slurm))
done
C=$(GROUP=tune4 sbatch --parsable --dependency=afterany:$(IFS=:; echo "${IDS[*]}") job_collect.slurm)
echo "tune4: ${#IDS[@]} jobs, collector $C -> fedphase/results/$DATASET/tune4/{tune_table.csv,best_hparams.env}"
