#!/bin/bash
# Main comparison: 8 methods x 5 seeds, tuned hyper-parameters (run submit_tune.sh first).
#   bash submit_main.sh                   # everything
#   bash submit_main.sh fedphase ditto    # subset
#   SEEDS=1-3 ROUNDS=60 bash submit_main.sh
cd "$(dirname "$0")"; mkdir -p logs
export DATASET=${DATASET:-data2} GROUP=main
CODE_DIR=${CODE_DIR:-${ROOT:-$HOME/fedphase_root}/fedphase}
export HP_FILE=results/$DATASET/tune/best_hparams.env
[ -f "$CODE_DIR/$HP_FILE" ] && echo "using $(tr '\n' ' ' < $CODE_DIR/$HP_FILE)" \
  || echo "WARNING: $HP_FILE not found -> default hyper-parameters (run submit_tune.sh first)"
SEEDS=${SEEDS:-1-5}; N=$((${CLIENT_NODES:-4}+1))
METHODS=${@:-centralized local fedavg fedprox fedper ditto ifca fedphase}
IDS=()
for M in $METHODS; do
  case $M in
    local)       ID=$(METHOD=$M sbatch --parsable --array=$SEEDS --job-name=main_$M --exclusive --cpus-per-task=168 --mem=0 job_method.slurm) ;;
    centralized) ID=$(METHOD=$M sbatch --parsable --array=$SEEDS --job-name=main_$M --cpus-per-task=32 --mem=64G job_method.slurm) ;;
    *)           ID=$(METHOD=$M sbatch --parsable --array=$SEEDS --job-name=main_$M --nodes=$N job_fl_multinode.slurm) ;;
  esac
  echo "$M -> $ID"; IDS+=($ID)
done
C=$(GROUP=main sbatch --parsable --dependency=afterany:$(IFS=:; echo "${IDS[*]}") job_collect.slurm)
echo "collector -> $C   results -> $CODE_DIR/results/$DATASET/main/"
