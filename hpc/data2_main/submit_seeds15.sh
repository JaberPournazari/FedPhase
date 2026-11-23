#!/bin/bash
# Extra seeds (default 6-15) for EVERY method in the main group, each with its own tuned hyper-parameters,
# so the paper table is paired on 15 seeds. Existing seeds 1-5 are kept. Then: collectors + paper_results.py.
#   bash submit_seeds15.sh                 # all methods, seeds 6-15
#   SEEDS=6-10 bash submit_seeds15.sh fedphase fedprox
cd "$(dirname "$0")"; mkdir -p logs
export DATASET=${DATASET:-data2} GROUP=main
CODE=${CODE_DIR:-${ROOT:-$HOME/fedphase_root}/fedphase}
R=$CODE/results/$DATASET
for f in $R/tune/best_hparams.env $R/tune4/combined.env; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
SEEDS=${SEEDS:-6-15}; N=$((${CLIENT_NODES:-4}+1))
METHODS=${@:-centralized local fedavg fedprox fedper ditto ifca fedphase}
IDS=()
for M in $METHODS; do
  case $M in   # same settings as the seed 1-5 runs currently in main/
    fedphase) HP=results/$DATASET/tune4/combined.env; X="" ;;               # mu=0.001, tau=0.5, LA=1.0, shrink=0.75
    ditto)    HP=results/$DATASET/tune/best_hparams.env; X="--ditto-lambda 0.01" ;;
    *)        HP=results/$DATASET/tune/best_hparams.env; X="" ;;
  esac
  case $M in
    local)       ID=$(METHOD=$M HP_FILE=$HP sbatch --parsable --array=$SEEDS --job-name=s15_$M --exclusive --cpus-per-task=168 --mem=0 job_method.slurm) ;;
    centralized) ID=$(METHOD=$M HP_FILE=$HP sbatch --parsable --array=$SEEDS --job-name=s15_$M --cpus-per-task=32 --mem=64G job_method.slurm) ;;
    *)           ID=$(METHOD=$M HP_FILE=$HP EXTRA_ARGS="$X" sbatch --parsable --array=$SEEDS --job-name=s15_$M --nodes=$N job_fl_multinode.slurm) ;;
  esac
  echo "$M -> $ID"; IDS+=($ID)
done
DEP=afterany:$(IFS=:; echo "${IDS[*]}")
C1=$(GROUP=main PROTOCOL=final   sbatch --parsable --dependency=$DEP job_collect.slurm)
C2=$(GROUP=main PROTOCOL=bestval sbatch --parsable --dependency=$DEP job_collect.slurm)
P=$(sbatch --parsable --dependency=afterany:$C1:$C2 --job-name=paper --output=logs/paper_%j.out \
      --partition=compute --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 \
      --wrap="source $(pwd)/common.sh; export MPLBACKEND=Agg; python paper_results.py --dataset $DATASET --boot 5000")
echo "collectors $C1 $C2, paper $P -> $R/paper/  (log: hpc/<group>/logs/paper_$P.out)"
