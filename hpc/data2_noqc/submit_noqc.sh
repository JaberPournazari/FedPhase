#!/bin/bash
# Sensitivity: MIT-140 WITHOUT the charge-capacity inputs (QC, QC_rollmean, QC_rollstd, QC_slope, QC_cummax:
# 27 -> 22) and WITHOUT phi^qc (qc_change) in the phase vector. Hyper-parameters transferred from the paper.
#   bash submit_noqc.sh                 # FedAvg, FedProx, FedPhase, Direction-only; seeds 1-15 + analysis
#   SEEDS=1-5 bash submit_noqc.sh fedphase
#   bash submit_noqc.sh fedavg fedprox fedphase dironly local centralized fedper ditto ifca   # full table
source "$(dirname "$0")/lib.sh"
G=noqc; SEEDS=${SEEDS:-1-15}
METHODS=${@:-fedavg fedprox fedphase dironly}
IDS=()
for M in $METHODS; do
  case $M in   # same settings as the main/ (and dironly15/) runs with QC
    fedphase)    ID=$(fl fedphase $G fedphase         $HP_MAIN $SEEDS "") ;;
    dironly)     ID=$(fl fedphase $G fedphase_dironly $HP_MAIN $SEEDS "--tau-scale 1e6") ;;
    ditto)       ID=$(fl ditto    $G ditto            $HP_BASE $SEEDS "--ditto-lambda 0.01") ;;
    fedavg|fedprox|fedper|ifca)
                 ID=$(fl $M       $G $M               $HP_BASE $SEEDS "") ;;
    local)       ID=$(METHOD=local GROUP=$G HP_FILE=$HP_BASE sbatch --parsable --array=$SEEDS --job-name=v5_local \
                        --exclusive --cpus-per-task=168 --mem=0 job_method.slurm) ;;
    centralized) ID=$(METHOD=centralized GROUP=$G HP_FILE=$HP_BASE sbatch --parsable --array=$SEEDS \
                        --job-name=v5_centralized --cpus-per-task=32 --mem=64G job_method.slurm) ;;
    *) echo "unknown method $M"; exit 1 ;;
  esac
  echo "$M -> $ID"; IDS+=($ID)
done
A=$(ana afterany:$(j "${IDS[@]}") "compare_noqc.py --dataset $DATASET --group $G --boot 5000")
echo "noqc seeds $SEEDS: ${IDS[*]}, analysis -> $A  =>  $CODE/$R/$G/"
