# Helpers for submit_*.sh (sourced on the login node from hpc/<group>/).
cd "$(dirname "${BASH_SOURCE[0]}")"; mkdir -p logs
export DATASET=${DATASET:-data2} FEATURE_SET=${FEATURE_SET:-noqc}
ROOT=${ROOT:-$HOME/fedphase_root}
CODE=${CODE_DIR:-$ROOT/fedphase}
V2=$ROOT/fedphase
R=results/$DATASET                                     # relative to CODE (jobs cd there)
# hyper-parameters TRANSFERRED from the paper's main runs (no re-tuning), read from fedphase (absolute paths)
HP_MAIN=$V2/$R/tune4/combined.env                      # FedPhase: mu=1e-3, tau=0.5, LA(kappa)=1, shrink=0.75
HP_BASE=$V2/$R/tune/best_hparams.env                   # baselines (same as submit_seeds15.sh)
N=$((${CLIENT_NODES:-4}+1))
for f in $HP_MAIN $HP_BASE; do [ -f $f ] || { echo "missing $f"; exit 1; }; done
_dep() { [ -n "$1" ] && echo "--dependency=$1"; }
j() { (IFS=:; echo "$*"); }
# fl METHOD GROUP VARIANT HP_FILE SEEDS "EXTRA_ARGS"  -> job id   (multi-node FL)
fl() {
  METHOD=$1 GROUP=$2 VARIANT=$3 HP_FILE=$4 EXTRA_ARGS="$6" sbatch --parsable --array=$5 \
    --job-name=v5_${2}_${3} --nodes=$N job_fl_multinode.slurm
}
# ana DEPENDENCY "python args"  -> job id   (analysis script in fedphase)
ana() { ARGS="$2" sbatch --parsable $(_dep "$1") job_v5.slurm; }
