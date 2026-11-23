# Helpers for submit_*.sh (sourced on the login node from hpc/<group>/).
cd "$(dirname "${BASH_SOURCE[0]}")"; mkdir -p logs
export DATASET=${DATASET:-data2}
ROOT=${ROOT:-$HOME/fedphase_root}
CODE=${CODE_DIR:-$ROOT/fedphase}
R=results/$DATASET                                     # relative to CODE (jobs cd there)
HP_MAIN=$R/tune4/combined.env                          # paper config: mu=1e-3, tau=0.5, LA(kappa)=1, shrink=0.75
N=$((${CLIENT_NODES:-4}+1))
[ -f "$CODE/$HP_MAIN" ] || { echo "missing $CODE/$HP_MAIN"; exit 1; }
_dep() { [ -n "$1" ] && echo "--dependency=$1"; }
j() { (IFS=:; echo "$*"); }
# fl GROUP VARIANT SEEDS "EXTRA_ARGS"  -> job id   (FedPhase, multi-node, HP_MAIN + EXTRA_ARGS)
fl() {
  METHOD=fedphase GROUP=$1 VARIANT=$2 HP_FILE=$HP_MAIN EXTRA_ARGS="$4" sbatch --parsable --array=$3 \
    --job-name=v4_$(echo "${1}_${2}" | tr '=,.' '___') --nodes=$N job_fl_multinode.slurm
}
# ana DEPENDENCY "python args"  -> job id   (analysis script from fedphase, run inside fedphase)
ana() { ARGS="$2" sbatch --parsable $(_dep "$1") job_v4.slurm; }
