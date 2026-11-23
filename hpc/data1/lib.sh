# Helpers for the submit_*.sh scripts (sourced, run on the login node from hpc/<group>/).
cd "$(dirname "${BASH_SOURCE[0]}")"; mkdir -p logs
export DATASET=${DATASET:-data1}
CODE=${CODE_DIR:-${ROOT:-$HOME/fedphase_root}/fedphase}
R=results/$DATASET                                       # relative to CODE (jobs cd there)
TUNED=$R/tune/best_hparams.env:$R/tune2/best_hparams.env:$R/tune3/best_hparams.env
SEEDS=${SEEDS:-1-15}
ALL_METHODS="centralized local fedavg fedprox fedper ditto ifca fedphase"
_dep() { [ -n "$1" ] && echo "--dependency=$1"; }

# sub GROUP VARIANT METHOD SEEDS "EXTRA_ARGS"   (env HP_FILE / TRAIN_FRAC / FEAT_MODE / DEP are passed through)
sub() {
  local n=$(echo "${1}_${2}" | tr '=,.' '___')
  GROUP=$1 VARIANT=$2 METHOD=$3 EXTRA_ARGS="$5" sbatch --parsable --array=$4 --job-name=d1_$n \
    $(_dep "$DEP") job_run.slurm
}
# collect GROUP PROTOCOL DEPENDENCY  -> job id
collect() { GROUP=$1 PROTOCOL=$2 sbatch --parsable $(_dep "$3") job_collect.slurm; }
j() { (IFS=:; echo "$*"); }
