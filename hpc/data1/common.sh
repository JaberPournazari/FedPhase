# Sourced by every job. Defaults can be overridden from the environment.
DATASET=${DATASET:-data1}
PARTS=${PARTS:-15}; ROUNDS=${ROUNDS:-100}; EPOCHS=${EPOCHS:-3}; LR=${LR:-0.001}
SEQ=${SEQ:-5}; TRAIN_FRAC=${TRAIN_FRAC:-0.3}; VAL_FRAC=${VAL_FRAC:-0.1}
export FEAT_MODE=${FEAT_MODE:-union}                      # data1 inputs: union (aligned) | positional
ROOT=${ROOT:-$HOME/fedphase_root}
CODE_DIR=${CODE_DIR:-$ROOT/fedphase}
export DATA_DIR=${DATA_DIR:-$ROOT/$DATASET}
eval "$(conda shell.bash hook)" 2>/dev/null; conda activate "${CONDA_ENV:-fedphase}"
cd "$CODE_DIR" || { echo "no $CODE_DIR"; exit 1; }
# method hyper-parameters: defaults <- HP_FILE (colon-separated list, later files win) <- EXTRA_ARGS
MU=${MU:-0.01}; TAU=${TAU:-1.0}; DITTO_LAMBDA=${DITTO_LAMBDA:-0.1}; IFCA_K=${IFCA_K:-3}
KNN=${KNN:-0}; ADAPT=${ADAPT:-1}; PHASESET=${PHASESET:-full}; LOOKAHEAD=${LOOKAHEAD:-0}; SHRINK=${SHRINK:-1.0}
if [ -n "$HP_FILE" ]; then
  IFS=: read -ra _HPS <<< "$HP_FILE"
  for _f in "${_HPS[@]}"; do
    if [ -f "$_f" ]; then set -a; source "$_f"; set +a; echo "hp <- $_f: $(tr '\n' ' ' < "$_f")"
    else echo "ERROR: hyper-parameter file $_f missing (run submit_tune.sh first)"; exit 1; fi
  done
fi
DATA_ARGS="--seq-len $SEQ --train-frac $TRAIN_FRAC --val-frac $VAL_FRAC"
SRV_HP="--mu $MU --tau-scale $TAU --ditto-lambda $DITTO_LAMBDA --ifca-k $IFCA_K --knn $KNN \
 --adaptive-mu $ADAPT --phase-set $PHASESET --lookahead $LOOKAHEAD --shrink $SHRINK --lr $LR $EXTRA_ARGS"
echo "dataset=$DATASET parts=$PARTS rounds=$ROUNDS epochs=$EPOCHS seq=$SEQ train_frac=$TRAIN_FRAC feat_mode=$FEAT_MODE"
