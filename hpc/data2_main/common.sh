# Sourced by every job. Defaults can be overridden from the environment.
DATASET=${DATASET:-data2}
PARTS=${PARTS:-140}; ROUNDS=${ROUNDS:-100}; EPOCHS=${EPOCHS:-3}; LR=${LR:-0.001}
SEQ=${SEQ:-10}; TRAIN_FRAC=${TRAIN_FRAC:-0.3}; VAL_FRAC=${VAL_FRAC:-0.1}
ROOT=${ROOT:-$HOME/fedphase_root}
CODE_DIR=${CODE_DIR:-$ROOT/fedphase}
export DATA_DIR=${DATA_DIR:-$ROOT/$DATASET}
eval "$(conda shell.bash hook)" 2>/dev/null; conda activate "${CONDA_ENV:-fedphase}"
cd "$CODE_DIR" || { echo "no $CODE_DIR"; exit 1; }
# method hyper-parameters: defaults <- tuned file (HP_FILE) <- EXTRA_ARGS (last wins in argparse)
MU=${MU:-0.01}; TAU=${TAU:-1.0}; DITTO_LAMBDA=${DITTO_LAMBDA:-0.1}; IFCA_K=${IFCA_K:-3}
KNN=${KNN:-0}; ADAPT=${ADAPT:-1}; PHASESET=${PHASESET:-full}; LOOKAHEAD=${LOOKAHEAD:-0}; SHRINK=${SHRINK:-1.0}
if [ -n "$HP_FILE" ] && [ -f "$HP_FILE" ]; then set -a; source "$HP_FILE"; set +a; echo "tuned: $(tr '\n' ' ' < $HP_FILE)"; fi
DATA_ARGS="--seq-len $SEQ --train-frac $TRAIN_FRAC --val-frac $VAL_FRAC"
SRV_HP="--mu $MU --tau-scale $TAU --ditto-lambda $DITTO_LAMBDA --ifca-k $IFCA_K --knn $KNN \
 --adaptive-mu $ADAPT --phase-set $PHASESET --lookahead $LOOKAHEAD --shrink $SHRINK --lr $LR $EXTRA_ARGS"
