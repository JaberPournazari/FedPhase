#!/bin/bash
# S5 Prediction horizon / early-life prediction: observed (train) fraction 0.2 / 0.4 / 0.5 of each cell's life
#    (0.3 = main). Same tuned hp (no re-tuning), all 8 methods.   FRACS="0.2 0.5" bash submit_horizon.sh
source "$(dirname "$0")/lib.sh"
export HP_FILE=$TUNED; CS=()
for tf in ${FRACS:-0.2 0.4 0.5}; do
  IDS=(); for M in ${@:-$ALL_METHODS}; do IDS+=($(TRAIN_FRAC=$tf sub horizon_tf$tf $M $M $SEEDS "")); done
  C=$(collect horizon_tf$tf final afterany:$(j "${IDS[@]}")); CS+=($C); echo "train_frac=$tf: ${#IDS[@]} jobs, collector $C"
done
echo "${CS[*]}" > logs/.horizon_done
