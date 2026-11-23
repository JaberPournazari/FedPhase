#!/bin/bash
# S6 Input alignment: positional inputs (original DFL-RUL: slot k = different statistic per cell)
#    vs the aligned union space used everywhere else. Same tuned hp, all 8 methods.
source "$(dirname "$0")/lib.sh"
export HP_FILE=$TUNED FEAT_MODE=positional; IDS=()
for M in ${@:-$ALL_METHODS}; do IDS+=($(sub featmode_positional $M $M $SEEDS "")); done
C=$(collect featmode_positional final afterany:$(j "${IDS[@]}")); echo $C > logs/.featmode_done
echo "${#IDS[@]} jobs, collector $C -> $CODE/$R/featmode_positional_final/"
