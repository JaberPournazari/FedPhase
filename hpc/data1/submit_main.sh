#!/bin/bash
# S2 Main comparison: 8 methods x SEEDS (default 1-15), tuned hyper-parameters, last-round test.
#    + S2b transfer: FedPhase with data2's tuned hyper-parameters (no re-tuning on data1).
#   bash submit_main.sh                    # everything     |  bash submit_main.sh fedphase ditto  # subset
source "$(dirname "$0")/lib.sh"
export HP_FILE=$TUNED; IDS=()
for M in ${@:-$ALL_METHODS}; do ID=$(sub main $M $M $SEEDS ""); echo "$M -> $ID"; IDS+=($ID); done
C=$(collect main final afterany:$(j "${IDS[@]}")); CB=$(collect main bestval afterany:$(j "${IDS[@]}"))
CT=""
if [ -z "$1" ]; then
  T=$(HP_FILE="" sub transfer "fedphase__data2hp" fedphase $SEEDS "--mu 0.001 --tau-scale 0.5 --lookahead 1.0 --shrink 0.75")
  CT=$(collect transfer final afterany:$T); echo "transfer -> $T"
fi
echo "$C $CB $CT" > logs/.main_done
echo "collectors: main_final $C, main(bestval) $CB, transfer $CT -> $CODE/$R/"
