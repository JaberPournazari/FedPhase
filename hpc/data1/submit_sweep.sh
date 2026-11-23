#!/bin/bash
# S4 Sensitivity of FedPhase: shrinkage lambda (sweep_lambda) and kernel bandwidth tau (sweep_tau).
source "$(dirname "$0")/lib.sh"
export HP_FILE=$TUNED; A=(); B=()
for l in 0.0 0.25 0.5 0.75 1.0; do A+=($(sub sweep_lambda shrink_$l fedphase $SEEDS "--shrink $l")); done
for t in 0.25 0.5 1.0 2.0;      do B+=($(sub sweep_tau tau_$t fedphase $SEEDS "--tau-scale $t")); done
CA=$(collect sweep_lambda final afterany:$(j "${A[@]}")); CB=$(collect sweep_tau final afterany:$(j "${B[@]}"))
echo "$CA $CB" > logs/.sweep_done
echo "collectors $CA $CB -> $CODE/$R/{sweep_lambda_final,sweep_tau_final}/"
