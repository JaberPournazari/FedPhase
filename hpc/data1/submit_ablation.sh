#!/bin/bash
# S3 FedPhase ablation (tuned hp, SEEDS). Each variant switches off ONE component.
source "$(dirname "$0")/lib.sh"
export HP_FILE=$TUNED; IDS=()
declare -A V=(
  [full]=""
  [no_lookahead]="--lookahead 0"            # symmetric phase kernel
  [no_phase_kernel]="--tau-scale 1e6"       # no phase information (uniform kernel)
  [no_adaptive_mu]="--adaptive-mu 0"        # mu_i = mu
  [no_shrink]="--shrink 1.0"                # pure phase kernel
  [no_prox]="--mu 0"                        # no proximal term
)
for k in "${!V[@]}"; do ID=$(sub ablation $k fedphase $SEEDS "${V[$k]}"); echo "$k -> $ID"; IDS+=($ID); done
C=$(collect ablation final afterany:$(j "${IDS[@]}")); echo $C > logs/.ablation_done
echo "collector $C -> $CODE/$R/ablation_final/"
