#!/bin/bash
# Reviewer comment 5 — EQUAL TUNING BUDGET on MIT-140 (data2).
# Grid fixed a priori, 4 configurations (= budget of each tuned baseline), seed-0 VALIDATION only:
#   kappa (look-ahead) = 1,  tau in {0.5, 1},  lambda in {0.5, 0.75},  mu = FedProx's (1e-3)
#   (0.5, 0.5) and (0.5, 0.75) already exist in tune4 -> only tau = 1 is run here (group tune5).
#
#   bash submit_equal_budget.sh          # step 1: 2 runs (seed 0) + equal_budget.py
#   bash submit_equal_budget.sh run15    # step 2 ONLY if step 1 selects a config other than (0.5, 1, 0.75):
#                                        #   selected config on seeds 1-15 + comparison with the main table
#   RERUN=1 bash submit_equal_budget.sh  # also re-run the two tau=0.5 points in tune5 (full grid in one group)
source "$(dirname "$0")/lib.sh"
EB=$R/equal_budget

if [ "$1" = run15 ]; then
  SEL=$CODE/$EB/selected_fedphase.env
  [ -f $SEL ] || { echo "run step 1 first ($SEL missing)"; exit 1; }
  source $SEL
  if [ "$SAME_AS_PAPER" = 1 ]; then
    echo "validation selected the paper configuration (tau=$TAU, kappa=$LOOKAHEAD, lambda=$SHRINK): nothing to run."
    exit 0
  fi
  ID=$(fl eqbudget fedphase_eb ${SEEDS:-1-15} "--tau-scale $TAU --lookahead $LOOKAHEAD --shrink $SHRINK")
  A=$(ana afterany:$ID "compare_variant.py --dataset $DATASET --group eqbudget --variant fedphase_eb --name FedPhase-EB --boot 5000")
  echo "eqbudget fedphase (tau=$TAU, lambda=$SHRINK) -> $ID, analysis -> $A  =>  $CODE/$R/eqbudget/"
  exit 0
fi

IDS=()
for t in 1.0; do for l in 0.5 0.75; do
  IDS+=($(fl tune5 "fedphase5__TAU=$t,SHRINK=$l" 0 "--tau-scale $t --lookahead 1.0 --shrink $l")); done; done
if [ "$RERUN" = 1 ]; then
  for l in 0.5 0.75; do IDS+=($(fl tune5 "fedphase5__TAU=0.5,SHRINK=$l" 0 "--tau-scale 0.5 --lookahead 1.0 --shrink $l")); done
fi
A=$(ana afterany:$(j "${IDS[@]}") "equal_budget.py --dataset $DATASET")
echo "tune5: ${#IDS[@]} runs (${IDS[*]}), analysis $A -> $CODE/$EB/{table_equal_budget.csv,.tex,summary.md}"
