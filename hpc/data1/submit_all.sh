#!/bin/bash
# Whole data1 study in one go (after the smoke test passed):
#   tune -> {main+transfer, ablation, sweep, horizon, featmode} -> paper
#   bash submit_all.sh                     # SEEDS=1-5 bash submit_all.sh for a quick pass
source "$(dirname "$0")/lib.sh"
bash submit_tune.sh || exit 1
export DEP=afterok:$(cat logs/.tune_done)
bash submit_main.sh; bash submit_ablation.sh; bash submit_sweep.sh; bash submit_horizon.sh; bash submit_featmode.sh
ALL=$(cat logs/.main_done logs/.ablation_done logs/.sweep_done logs/.horizon_done logs/.featmode_done | tr -s ' \n' ':' | sed 's/:$//')
DEP=afterany:$ALL bash submit_paper.sh
echo "submitted. watch: squeue -u \$USER | results: $CODE/$R/paper/"
