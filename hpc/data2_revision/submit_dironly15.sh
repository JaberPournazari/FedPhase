#!/bin/bash
# Direction-only FedPhase (tau -> inf, look-ahead gate kept) on 15 seeds, MIT-140 (data2).
# Same config as ablation4/no_phase_kernel (paper's 5-seed ablation): tune4/combined.env + --tau-scale 1e6.
#   bash submit_dironly15.sh            # seeds 6-15 run; seeds 1-5 copied from ablation4/no_phase_kernel
#   FRESH=1 bash submit_dironly15.sh    # run all seeds 1-15 from scratch (no copy)
source "$(dirname "$0")/lib.sh"
G=dironly15; V=fedphase_dironly
if [ "$FRESH" = 1 ]; then
  SEEDS=${SEEDS:-1-15}
else
  SEEDS=${SEEDS:-6-15}
  for d in client_logs server_logs; do
    SRC=$CODE/$d/$DATASET/ablation4/no_phase_kernel; DST=$CODE/$d/$DATASET/$G/$V
    mkdir -p $DST
    for s in 1 2 3 4 5; do
      [ -d $SRC/seed_$s ] || { echo "missing $SRC/seed_$s (use FRESH=1)"; exit 1; }
      [ -d $DST/seed_$s ] || cp -r $SRC/seed_$s $DST/
    done
  done
  echo "seeds 1-5 copied from ablation4/no_phase_kernel"
fi
ID=$(fl $G $V $SEEDS "--tau-scale 1e6")
A=$(ana afterany:$ID "compare_variant.py --dataset $DATASET --group $G --variant $V --name Direction-only --boot 5000")
echo "direction-only seeds $SEEDS -> $ID, analysis -> $A  =>  $CODE/$R/$G/"
