#!/bin/bash
# STEP 2 (after step 1 finished). With the tuned lambda:
#  main    : FedPhase (shrink) x 5 seeds -> replaces main/fedphase (old look-ahead run kept as main/fedphase_la)
#  sweep4  : lambda in {0,.25,.5,.75,1} x 5 seeds   (sensitivity figure)
#  ablation4: full / no_lookahead / no_phase_kernel / no_adaptive_mu / no_shrink / no_prox x 5 seeds
# Collectors: main_final (last round, primary), main (best-val round, sensitivity), sweep4_final, ablation4_final.
cd "$(dirname "$0")"; mkdir -p logs
export DATASET=${DATASET:-data2}
CODE=${CODE_DIR:-${ROOT:-$HOME/fedphase_root}/fedphase}
T4=$CODE/results/$DATASET/tune4/best_hparams.env
[ -f $T4 ] || { echo "run submit_shrink_tune.sh first (and wait for it)"; exit 1; }
[ -n "$SHRINK_SEL" ] && echo "SHRINK=$SHRINK_SEL" > $T4          # manual override: SHRINK_SEL=0.5 bash submit_shrink_main.sh
cat $CODE/results/$DATASET/tune/best_hparams.env $CODE/results/$DATASET/tune3/best_hparams.env $T4 > $CODE/results/$DATASET/tune4/combined.env
export HP_FILE=results/$DATASET/tune4/combined.env
echo "using: $(tr '\n' ' ' < $CODE/$HP_FILE)"
for d in client_logs server_logs; do
  [ -d $CODE/$d/$DATASET/main/fedphase ] && [ ! -d $CODE/$d/$DATASET/main/fedphase_la ] && \
    mv $CODE/$d/$DATASET/main/fedphase $CODE/$d/$DATASET/main/fedphase_la
done
SEEDS=${SEEDS:-1-5}; N=$((${CLIENT_NODES:-4}+1))
sub() { # group variant extra-args
  METHOD=fedphase GROUP=$1 VARIANT=$2 EXTRA_ARGS="$3" sbatch --parsable --array=$SEEDS --job-name=$1_$2 --nodes=$N job_fl_multinode.slurm; }
M=(); S=(); A=()
M+=($(sub main fedphase ""))
for l in 0.0 0.25 0.5 0.75 1.0; do S+=($(sub sweep4 shrink_$l "--shrink $l")); done
A+=($(sub ablation4 full ""))
A+=($(sub ablation4 no_lookahead "--lookahead 0"))
A+=($(sub ablation4 no_phase_kernel "--tau-scale 1e6"))
A+=($(sub ablation4 no_adaptive_mu "--adaptive-mu 0"))
A+=($(sub ablation4 no_shrink "--shrink 1.0"))
A+=($(sub ablation4 no_prox "--mu 0"))
j() { (IFS=:; echo "$*"); }
GROUP=main PROTOCOL=final sbatch --parsable --dependency=afterany:$(j "${M[@]}") job_collect.slurm
GROUP=main PROTOCOL=bestval sbatch --parsable --dependency=afterany:$(j "${M[@]}") job_collect.slurm
GROUP=sweep4 sbatch --parsable --dependency=afterany:$(j "${S[@]}") job_collect.slurm
GROUP=ablation4 sbatch --parsable --dependency=afterany:$(j "${A[@]}") job_collect.slurm
echo "submitted: main ${M[*]} | sweep4 ${S[*]} | ablation4 ${A[*]}"
echo "outputs: results/$DATASET/{main_final,main,sweep4_final,ablation4_final}/"
