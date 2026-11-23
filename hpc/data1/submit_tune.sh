#!/bin/bash
# S1 Hyper-parameter tuning on VALIDATION (seed 0; reported runs use seeds 1-15), last-round val RMSE.
#  stage 1 (tune) : FedProx mu, Ditto lambda, IFCA K                     (4 configs each)
#  stage 2 (tune2): FedPhase tau x look-ahead, mu FIXED to FedProx's mu   (4 configs)
#  stage 3 (tune3): FedPhase shrinkage lambda                             (4 configs)
# Stages are chained with SLURM dependencies -> submit once and wait.
source "$(dirname "$0")/lib.sh"
unset HP_FILE; DEP=""; IDS=()
for v in 0.001 0.01 0.1 1.0; do IDS+=($(sub tune "fedprox__MU=$v"          fedprox 0 "--mu $v")); done
for v in 0.001 0.01 0.1 1.0; do IDS+=($(sub tune "ditto__DITTO_LAMBDA=$v"  ditto   0 "--ditto-lambda $v")); done
for v in 2 3 4 5;            do IDS+=($(sub tune "ifca__IFCA_K=$v"         ifca    0 "--ifca-k $v")); done
C1=$(collect tune final afterany:$(j "${IDS[@]}"))

export HP_FILE=$R/tune/best_hparams.env; DEP=afterok:$C1; I2=()
for cfg in "0.25 0.5" "0.25 1.0" "0.5 0.5" "0.5 1.0"; do set -- $cfg
  I2+=($(sub tune2 "fedphase__TAU=$1,LOOKAHEAD=$2" fedphase 0 "--tau-scale $1 --lookahead $2")); done
DEP=""; C2=$(collect tune2 final afterany:$(j "${I2[@]}"))

export HP_FILE=$R/tune/best_hparams.env:$R/tune2/best_hparams.env; DEP=afterok:$C2; I3=()
for l in 0.25 0.5 0.75 1.0; do I3+=($(sub tune3 "fedphase__SHRINK=$l" fedphase 0 "--shrink $l")); done
DEP=""; C3=$(collect tune3 final afterany:$(j "${I3[@]}"))
echo $C3 > logs/.tune_done
echo "tune: stage1 ${#IDS[@]} jobs -> $C1 | stage2 ${#I2[@]} -> $C2 | stage3 ${#I3[@]} -> $C3"
echo "-> $CODE/$R/{tune,tune2,tune3}/{tune_table.csv,best_hparams.env}"
