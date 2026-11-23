# FedPhase: Degradation-Phase-Guided Federated Learning for Battery State-of-Health Estimation

Code for the paper *FedPhase: Degradation-Phase-Guided Federated Learning for Battery State-of-Health Estimation Under Degradation Shift* (under review).

FedPhase aggregates client models with weights from a kernel over each cell's degradation-phase descriptor
(`θ̃_i = Σ_j a_ij θ_j`, `a_ij ∝ n_j·exp(-‖z_i−z_j‖²/2h²)`), adds a phase-adaptive proximal term and a look-ahead gate with shrinkage.
Baselines in the same pipeline: Centralized, Local, FedAvg, FedProx, FedPer, Ditto, IFCA.

## Layout
```
fedphase/   all Python: task.py (data, model, split, phase descriptor), server_app.py / client.py (Flower),
            baselines.py, run_seeds.py, collect_results.py, paper_results.py + revision analyses
hpc/        SLURM scripts, one folder per experiment group (see below)
data/       data instructions (README.md); data2 CSVs are not stored in git
```
| `hpc/` folder | what | dataset |
|---|---|---|
| `data2_main` | tuning, main comparison (8 methods), ablation, shrinkage, seeds 1–15, paper tables | MIT-140 |
| `data2_revision` | equal tuning budget for FedPhase, direction-only variant (15 seeds) | MIT-140 |
| `data2_noqc` | sensitivity run without the 5 charge-capacity inputs and φ^qc | MIT-140 |
| `data1` | all scenarios on the 15-cell dataset | WMG-15 |

## Install
```bash
conda create -n fedphase python=3.10 && conda activate fedphase
pip install -r requirements.txt
```

## Data
See [`data/README.md`](data/README.md). Expected layout: `<ROOT>/data2/Cell_<i>.csv` (140 files), optionally `<ROOT>/data1/`.
Set `DATA_DIR` to the dataset folder; the format (data1/data2) is detected automatically.

## Quick test (single machine, no SLURM)
```bash
cd fedphase
export DATA_DIR=../data/data2
python run_seeds.py --method fedphase --seeds 1 --num-partitions 8 --num-rounds 3 --local-epochs 1 \
    --port 0 --run-tag data2/smoke/fedphase -- --mu 1e-3 --tau-scale 0.5 --lookahead 1.0 --shrink 0.75
python baselines.py --mode local --seed 1 --num-partitions 8 --num-rounds 3 --run-tag data2/smoke/local
```

## Reproduce the paper (SLURM)
Cluster layout: `$ROOT/{fedphase,data1,data2}` (`ROOT` defaults to `$HOME/fedphase_root`; `fedphase` = this repo's `fedphase/` folder).
Run every script from its own `hpc/<group>/` folder. Overrides: `ROOT PARTS ROUNDS(100) EPOCHS(3) SEQ(10) TRAIN_FRAC(0.3) VAL_FRAC(0.1) SEEDS CLIENT_NODES(4) CONDA_ENV(fedphase)`.

```bash

# MIT-140 main results
cd hpc/data2_main && mkdir -p logs
sbatch job_smoke.slurm                 # sanity check, all methods, 3 rounds
bash submit_tune.sh                    # baselines, val-only tuning (seed 0)
bash submit_shrink_tune.sh             # FedPhase tuning
bash submit_main.sh; bash submit_shrink_main.sh
bash submit_ablation.sh
bash submit_seeds15.sh                 # seeds 6-15 for all methods + paper tables -> fedphase/results/data2/paper/

# revision experiments (need tune/ and tune4/ from above)
cd ../data2_revision && bash submit_equal_budget.sh && bash submit_dironly15.sh
cd ../data2_noqc && (cd ../../fedphase && python qc_leak_check.py --dataset data2) && bash submit_noqc.sh

# 15-cell dataset
cd ../data1 && sbatch job_smoke.slurm && bash submit_all.sh
```
Tables (`.csv/.tex`) and figures (`.png/.pdf`) are written to `fedphase/results/<dataset>/paper/`.

## Protocol
Per cell: temporal split 30 % train / 10 % val / 60 % test; scaler fit on train rows; sequence length 10 (data1: 5);
100 rounds × 3 local epochs; same initialisation for every method per seed; all hyper-parameters and the reported round are
chosen on validation. Reported: last-round test RMSE, mean ± std over 15 seeds, paired tests with Holm correction.
FedPhase (MIT-140): μ = 1e-3, τ = 0.5, look-ahead κ = 1, shrinkage λ = 0.75 (`results/data2/tune4/combined.env`).
Special cases: `--tau-scale 1e6 --adaptive-mu 0` = FedProx; `--mu 0 --tau-scale 1e6` = FedAvg.

## Notes
- `FEATURE_SET=noqc` (data2 only) removes `QC, QC_rollmean, QC_rollstd, QC_slope, QC_cummax` (27 → 22 inputs) and `qc_change` from the phase vector; default `full` is the paper setting.
- MIT-140 tuning in the paper used the best-validation round (`collect_results.py --protocol bestval`); data1 tuning uses the last-round validation RMSE (default).
- Runs are stochastic (PyTorch/Flower, multi-node); expect small differences from the published numbers.

## Citation
See `CITATION.cff`. Please also cite the original MIT–Stanford–Toyota dataset (Severson et al., *Nature Energy* 2019; Attia et al., *Nature* 2020) when using the data.

## License
MIT (code). Data are covered by their own licences, see `data/README.md`.
