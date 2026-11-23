#!/usr/bin/env python
"""Is the target spike-cleaning split-aware, and does it matter?  No training, a few seconds per dataset.

task.load_full_cell() applies _spike_clean() to the whole target series BEFORE the temporal split. The rule
replaces a cycle that deviates >5 % from both neighbours while the neighbours agree within 2 %, by linear
interpolation of its neighbours. It acts on the target only, never on the inputs. This script reports:
  1. how many labels are cleaned, per window (train / val / test);
  2. whole-series vs split-aware cleaning (each window cleaned on its own): number of labels that differ, and the
     change of the phase descriptor (computed from the train window);
  3. raw vs cleaned TEST labels: per-cell RMSE(raw - clean). By the triangle inequality,
     |RMSE(pred, raw) - RMSE(pred, clean)| <= RMSE(raw - clean) for ANY predictor, so the fleet mean of this
     quantity bounds the change of every method's fleet-mean test RMSE (and 2x it bounds any pairwise gap change).

  MIT-140 (cwd = fedphase/):
    DATA_DIR=$ROOT/data2 python label_clean_check.py
  WMG-15 (V3 task.py handles data1):
    DATA_DIR=$ROOT/data1 python label_clean_check.py \
        --task-dir . --seq-len 5 --tag data1
-> results/<tag>/provenance/{label_clean_cells.csv, label_clean_summary.md}
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", default=".")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--tag", default="data2")
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.task_dir))
    import task as T

    clean = T._spike_clean
    out = os.path.join("results", a.tag, "provenance"); os.makedirs(out, exist_ok=True)
    rows = []
    for pid in range(T.num_cells()):
        T._spike_clean = lambda s, *x, **k: (s.copy(), np.zeros(len(s), dtype=bool))
        raw_df = T.load_full_cell(pid)                       # identical rows, target not cleaned
        T._spike_clean = clean
        cl_df = T.load_full_cell(pid)                        # what every run used
        raw = raw_df[T.TARGET_COLUMN].astype(float).reset_index(drop=True)
        cl = cl_df[T.TARGET_COLUMN].astype(float).reset_index(drop=True)
        n = len(cl)
        n_tr = max(int(a.train_frac * n), a.seq_len + 2)
        n_va = max(int(a.val_frac * n), 5)
        win = [(0, n_tr), (n_tr, n_tr + n_va), (n_tr + n_va, n)]
        sa = pd.concat([clean(raw.iloc[s:e])[0] for s, e in win if e > s]).reset_index(drop=True)
        q0 = cl.iloc[0]                                      # normalisation used by load_data (row 0 never cleaned)
        y_raw, y_cl, y_sa = raw / q0, cl / q0, sa / q0
        changed = np.abs(y_raw - y_cl).to_numpy() > 1e-12
        differ = np.abs(y_sa - y_cl).to_numpy() > 1e-12
        idx = np.arange(n)
        w = np.where(idx < n_tr, 0, np.where(idx < n_tr + n_va, 1, 2))
        te = slice(n_tr + n_va, n)
        sa_df = cl_df.copy(); sa_df[T.TARGET_COLUMN] = sa.to_numpy()
        p_cl, p_sa = T.phase_descriptor(cl_df, n_tr), T.phase_descriptor(sa_df, n_tr)
        dphi = max(abs(p_cl[k] - p_sa[k]) for k in p_cl)
        rows.append({"cell": pid, "n": n, "n_train": n_tr, "n_val": n_va,
                     "cleaned_train": int(changed[w == 0].sum()), "cleaned_val": int(changed[w == 1].sum()),
                     "cleaned_test": int(changed[w == 2].sum()),
                     "differ_split_aware": int(differ.sum()), "max_dphi": dphi,
                     "test_rmse_raw_vs_clean": float(np.sqrt(np.mean((y_raw[te] - y_cl[te]) ** 2))),
                     "test_max_abs_raw_vs_clean": float(np.max(np.abs(y_raw[te] - y_cl[te]))),
                     "test_rmse_sa_vs_clean": float(np.sqrt(np.mean((y_sa[te] - y_cl[te]) ** 2)))})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "label_clean_cells.csv"), index=False, float_format="%.6g")
    tot = d[["cleaned_train", "cleaned_val", "cleaned_test"]].sum()
    b = d.test_rmse_raw_vs_clean
    S = [f"# Target spike-cleaning check ({a.tag}, {len(d)} cells, no learning)", "",
         f"- labels cleaned: {int(tot.sum())} of {int(d.n.sum())} ({100 * tot.sum() / d.n.sum():.3f} %) in "
         f"{int(((d.cleaned_train + d.cleaned_val + d.cleaned_test) > 0).sum())} cells; "
         f"train {int(tot.cleaned_train)}, val {int(tot.cleaned_val)}, test {int(tot.cleaned_test)}",
         f"- whole-series vs split-aware cleaning: {int(d.differ_split_aware.sum())} labels differ in "
         f"{int((d.differ_split_aware > 0).sum())} cells; max |d phi| = {d.max_dphi.max():.3g}; "
         f"max test RMSE(split-aware - whole) = {d.test_rmse_sa_vs_clean.max():.3g}",
         f"- raw vs cleaned test labels: per-cell RMSE mean {b.mean():.2e}, median {b.median():.2e}, "
         f"max {b.max():.2e} (cell {int(d.cell[b.idxmax()])}); max |diff| {d.test_max_abs_raw_vs_clean.max():.3g}",
         f"- => every method's fleet-mean test RMSE changes by at most {b.mean():.2e} if raw labels are used; "
         f"any pairwise gap by at most {2 * b.mean():.2e}",
         "  (smallest gaps in the paper: MIT-140 FedPhase-FedProx 0.0013 full / 0.0024 field)"]
    open(os.path.join(out, "label_clean_summary.md"), "w").write("\n".join(S) + "\n")
    print("\n".join(S)); print(f"-> {out}")


if __name__ == "__main__":
    main()
