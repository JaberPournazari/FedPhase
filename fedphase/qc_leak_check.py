#!/usr/bin/env python
"""How much of the target QD(t) is already in the input QC(t)?  MIT-140, no learning.
Same rows / normalisation (QD(t)/QD(0)) / temporal split as task.load_data.

Per cell:
  corr_all, corr_test      Pearson r(QC, QD) over the whole life / the test window
  corr_diff                r(dQC, dQD) of cycle-to-cycle differences (whole life)
  ce_median                median QD/QC (Coulombic efficiency)
  rmse_identity            test RMSE of  y_hat(t) = QC(t)/QD(0)                       (no fitting)
  rmse_linear              test RMSE of  y_hat(t) = a*QC(t)/QD(0) + b, (a,b) fit on train rows
  rmse_last_train          test RMSE of  y_hat(t) = y(last train row)                  (no-input persistence)
  python qc_leak_check.py --dataset data2  -> results/<ds>/noqc/{qc_leak_cells.csv, qc_leak_summary.md}
"""
import argparse
import os

import numpy as np
import pandas as pd

from task import TARGET_COLUMN, load_full_cell, num_cells


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else np.nan


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data2")
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args()
    out = os.path.join("results", a.dataset, "noqc"); os.makedirs(out, exist_ok=True)

    rows = []
    for pid in range(num_cells()):
        df = load_full_cell(pid)
        q0 = df[TARGET_COLUMN].iloc[0]
        y = (df[TARGET_COLUMN] / q0).to_numpy(float)
        qc = (df["QC"] / q0).to_numpy(float)
        n = len(y)
        n_tr = max(int(a.train_frac * n), a.seq_len + 2)
        n_va = max(int(a.val_frac * n), 5)
        ts = np.arange(a.seq_len - 1, n)
        tr, te = ts[ts < n_tr], ts[ts >= n_tr + n_va]
        A = np.c_[qc[tr], np.ones(len(tr))]
        coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        rows.append({"partition": pid, "n_rows": n, "n_test": len(te),
                     "corr_all": corr(qc, y), "corr_test": corr(qc[te], y[te]),
                     "corr_diff": corr(np.diff(qc), np.diff(y)),
                     "ce_median": float(np.median(df[TARGET_COLUMN] / df["QC"].replace(0, np.nan))),
                     "rmse_identity": rmse(qc[te], y[te]),
                     "rmse_linear": rmse(coef[0] * qc[te] + coef[1], y[te]),
                     "rmse_last_train": rmse(np.full(len(te), y[tr[-1]]), y[te])})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "qc_leak_cells.csv"), index=False, float_format="%.6g")

    def q(c):
        s = t[c].dropna()
        return f"median {s.median():.5f}, min {s.min():.5f}, 5th pct {s.quantile(0.05):.5f}"

    S = [f"# QC vs QD leakage check ({a.dataset}, {len(t)} cells, no learning)", "",
         f"- r(QC, QD) whole life: {q('corr_all')}",
         f"- r(QC, QD) test window: {q('corr_test')}",
         f"- r(dQC, dQD) cycle-to-cycle: {q('corr_diff')}",
         f"- cells with r_all > 0.9999: {(t.corr_all > 0.9999).sum()}/{len(t)};  > 0.999: {(t.corr_all > 0.999).sum()}/{len(t)}",
         f"- Coulombic efficiency QD/QC: {q('ce_median')}", "",
         "Test RMSE (normalised capacity, mean over cells; compare with the main table):",
         f"- y_hat = QC(t)/QD(0)            : {t.rmse_identity.mean():.4f}",
         f"- y_hat = a*QC(t)/QD(0)+b (train): {t.rmse_linear.mean():.4f}",
         f"- y_hat = y(last train cycle)    : {t.rmse_last_train.mean():.4f}  (no inputs)"]
    open(os.path.join(out, "qc_leak_summary.md"), "w").write("\n".join(S) + "\n")
    print("\n".join(S)); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
