#!/usr/bin/env python
"""MIT-140 provenance, no learning: Cell_i -> batch, continuation pairs, cells excluded by Severson et al.
Answers the CONFIRM comment of Supplementary.tex (mapping Cell_i -> batch, exclusion criteria).

Hypothesis from the row counts (qc_leak_cells.csv): Cell_0-45 = 2017-05-12 (46), Cell_46-93 = 2017-06-30 (48),
Cell_94-139 = 2018-04-12 (46), i.e. all 140 channels of the three Severson batches, none excluded.
In Severson's processing, b1c0-4 continue as b2c7,8,9,15,16 (-> Cell_53,54,55,61,62): same physical cell
as two clients. Check: n_i + n_j ~ cycle life, and QD/IR continuity at the junction.

  python data2_provenance.py            (cwd = fedphase, DATA_DIR = .../data2; setup as qc_leak_check.py)
-> results/data2/provenance/{cells.csv, pairs.csv, summary.md}
"""
import os

import numpy as np
import pandas as pd

from task import TARGET_COLUMN, _cell_path, load_full_cell, num_cells

BATCHES = [("2017-05-12", 0, 46), ("2017-06-30", 46, 94), ("2018-04-12", 94, 140)]
CONT = {0: 53, 1: 54, 2: 55, 3: 61, 4: 62}          # b1c0-4 -> b2c7, b2c8, b2c9, b2c15, b2c16
NO_EOL = [8, 10, 12, 13, 22]                        # b1: did not reach 80 % (Severson)
NOISY = [94 + k for k in (2, 23, 32, 37, 42, 43)]   # b3: noisy channels (Severson)


def batch_of(pid):
    return next(b for b, lo, hi in BATCHES if lo <= pid < hi)


def main():
    out = os.path.join("results", "data2", "provenance"); os.makedirs(out, exist_ok=True)
    cols = list(pd.read_csv(_cell_path(0), nrows=1).columns)
    rows, raw = [], {}
    for pid in range(num_cells()):
        r = pd.read_csv(_cell_path(pid))
        d = load_full_cell(pid)
        raw[pid] = r
        cl = r["cycle_life"].dropna().unique() if "cycle_life" in r else []
        rows.append({"partition": pid, "batch": batch_of(pid), "n_raw": len(r), "n_rows": len(d),
                     "cycle_first": r["cycle"].iloc[0], "cycle_last": r["cycle"].iloc[-1],
                     "cycle_life": cl[0] if len(cl) == 1 else (np.nan if len(cl) == 0 else "multi"),
                     "QD_first": d[TARGET_COLUMN].iloc[0], "QD_last": d[TARGET_COLUMN].iloc[-1],
                     "soh_min": float((d[TARGET_COLUMN] / d[TARGET_COLUMN].iloc[0]).min()),
                     "severson_excluded": ("continuation" if pid in CONT.values() else
                                           "no_EOL" if pid in NO_EOL else "noisy" if pid in NOISY else "")})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "cells.csv"), index=False, float_format="%.6g")

    P = []
    for i, j in CONT.items():
        a, b = raw[i], raw[j]
        P.append({"cell_i": i, "cell_j": j, "n_i": len(a), "n_j": len(b), "n_sum": len(a) + len(b),
                  "cycle_life_i": t.cycle_life[i], "cycle_life_j": t.cycle_life[j],
                  "cycle_last_i": a["cycle"].iloc[-1], "cycle_first_j": b["cycle"].iloc[0],
                  "QD_last_i": a[TARGET_COLUMN].iloc[-5:].mean(), "QD_first_j": b[TARGET_COLUMN].iloc[:5].mean(),
                  "IR_last_i": a["IR"].iloc[-5:].mean(), "IR_first_j": b["IR"].iloc[:5].mean(),
                  "QD0_i": a[TARGET_COLUMN][a[TARGET_COLUMN] > 0].iloc[0]})
    p = pd.DataFrame(P)
    p["QD_jump_%"] = 100 * (p.QD_first_j - p.QD_last_i) / p.QD_last_i
    p["soh_start_j"] = p.QD_first_j / p.QD0_i            # true SOH at which client j starts
    p.to_csv(os.path.join(out, "pairs.csv"), index=False, float_format="%.5g")

    S = [f"# MIT-140 provenance ({len(t)} cells, no learning)", "", f"Columns: {cols}", "",
         t.groupby("batch").agg(cells=("partition", "size"), first=("partition", "min"), last=("partition", "max"),
                                n_rows_median=("n_rows", "median"), n_rows_max=("n_rows", "max")).to_string(), "",
         "Continuation pairs (same physical cell?): n_sum ~ cycle life, small QD/IR jump, cycle restart in j",
         p[["cell_i", "cell_j", "n_i", "n_j", "n_sum", "cycle_life_i", "cycle_life_j", "cycle_last_i",
            "cycle_first_j", "QD_jump_%", "soh_start_j"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
         f"Cells Severson et al. excluded (by their script): {(t.severson_excluded != '').sum()} "
         f"-> {dict(t[t.severson_excluded != ''].groupby('severson_excluded').partition.apply(list))}",
         f"Cells that never reach SOH 0.8 (normalised by QD after trimming): {list(t.partition[t.soh_min > 0.8])}"]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(S) + "\n")
    print("\n".join(S)); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
