#!/usr/bin/env python
"""MIT-140: are the conclusions robust to the cells Severson et al. merged/excluded?  No training: existing logs.

data2_provenance.py showed Cell_53,54,55,61,62 are the continuations of Cell_0-4 (same physical cells; QD jump <1.2 %,
n_i + n_j = cycle life). Evaluation-only subsets of the SAME runs (training federation unchanged):
  all        140 cells (paper)
  no_pairs   130 cells: without the 5 continuation pairs (0-4, 53, 54, 55, 61, 62)
  severson   124 cells: without the 16 cells Severson et al. excluded (continuations, no-EOL b1, noisy b3)
  pairs_i / pairs_j  Cell_0-4 / their continuations (descriptive only, n=5)
Full input set : <ref>/client_logs/data2/main/<method>  (+ dironly15/fedphase_dironly)
Field input set: client_logs/data2/noqc/<method>         (this folder)
Last-round test RMSE, seeds paired. FedPhase vs each baseline: gain, seed wins, seed-level Wilcoxon (Holm over the
baselines of that input set; Direction-only separate), cell-level Wilcoxon.

  python subset_eval.py                (cwd = fedphase/)
-> results/data2/provenance/{subset_rmse.csv, subset_stats.csv, table_subset.tex, summary_subset.md}
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from v4common import LABELS, ORDER, fmt_p, holm, load_errors

DIR = "fedphase_dironly"
LAB = {**LABELS, DIR: "Direction-only"}
PAIRS_I, PAIRS_J = [0, 1, 2, 3, 4], [53, 54, 55, 61, 62]
NO_EOL, NOISY = [8, 10, 12, 13, 22], [96, 117, 126, 131, 136, 137]
PAPER_ALL = {"full": {"fedphase": 0.0506, "fedprox": 0.0519}, "field": {"fedphase": 0.0481, "fedprox": 0.0505}}


def wil(x, y):
    return stats.wilcoxon(x, y).pvalue if len(x) >= 5 and np.any(x != y) else np.nan


def load_set(name, ref, col):
    if name == "full":
        src = {m: os.path.join(ref, "client_logs", "data2", "main", m) for m in ORDER}
        src[DIR] = os.path.join(ref, "client_logs", "data2", "dironly15", DIR)
    else:
        src = {m: os.path.join("client_logs", "data2", "noqc", m) for m in ORDER + [DIR]}
    D = {m: d for m, p in src.items() if not (d := load_errors(p)).empty}
    seeds = sorted(set.intersection(*[set(d.seed) for d in D.values()]))
    cells = sorted(set.intersection(*[set(d.partition) for d in D.values()]))
    R = {m: d.pivot_table(index="seed", columns="partition", values=col).loc[seeds, cells] for m, d in D.items()}
    return R, seeds, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=".")
    ap.add_argument("--col", default="final_rmse")
    a = ap.parse_args()
    out = os.path.join("results", "data2", "provenance"); os.makedirs(out, exist_ok=True)
    rows, srows, lines = [], [], []

    for iset in ("full", "field"):
        R, seeds, cells = load_set(iset, a.ref, a.col)
        subsets = {"all": cells,
                   "no_pairs": [c for c in cells if c not in PAIRS_I + PAIRS_J],
                   "severson": [c for c in cells if c not in PAIRS_J + NO_EOL + NOISY],
                   "pairs_i": [c for c in cells if c in PAIRS_I],
                   "pairs_j": [c for c in cells if c in PAIRS_J]}
        methods = [m for m in ORDER + [DIR] if m in R]
        lines.append(f"## {iset} input set: {len(seeds)} seeds, methods {methods}")
        for m, v in PAPER_ALL[iset].items():
            if m in R:
                x = R[m].values.mean()
                lines.append(f"check all-140 {m}: {x:.4f} (paper {v:.4f})" + ("" if abs(x - v) < 5e-4 else "  <-- MISMATCH"))
        for sname, cs in subsets.items():
            for m in methods:
                s = R[m][cs].values.mean(1)
                rows.append({"input_set": iset, "subset": sname, "n_cells": len(cs), "method": LAB[m],
                             "rmse_mean": s.mean(), "rmse_std": s.std(ddof=1),
                             "rmse_p90": np.quantile(R[m][cs].values, 0.9, axis=1).mean()})
            if sname.startswith("pairs"):
                continue
            A = R["fedphase"][cs].values
            base = [m for m in methods if m not in ("fedphase", DIR)]
            st = []
            for m in base + ([DIR] if DIR in methods else []):
                B = R[m][cs].values
                sa, sb, ca, cb = A.mean(1), B.mean(1), A.mean(0), B.mean(0)
                st.append({"input_set": iset, "subset": sname, "n_cells": len(cs), "baseline": LAB[m], "variant": m,
                           "gain_%": 100 * (B.mean() - A.mean()) / B.mean(), "seed_win": f"{int((sa < sb).sum())}/{len(sa)}",
                           "seed_p": wil(sa, sb), "cell_win": float((cb > ca).mean()), "cell_p": wil(ca, cb)})
            st = pd.DataFrame(st)
            fam = st.variant != DIR
            st["seed_p_holm"] = np.nan
            st.loc[fam, "seed_p_holm"] = holm(st.loc[fam, "seed_p"].fillna(1).values)
            srows.append(st)
            for _, r in st[st.variant.isin(["fedprox", "fedavg", "ditto", DIR])].iterrows():
                ps = r.seed_p_holm if np.isfinite(r.seed_p_holm) else r.seed_p
                lines.append(f"- {sname:9s} ({r.n_cells:3d} cells) vs {r.baseline:14s}: gain {r['gain_%']:+5.1f}%, "
                             f"seeds {r.seed_win}, p_seed{'(Holm)' if r.variant != DIR else ''} {fmt_p(ps)}, "
                             f"cells {100 * r.cell_win:.0f}%, p_cell {fmt_p(r.cell_p)}")
        for sname in ("pairs_i", "pairs_j"):
            cs = subsets[sname]
            if cs:
                f, p = R["fedphase"][cs].values.mean(), R["fedprox"][cs].values.mean()
                lines.append(f"- {sname} {cs}: FedPhase {f:.4f} vs FedProx {p:.4f} ({100 * (p - f) / p:+.1f}%, descriptive)")
        lines.append("")

    t, s = pd.DataFrame(rows), pd.concat(srows, ignore_index=True)
    t.to_csv(os.path.join(out, "subset_rmse.csv"), index=False, float_format="%.5f")
    s.to_csv(os.path.join(out, "subset_stats.csv"), index=False, float_format="%.4g")

    T = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"Inputs & FedPhase vs. & All (140) & Without pairs (130) & Severson subset (124) \\", r"\midrule"]
    for iset in ("full", "field"):
        for m in ("fedprox", "fedavg", "ditto", DIR):
            q = s[(s.input_set == iset) & (s.variant == m)].set_index("subset")
            if q.empty:
                continue
            cell = []
            for sn in ("all", "no_pairs", "severson"):
                r = q.loc[sn]; ps = r.seed_p_holm if np.isfinite(r.seed_p_holm) else r.seed_p
                cell.append(f"{r['gain_%']:+.1f}\\% ({r.seed_win}, {fmt_p(ps, True)})")
            T.append(f"{iset} & {LAB[m]} & " + " & ".join(cell) + r" \\")
        T.append(r"\midrule" if iset == "full" else r"\bottomrule")
    T.append(r"\end{tabular}")
    open(os.path.join(out, "table_subset.tex"), "w").write("\n".join(T))
    S = ["# MIT-140: robustness to merged/excluded cells (evaluation subsets, same runs)", ""] + lines
    open(os.path.join(out, "summary_subset.md"), "w").write("\n".join(S) + "\n")
    print("\n".join(S)); print(f"-> {out}")


if __name__ == "__main__":
    main()
