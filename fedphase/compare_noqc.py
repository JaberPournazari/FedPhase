#!/usr/bin/env python
"""Sensitivity to the charge-capacity inputs, MIT-140, LAST-round test RMSE (protocol of the paper's main table).

noQC: QC, QC_rollmean, QC_rollstd, QC_slope, QC_cummax removed (27 -> 22 inputs) and qc_change removed from the
phase vector (9 -> 8). Hyper-parameters transferred from the main runs (no re-tuning).

  python compare_noqc.py --dataset data2 --group noqc            (cwd = fedphase/)

Reads  client_logs/<ds>/<group>/<variant>/seed_*/error_client*.csv                      (noQC, this code)
       <ref>/client_logs/<ds>/main/<method>/...  and  dironly15/fedphase_dironly          (with QC, the paper)
Only seeds/cells present for every noQC variant are used (paired). Statistics as compare_variant.py:
  FedPhase(noQC) vs each noQC baseline: hierarchical bootstrap CI of the gain, seed-level Wilcoxon (Holm, primary),
  cell-level Wilcoxon (secondary); FedPhase vs Direction-only reported separately. Gains WITH QC on the same seeds
  are given next to them.
-> results/<ds>/<group>/{table_noqc_main.{csv,tex}, table_noqc_stats.{csv,tex}, fig_noqc.{png,pdf}, summary_noqc.md}
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from v4common import COLORS, LABELS, ORDER, fmt_p, hier_boot_gain, holm, load_errors, plt, save

DIR = "fedphase_dironly"
VORDER = ORDER + [DIR]
LAB = {**LABELS, DIR: "Direction-only"}
COL = {**COLORS, DIR: "#555555"}
QC_REF = {DIR: [("dironly15", DIR), ("ablation4", "no_phase_kernel")]}   # with-QC source of each variant


def wilcoxon_p(x, y):
    return stats.wilcoxon(x, y).pvalue if len(x) >= 5 and np.any(x != y) else np.nan


def pivot(d, seeds, clients, col):
    return d.pivot_table(index="seed", columns="partition", values=col).loc[seeds, clients].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data2")
    ap.add_argument("--group", default="noqc")
    ap.add_argument("--ref", default=".", help="fedphase (with-QC results)")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--col", default="final_rmse", help="final_rmse = last round (primary); rmse = best-val round")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ds = a.dataset
    out = os.path.join("results", ds, a.group); os.makedirs(out, exist_ok=True)
    problems = []

    # ---- noQC runs (this code) + provenance check
    N = {}
    for v in VORDER:
        d = load_errors(os.path.join("client_logs", ds, a.group, v))
        if d.empty:
            continue
        if "feature_set" not in d or not (d.feature_set == "noqc").all() or not (d.n_features == 22).all():
            problems.append(f"{v}: error files are not from FEATURE_SET=noqc (22 inputs) -> excluded")
            continue
        if any(c == "phi_qc_change" for c in d.columns):
            problems.append(f"{v}: phi_qc_change present in client logs")
        N[v] = d
    if "fedphase" not in N:
        raise SystemExit("no valid noQC FedPhase results\n  " + "\n  ".join(problems))
    methods = [v for v in VORDER if v in N]
    seeds = sorted(set.intersection(*[set(N[v].seed) for v in methods]))
    clients = sorted(set.intersection(*[set(N[v].partition) for v in methods]))
    for v in methods:
        ns = N[v][N[v].seed.isin(seeds)].groupby("seed").partition.nunique()
        if (ns < len(clients)).any() or len(clients) < 140:
            problems.append(f"{v}: incomplete seeds {dict(ns[ns < 140])}")
    Rn = {v: pivot(N[v], seeds, clients, a.col) for v in methods}
    S = len(seeds)
    print(f"noQC methods {methods}\nseeds {seeds}  cells {len(clients)}")

    # ---- with-QC reference (paper), same seeds/cells where available
    Rq = {}
    for v in methods:
        for g, name in QC_REF.get(v, [("main", v)]):
            d = load_errors(os.path.join(a.ref, "client_logs", ds, g, name))
            if not d.empty and set(seeds) <= set(d.seed) and set(clients) <= set(d.partition):
                Rq[v] = pivot(d, seeds, clients, a.col)
                break
        else:
            problems.append(f"{v}: no with-QC reference covering seeds {seeds[0]}-{seeds[-1]}")

    # ---- main table
    ranks = stats.rankdata(np.stack([Rn[v] for v in methods]), axis=0).mean(axis=(1, 2))
    rows = []
    for i, v in enumerate(methods):
        sn = Rn[v].mean(1)
        r = {"method": LAB[v], "variant": v, "n_seeds": S, "rmse_noqc_mean": sn.mean(), "rmse_noqc_std": sn.std(ddof=1),
             "rmse_p90_noqc": np.quantile(Rn[v], 0.9, axis=1).mean(), "avg_rank_noqc": ranks[i],
             "rmse_qc_mean": np.nan, "rmse_qc_std": np.nan, "delta_%": np.nan, "qc_vs_noqc_seed_p": np.nan}
        if v in Rq:
            sq = Rq[v].mean(1)
            r.update({"rmse_qc_mean": sq.mean(), "rmse_qc_std": sq.std(ddof=1),
                      "delta_%": 100 * (sn.mean() - sq.mean()) / sq.mean(), "qc_vs_noqc_seed_p": wilcoxon_p(sn, sq)})
        rows.append(r)
    tm = pd.DataFrame(rows)
    tm.to_csv(os.path.join(out, "table_noqc_main.csv"), index=False, float_format="%.5f")

    T = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"Method & RMSE (with $Q^c$) & RMSE (without $Q^c$) & $\Delta$ (\%) & Rank \\", r"\midrule"]
    fl_best = tm[~tm.variant.isin(["centralized", "local"])].rmse_noqc_mean.min()
    for _, r in tm.iterrows():
        q = "--" if np.isnan(r.rmse_qc_mean) else f"{r.rmse_qc_mean:.4f} $\\pm$ {r.rmse_qc_std:.4f}"
        x = f"{r.rmse_noqc_mean:.4f} $\\pm$ {r.rmse_noqc_std:.4f}"
        if np.isclose(r.rmse_noqc_mean, fl_best):
            x = rf"\textbf{{{x}}}"
        dl = "--" if np.isnan(r["delta_%"]) else f"{r['delta_%']:+.1f}"
        T.append(f"{r.method} & {q} & {x} & {dl} & {r.avg_rank_noqc:.2f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_noqc_main.tex"), "w").write("\n".join(T))

    # ---- paired statistics: FedPhase(noQC) vs noQC baselines (Holm) + vs Direction-only (separate)
    def pair(m):
        A, B = Rn["fedphase"], Rn[m]
        g = hier_boot_gain(A, B, a.boot, rng)
        sa, sb, ca, cb = A.mean(1), B.mean(1), A.mean(0), B.mean(0)
        d = cb - ca; nz = d[d != 0]; rk = stats.rankdata(np.abs(nz))
        r = {"baseline": LAB[m], "variant": m, "rmse_baseline": B.mean(), "rmse_fedphase": A.mean(),
             "gain_%": 100 * (B.mean() - A.mean()) / B.mean(),
             "gain_ci_lo": np.percentile(g, 2.5), "gain_ci_hi": np.percentile(g, 97.5),
             "seed_win": f"{int((sa < sb).sum())}/{S}", "seed_wilcoxon_p": wilcoxon_p(sa, sb),
             "client_win_rate": (d > 0).mean(), "client_wilcoxon_p": stats.wilcoxon(ca, cb).pvalue,
             "rank_biserial": (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if rk.size else 0.0,
             "gain_with_qc_%": np.nan}
        if "fedphase" in Rq and m in Rq:
            r["gain_with_qc_%"] = 100 * (Rq[m].mean() - Rq["fedphase"].mean()) / Rq[m].mean()
        return r

    base = [m for m in methods if m not in ("fedphase", DIR)]
    ts = pd.DataFrame([pair(m) for m in base])
    if len(ts):
        ts["seed_p_holm"] = holm(ts.seed_wilcoxon_p.fillna(1).values)
        ts["client_p_holm"] = holm(ts.client_wilcoxon_p.values)
    if DIR in methods:
        r = pair(DIR); r["seed_p_holm"] = r["client_p_holm"] = np.nan
        ts = pd.concat([ts, pd.DataFrame([r])], ignore_index=True)
    ts.to_csv(os.path.join(out, "table_noqc_stats.csv"), index=False, float_format="%.4g")

    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"vs.\ & Gain (\%) [95\% CI] & Seeds & $p_{\text{seed}}$ & $p_{\text{cell}}$ & $r_{rb}$ & Gain with $Q^c$ (\%) \\",
         r"\midrule"]
    for _, r in ts.iterrows():
        ps = r.seed_p_holm if np.isfinite(r.seed_p_holm) else r.seed_wilcoxon_p
        pc = r.client_p_holm if np.isfinite(r.client_p_holm) else r.client_wilcoxon_p
        sep = r"\midrule" + "\n" if r.variant == DIR else ""
        gq = "--" if np.isnan(r["gain_with_qc_%"]) else f"{r['gain_with_qc_%']:+.1f}"
        L.append(f"{sep}{r.baseline} & {r['gain_%']:+.1f} [{r.gain_ci_lo:+.1f}, {r.gain_ci_hi:+.1f}] & {r.seed_win} & "
                 f"{fmt_p(ps, True)} & {fmt_p(pc, True)} & {r.rank_biserial:+.2f} & {gq} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_noqc_stats.tex"), "w").write("\n".join(L))

    # ---- figure: (a) with vs without QC per method, (b) per-seed RMSE without QC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(methods)); w = 0.38
    q = [tm.set_index("variant").rmse_qc_mean.get(v, np.nan) for v in methods]
    n_ = [tm.set_index("variant").rmse_noqc_mean[v] for v in methods]
    ax1.bar(x - w / 2, q, w, color=[COL[v] for v in methods], alpha=0.45, edgecolor="black", lw=0.8)
    ax1.bar(x + w / 2, n_, w, color=[COL[v] for v in methods], edgecolor="black", lw=0.8, hatch="//")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor="lightgrey", edgecolor="black", label="With $Q^c$"),
                        Patch(facecolor="grey", edgecolor="black", hatch="//", label="Without $Q^c$")],
               frameon=False, ncol=2, loc="upper left")
    ax1.set_ylim(0, 1.2 * np.nanmax(q + n_))
    ax1.set_xticks(x); ax1.set_xticklabels([LAB[v] for v in methods], rotation=30, ha="right")
    ax1.set_ylabel("Mean test RMSE (last round)")
    ax1.set_title("(a)", loc="left")
    xs = np.arange(S)
    for v, mk in [("fedavg", "^"), ("fedprox", "s"), (DIR, "D"), ("fedphase", "o")]:
        if v in Rn:
            ax2.plot(xs, Rn[v].mean(1), mk + "-", color=COL[v], lw=1.6, ms=6, label=LAB[v])
    ax2.set_xticks(xs); ax2.set_xticklabels(seeds)
    ax2.set_xlabel("Seed"); ax2.set_ylabel("Mean test RMSE, without $Q^c$")
    ax2.legend(frameon=False, ncol=2); ax2.set_title("(b)", loc="left")
    save(fig, out, "fig_noqc_mit140_with_vs_without_qc")

    # ---- summary
    fl = [v for v in methods if v not in ("centralized", "local")]
    t_ = tm.set_index("variant")
    spread_n = t_.loc[fl, "rmse_noqc_mean"].max() - t_.loc[fl, "rmse_noqc_mean"].min()
    spread_q = (t_.loc[fl, "rmse_qc_mean"].max() - t_.loc[fl, "rmse_qc_mean"].min()) if all(v in Rq for v in fl) else np.nan
    lines = [f"# MIT-140 without charge-capacity inputs ({S} seeds {seeds[0]}-{seeds[-1]}, {len(clients)} cells, {a.col})",
             "", tm[["method", "rmse_qc_mean", "rmse_noqc_mean", "delta_%", "rmse_p90_noqc", "avg_rank_noqc"]]
             .to_string(index=False, float_format=lambda v: f"{v:.4f}"), "",
             f"Spread of FL methods (max - min mean RMSE): with QC {spread_q:.4f}, without QC {spread_n:.4f}", ""]
    for _, r in ts.iterrows():
        ps = r.seed_p_holm if np.isfinite(r.seed_p_holm) else r.seed_wilcoxon_p
        lines.append(f"- FedPhase vs {r.baseline} (noQC): {r.rmse_fedphase:.4f} vs {r.rmse_baseline:.4f}, gain "
                     f"{r['gain_%']:+.1f}% [{r.gain_ci_lo:+.1f}, {r.gain_ci_hi:+.1f}], seeds {r.seed_win}, "
                     f"p_seed{' (Holm)' if r.variant != DIR else ''} = {fmt_p(ps)}, p_cell = {fmt_p(r.client_wilcoxon_p)}"
                     f"; gain with QC {r['gain_with_qc_%']:+.1f}%")
    best = tm[tm.variant.isin(fl)].sort_values("rmse_noqc_mean").iloc[0].method
    lines += ["", f"Best FL method without QC: {best}"]
    if problems:
        lines += ["", "## Checks that FAILED", ""] + [f"- {p}" for p in problems]
    else:
        lines += ["", "All checks passed: every noQC run has 22 inputs, no phi_qc_change, all cells and seeds."]
    open(os.path.join(out, "summary_noqc.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
