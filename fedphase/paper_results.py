#!/usr/bin/env python
"""Paper-ready results for FedPhase (run after collect_results.py --group main/sweep4/ablation4).

  python paper_results.py --dataset data1 --boot 5000      # -> results/data1/paper/

Primary protocol = LAST round (results/<ds>/main_final); best-val round is a robustness check.
Only seeds present for EVERY method are used (fair, paired comparison).

Tables (csv + tex):
  T1 table_main        accuracy, worst-10% client, fairness, average rank, seeds won
  T2 table_stats       paired gain vs each baseline: hierarchical-bootstrap 95% CI, seed-level and
                       client-level Wilcoxon (Holm), win rates, rank-biserial
  T3 table_subgroup    RMSE by degradation-rate tertile and by cell-length tertile
  T4 table_protocol    last-round vs best-val-round (robustness of the ranking)
  T5 table_ablation, table_sweep    (from ablation4_final / sweep4_final)
  T6 table_cost        wall time, aggregation time, communication
  V3 (optional, if the groups exist):
  T7 table_horizon     RMSE vs training fraction (early-life prediction): horizon_tf*_final + main_final
  T8 table_featmode    union-aligned inputs (main) vs positional inputs (featmode_positional_final)
  T9 table_transfer    FedPhase tuned on this dataset vs FedPhase with data2's hyper-parameters
  table_sweep_tau      sensitivity to the kernel bandwidth (sweep_tau_final)
Figures (png 300 dpi + pdf): fig_seeds, fig_gain_ci, fig_subgroup, fig_rank, fig_client_scatter,
  fig_ablation_paper, fig_sweep_paper.   summary.md = sentence-ready numbers.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("TkAgg" if (sys.platform.startswith("win") or os.environ.get("DISPLAY"))
               and os.environ.get("MPLBACKEND", "").lower() != "agg" else "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
    "axes.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

ORDER = ["centralized", "local", "fedavg", "fedprox", "fedper", "ditto", "ifca", "fedphase"]
LABELS = {"centralized": "Centralized", "local": "Local", "fedavg": "FedAvg", "fedprox": "FedProx",
          "fedper": "FedPer", "ditto": "Ditto", "ifca": "IFCA", "fedphase": "FedPhase (ours)"}
COLORS = {"centralized": "#7f7f7f", "local": "#bcbd22", "fedavg": "#1f77b4", "fedprox": "#ff7f0e",
          "fedper": "#9467bd", "ditto": "#2ca02c", "ifca": "#8c564b", "fedphase": "#d62728"}
FL = ["fedavg", "fedprox", "fedper", "ditto", "ifca", "fedphase"]
P = "fedphase"
ABL_NAMES = {"full": "FedPhase (full)", "no_lookahead": "w/o look-ahead", "no_phase_kernel": "w/o phase kernel",
             "no_adaptive_mu": r"w/o adaptive $\mu_i$", "no_shrink": r"w/o shrinkage ($\lambda$=1)",
             "no_prox": r"w/o proximal term ($\mu$=0)"}


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); k = len(p)
    h = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * p[o]))
    r = np.empty(k); r[o] = h
    return r


def fmt_p(p, tex=False):
    return (r"$<$0.001" if tex else "<0.001") if p < 1e-3 else f"{p:.3f}"


def cube(df, methods, seeds, col="rmse"):
    """R[m] : (n_seeds, n_clients) array, clients aligned across methods."""
    clients = sorted(set.intersection(*[set(df[df.variant == m].partition) for m in methods]))
    R = {}
    for m in methods:
        t = df[df.variant == m].pivot_table(index="seed", columns="partition", values=col)
        R[m] = t.loc[seeds, clients].values
    return R, clients


# ------------------------------------------------------------------ tables
def t_main(R, methods, seeds):
    allm = np.stack([R[m] for m in methods])                       # (M, S, C)
    ranks = stats.rankdata(allm, axis=0).mean(axis=(1, 2))         # rank per (seed, client)
    rows = []
    for i, m in enumerate(methods):
        s = R[m].mean(1)                                            # per-seed macro mean
        p90 = np.quantile(R[m], 0.9, axis=1)
        cstd = R[m].std(1)
        r = {"method": LABELS[m], "variant": m, "n_seeds": len(seeds),
             "rmse_mean": s.mean(), "rmse_std": s.std(ddof=1),
             "rmse_ci95": stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s)),
             "rmse_p90_mean": p90.mean(), "rmse_p90_std": p90.std(ddof=1),
             "client_std_mean": cstd.mean(), "avg_rank": ranks[i]}
        if m != P:
            r["seeds_fedphase_better"] = f"{int((R[P].mean(1) < s).sum())}/{len(s)}"
        rows.append(r)
    return pd.DataFrame(rows)


def t_stats(R, methods, n_boot, rng):
    rows = []
    S, C = R[P].shape
    for m in [x for x in methods if x != P]:
        a, b = R[P], R[m]
        # hierarchical bootstrap: seeds, then clients (paired: same indices for both methods)
        g = np.empty(n_boot)
        for k in range(n_boot):
            si = rng.integers(0, S, S); ci = rng.integers(0, C, C)
            A, B = a[np.ix_(si, ci)].mean(), b[np.ix_(si, ci)].mean()
            g[k] = 100 * (B - A) / B
        sa, sb = a.mean(1), b.mean(1)
        ca, cb = a.mean(0), b.mean(0)                                 # seed-averaged per client
        d = cb - ca
        nz = d[d != 0]
        rk = stats.rankdata(np.abs(nz))
        rows.append({
            "baseline": LABELS[m], "variant": m,
            "rmse_baseline": b.mean(), "rmse_fedphase": a.mean(),
            "gain_%": 100 * (b.mean() - a.mean()) / b.mean(),
            "gain_ci_lo": np.percentile(g, 2.5), "gain_ci_hi": np.percentile(g, 97.5),
            "p_boot_gain_le0": (g <= 0).mean(),
            "seed_win": f"{int((sa < sb).sum())}/{S}",
            "seed_wilcoxon_p": stats.wilcoxon(sa, sb).pvalue if S >= 5 and np.any(sa != sb) else np.nan,
            "client_win_rate": (d > 0).mean(),
            "client_wilcoxon_p": stats.wilcoxon(ca, cb).pvalue,
            "rank_biserial": (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if rk.size else 0.0,
            "pairwise_win_rate": (a < b).mean(),                      # over all (seed, client) pairs
        })
    t = pd.DataFrame(rows)
    t["seed_p_holm"] = holm(t.seed_wilcoxon_p.fillna(1).values)
    t["client_p_holm"] = holm(t.client_wilcoxon_p.values)
    return t


def t_subgroup(R, clients, phi, methods, key, names):
    v = phi.reindex(clients)[key].values
    q = np.nanquantile(v, [1 / 3, 2 / 3])
    grp = np.digitize(v, q)                                          # 0,1,2
    rows = []
    for gi, gname in enumerate(names):
        idx = grp == gi
        r = {"group": gname, "n_cells": int(idx.sum()),
             "range": f"[{np.nanmin(v[idx]):.3g}, {np.nanmax(v[idx]):.3g}]"}
        for m in methods:
            r[LABELS[m]] = R[m][:, idx].mean()
        best_other = min((R[m][:, idx].mean(), m) for m in FL if m != P and m in methods)
        r["best_baseline"] = LABELS[best_other[1]]
        r["gain_vs_best_%"] = 100 * (best_other[0] - R[P][:, idx].mean()) / best_other[0]
        if "fedprox" in methods:
            r["gain_vs_FedProx_%"] = 100 * (R["fedprox"][:, idx].mean() - R[P][:, idx].mean()) / R["fedprox"][:, idx].mean()
        rows.append(r)
    return pd.DataFrame(rows), grp


# ------------------------------------------------------------------ latex
def latex_main(t, path, n_cells=140):
    fl = t[t.variant.isin(FL)]
    best = {c: fl[c].min() for c in ["rmse_mean", "rmse_p90_mean", "client_std_mean", "avg_rank"]}
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Test capacity-prediction error (last round, mean $\pm$ std over seeds; per-client RMSE "
         f"macro-averaged over {n_cells} cells). " r"RMSE$_{p90}$: worst-10\% cell; $\sigma_{\text{cell}}$: spread across "
         r"cells (fairness); rank: average rank over all (seed, cell) pairs; wins: seeds in which FedPhase is better.}",
         r"\label{tab:main}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Method & RMSE $\downarrow$ & RMSE$_{p90}$ $\downarrow$ & $\sigma_{\text{cell}}$ $\downarrow$ & "
         r"Avg. rank $\downarrow$ & FedPhase wins \\", r"\midrule"]
    for _, r in t.iterrows():
        def c(col, txt):
            return rf"\textbf{{{txt}}}" if r.variant in FL and np.isclose(r[col], best[col]) else txt
        name = r.method + (r"$^\dagger$" if r.variant == "centralized" else "")
        pm = r" $\pm$ "
        wins = r.get("seeds_fedphase_better")
        cells = [c("rmse_mean", f"{r.rmse_mean:.4f}{pm}{r.rmse_std:.4f}"), c("rmse_p90_mean", f"{r.rmse_p90_mean:.4f}"),
                 c("client_std_mean", f"{r.client_std_mean:.4f}"), c("avg_rank", f"{r.avg_rank:.2f}"),
                 wins if isinstance(wins, str) else "--"]
        L.append(name + " & " + " & ".join(cells) + r" \\")
        if r.variant == "local":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\\[2pt]\footnotesize $^\dagger$Non-private reference (pooled data).",
          r"\end{table}"]
    open(path, "w").write("\n".join(L))


def latex_stats(t, path):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired comparison of FedPhase against each baseline. Gain = relative RMSE reduction with a "
         r"hierarchical (seed $\times$ cell) bootstrap 95\% CI. Seed level: Wilcoxon over seeds; cell level: "
         r"Wilcoxon over seed-averaged cells; both Holm-corrected. $r_{rb}$: rank-biserial effect size.}",
         r"\label{tab:stats}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Baseline & Gain (\%) [95\% CI] & Seed wins & $p_{\text{seed}}$ & Cell win rate & $p_{\text{cell}}$ & $r_{rb}$ \\",
         r"\midrule"]
    for _, r in t.iterrows():
        L.append(f"{r.baseline} & {r['gain_%']:+.1f} [{r.gain_ci_lo:+.1f}, {r.gain_ci_hi:+.1f}] & {r.seed_win} & "
                 f"{fmt_p(r.seed_p_holm, True)} & {100 * r.client_win_rate:.1f}\\% & {fmt_p(r.client_p_holm, True)} & "
                 f"{r.rank_biserial:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    open(path, "w").write("\n".join(L))


def latex_simple(df, path, caption, label, fmt="%.4f"):
    s = df.to_latex(index=False, float_format=lambda x: fmt % x, escape=False)
    open(path, "w").write("\n".join([r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}",
                                     rf"\label{{{label}}}", r"\resizebox{\linewidth}{!}{%", s.strip() + "}",
                                     r"\end{table}"]))


# ------------------------------------------------------------------ figures
def fig_seeds(R, methods, seeds, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    S = np.array([R[m].mean(1) for m in methods])                   # (M, S)
    x = np.arange(len(methods))
    jit = np.linspace(-0.12, 0.12, S.shape[1])
    for j in range(S.shape[1]):
        ax.plot(x + jit[j], S[:, j], color="#cccccc", linewidth=0.8, zorder=1)
    for i, m in enumerate(methods):
        ax.scatter(x[i] + jit, S[i], s=28 if m != P else 46, color=COLORS[m], edgecolor="k", linewidth=0.4, zorder=3)
        ax.hlines(S[i].mean(), x[i] - 0.28, x[i] + 0.28, color="k", linewidth=2.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m].replace(" (ours)", "\n(ours)") for m in methods])
    ax.set_ylabel("Mean test RMSE per seed")
    save(fig, out, "fig_seeds")


def fig_gain(ts, out):
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(ts) + 1.6))
    y = np.arange(len(ts))[::-1]
    for yi, (_, r) in zip(y, ts.iterrows()):
        c = COLORS[r.variant]
        ax.errorbar(r["gain_%"], yi, xerr=[[r["gain_%"] - r.gain_ci_lo], [r.gain_ci_hi - r["gain_%"]]],
                    fmt="o", color=c, markersize=8, capsize=5, linewidth=2)
    ax.axvline(0, color="k", linewidth=1.2, linestyle="--")
    ax.set_yticks(y); ax.set_yticklabels(ts.baseline)
    ax.set_xlabel("RMSE reduction by FedPhase (%) with 95% CI")
    save(fig, out, "fig_gain_ci")


def fig_subgroup(sg, methods, title, name, out):
    show = [m for m in methods if m in FL]
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.8 / len(show); x = np.arange(len(sg))
    for i, m in enumerate(show):
        ax.bar(x + (i - (len(show) - 1) / 2) * w, sg[LABELS[m]], w, color=COLORS[m], label=LABELS[m],
               edgecolor="k" if m == P else "none", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(sg.group, sg.n_cells)])
    ax.set_xlabel(title); ax.set_ylabel("Mean test RMSE")
    ax.legend(frameon=False, ncol=3, fontsize=11)
    save(fig, out, name)


def fig_rank(tm, out):
    t = tm[tm.variant.isin(FL)].sort_values("avg_rank")
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(range(len(t))[::-1], t.avg_rank, color=[COLORS[v] for v in t.variant])
    ax.set_yticks(range(len(t))[::-1]); ax.set_yticklabels(t.method)
    ax.set_xlabel("Average rank over (seed, cell) pairs (lower is better)")
    save(fig, out, "fig_rank")


def fig_scatter(R, others, out):
    others = [m for m in others if m in R]
    fig, axes = plt.subplots(1, len(others), figsize=(5 * len(others), 4.8))
    axes = np.atleast_1d(axes)
    a = R[P].mean(0)
    for ax, m in zip(axes, others):
        b = R[m].mean(0)
        lim = [min(a.min(), b.min()) * 0.9, max(a.max(), b.max()) * 1.1]
        ax.scatter(b, a, s=20, color=np.where(a < b, COLORS[P], "#999999"), alpha=0.8, edgecolor="none")
        ax.plot(lim, lim, "k--", linewidth=1.2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"{LABELS[m]} RMSE"); ax.set_title(f"{100 * (a < b).mean():.0f}% cells below diagonal")
    axes[0].set_ylabel("FedPhase RMSE")
    save(fig, out, "fig_client_scatter")


# ------------------------------------------------------------------ V3 extras
def first(base, dirs, fname):
    for d in dirs:
        p = os.path.join(base, d, fname)
        if os.path.exists(p):
            return p
    return None


def seed_mean(df, m, col="rmse"):
    """per-seed macro mean over cells -> (mean, std, n_seeds)"""
    s = df[df.variant == m].groupby("seed")[col].mean()
    return s.mean(), s.std(ddof=1) if len(s) > 1 else np.nan, len(s)


def extra_tables(base, out, df_main, methods, seeds, R):
    # T7 horizon: RMSE vs training fraction (0.3 = main)
    import glob as _g
    hz = {0.3: df_main}
    for d in sorted(_g.glob(os.path.join(base, "horizon_tf*_final"))):
        f = os.path.join(d, "long_results.csv")
        if os.path.exists(f):
            hz[float(os.path.basename(d)[len("horizon_tf"):-len("_final")])] = pd.read_csv(f)
    if len(hz) > 1:
        rows = []
        for tf in sorted(hz):
            d = hz[tf]
            r = {"train_frac": tf}
            for m in methods:
                if m in d.variant.unique():
                    mu, sd, n = seed_mean(d, m)
                    r[LABELS[m]], r[LABELS[m] + "_std"], r["n_seeds"] = mu, sd, n
            fl = {m: r[LABELS[m]] for m in FL if LABELS[m] in r and m != P}
            if fl and LABELS[P] in r:
                bm = min(fl, key=fl.get)
                r["best_baseline"], r["gain_vs_best_%"] = LABELS[bm], 100 * (fl[bm] - r[LABELS[P]]) / fl[bm]
            rows.append(r)
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(out, "table_horizon.csv"), index=False, float_format="%.5f")
        cols = ["train_frac"] + [LABELS[m] for m in methods if LABELS[m] in t] + \
               [c for c in ["gain_vs_best_%"] if c in t]
        latex_simple(t[cols].rename(columns={"train_frac": "Train fraction", "gain_vs_best_%": r"Gain vs best (\%)"}),
                     os.path.join(out, "table_horizon.tex"),
                     "Test RMSE vs.\\ length of the observed (training) part of each cell's life; the remaining "
                     "life after a 10\\% validation window is predicted.", "tab:horizon")
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for m in methods:
            if LABELS[m] in t:
                ax.errorbar(t.train_frac, t[LABELS[m]], yerr=t[LABELS[m] + "_std"], color=COLORS[m], marker="o",
                            linewidth=2.6 if m == P else 1.5, capsize=3, label=LABELS[m])
        ax.set_xlabel("Training fraction of each cell's life"); ax.set_ylabel("Mean test RMSE")
        ax.legend(frameon=False, ncol=2, fontsize=11)
        save(fig, out, "fig_horizon")
        print(t[[c for c in t.columns if not c.endswith("_std")]].to_string(index=False))

    # T8 feature alignment: union (main) vs positional
    f = os.path.join(base, "featmode_positional_final", "long_results.csv")
    if os.path.exists(f):
        dp = pd.read_csv(f)
        rows = []
        for m in methods:
            if m in dp.variant.unique():
                a_, sa, _ = seed_mean(df_main[df_main.seed.isin(seeds)], m)
                b_, sb, n = seed_mean(dp, m)
                rows.append({"method": LABELS[m], "union_aligned": a_, "union_std": sa, "positional": b_,
                             "positional_std": sb, "n_seeds_pos": n, "change_%": 100 * (b_ - a_) / a_})
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(out, "table_featmode.csv"), index=False, float_format="%.5f")
        latex_simple(t[["method", "union_aligned", "positional", "change_%"]].rename(
            columns={"method": "Method", "union_aligned": "Aligned (ours)", "positional": "Positional",
                     "change_%": r"$\Delta$ (\%)"}), os.path.join(out, "table_featmode.tex"),
            "Effect of input alignment on data1 (per-cell feature subsets): fixed union space with availability "
            "mask vs.\\ positional inputs.", "tab:featmode")
        print(t.to_string(index=False))

    # T9 hyper-parameter transfer
    f = os.path.join(base, "transfer_final", "long_results.csv")
    if os.path.exists(f):
        dt = pd.read_csv(f)
        rows = [{"setting": "FedPhase, tuned on this dataset", **dict(zip(["rmse_mean", "rmse_std", "n_seeds"],
                                                                         seed_mean(df_main, P)))}]
        for v in sorted(dt.variant.unique()):
            rows.append({"setting": f"FedPhase, {v.split('__')[-1]}", **dict(zip(["rmse_mean", "rmse_std", "n_seeds"],
                                                                                seed_mean(dt, v)))})
        for m in ["fedprox", "ditto"]:
            if m in methods:
                rows.append({"setting": LABELS[m] + " (tuned)", **dict(zip(["rmse_mean", "rmse_std", "n_seeds"],
                                                                          seed_mean(df_main, m)))})
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(out, "table_transfer.csv"), index=False, float_format="%.5f")
        latex_simple(t, os.path.join(out, "table_transfer.tex"),
                     "Hyper-parameter transfer: FedPhase with the hyper-parameters tuned on the 140-cell dataset.",
                     "tab:transfer")
        print(t.to_string(index=False))

    # tau sensitivity
    f = first(base, ["sweep_tau_final"], "sweep_table.csv")
    if f:
        t = pd.read_csv(f)
        t.to_csv(os.path.join(out, "table_sweep_tau.csv"), index=False, float_format="%.5f")
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.errorbar(t.lam, t["mean"], yerr=t["std"], color=COLORS[P], marker="o", linewidth=2.2, capsize=4,
                    label="FedPhase")
        for m in ["fedavg", "fedprox", "ditto", "ifca"]:
            if m in R:
                ax.axhline(R[m].mean(), color=COLORS[m], linestyle="--", linewidth=1.4, label=LABELS[m])
        ax.set_xscale("log", base=2); ax.set_xlabel(r"Kernel bandwidth scale $\tau$")
        ax.set_ylabel("Mean test RMSE"); ax.legend(frameon=False, ncol=2, fontsize=11)
        save(fig, out, "fig_sweep_tau_paper")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data1")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    base = os.path.join("results", a.dataset)
    out = os.path.join(base, "paper"); os.makedirs(out, exist_ok=True)

    df = pd.read_csv(os.path.join(base, "main_final", "long_results.csv"))
    methods = [m for m in ORDER if m in df.variant.unique()]
    seeds = sorted(set.intersection(*[set(df[df.variant == m].seed) for m in methods]))
    print("methods:", methods, "| common seeds:", seeds)
    R, clients = cube(df, methods, seeds)
    print(f"clients: {len(clients)}")

    tm = t_main(R, methods, seeds)
    tm.to_csv(os.path.join(out, "table_main.csv"), index=False, float_format="%.5f")
    latex_main(tm, os.path.join(out, "table_main.tex"), len(clients))

    ts = t_stats(R, methods, a.boot, rng)
    ts.to_csv(os.path.join(out, "table_stats.csv"), index=False, float_format="%.4g")
    latex_stats(ts, os.path.join(out, "table_stats.tex"))

    phi = df.groupby("partition")[[c for c in df.columns if c.startswith("phi_")] + ["n_rows"]].first()
    sg_fade, _ = t_subgroup(R, clients, phi, methods, "phi_fade_rate_late", ["slow fade", "medium fade", "fast fade"])
    sg_len, _ = t_subgroup(R, clients, phi, methods, "n_rows", ["short life", "medium life", "long life"])
    sg_soh, _ = t_subgroup(R, clients, phi, methods, "phi_soh_end", ["low SOH", "mid SOH", "high SOH"])
    sg = pd.concat([sg_fade.assign(split="fade rate (train window)"), sg_soh.assign(split="SOH at end of train"),
                    sg_len.assign(split="cell life length")])
    sg.to_csv(os.path.join(out, "table_subgroup.csv"), index=False, float_format="%.5f")
    cols = ["split", "group", "n_cells"] + [LABELS[m] for m in methods if m in FL] + ["gain_vs_best_%"]
    latex_simple(sg[cols].rename(columns={"gain_vs_best_%": r"Gain vs best (\%)"}),
                 os.path.join(out, "table_subgroup.tex"),
                 "Test RMSE by degradation regime (tertiles of the train-window phase descriptor) and by cell life length.",
                 "tab:subgroup")
    fig_subgroup(sg_fade, methods, "Capacity-fade rate in the training window", "fig_subgroup_fade", out)
    fig_subgroup(sg_soh, methods, "SOH at the end of the training window", "fig_subgroup_soh", out)

    # robustness: best-val-round protocol
    bv = os.path.join(base, "main", "long_results.csv")
    if os.path.exists(bv):
        db = pd.read_csv(bv)
        Rb, _ = cube(db[db.seed.isin(seeds)], methods, seeds)
        rp = pd.DataFrame({"method": [LABELS[m] for m in methods],
                           "rmse_last_round": [R[m].mean() for m in methods],
                           "rmse_best_val_round": [Rb[m].mean() for m in methods]})
        rp["rank_last"] = rp.rmse_last_round.rank().astype(int)
        rp["rank_best_val"] = rp.rmse_best_val_round.rank().astype(int)
        rp.to_csv(os.path.join(out, "table_protocol.csv"), index=False, float_format="%.5f")
        latex_simple(rp, os.path.join(out, "table_protocol.tex"),
                     "Robustness to the model-selection protocol: last round (primary) vs.\\ round with the lowest "
                     "validation error.", "tab:protocol")

    # ablation + sweep (from their own collectors)
    abl = first(base, ["ablation_final", "ablation4_final"], "ablation_table.csv")
    if abl:
        t = pd.read_csv(abl)
        t["variant_name"] = t.variant.map(ABL_NAMES).fillna(t.variant)
        t[["variant_name", "n_seeds", "rmse_mean", "rmse_std", "rmse_p90_mean", "delta_vs_full_%"]].to_csv(
            os.path.join(out, "table_ablation.csv"), index=False, float_format="%.5f")
        latex_simple(t[["variant_name", "rmse_mean", "rmse_std", "rmse_p90_mean", "delta_vs_full_%"]].rename(
            columns={"variant_name": "Variant", "delta_vs_full_%": r"$\Delta$ vs full (\%)"}),
            os.path.join(out, "table_ablation.tex"), "Ablation of FedPhase components (last round).", "tab:ablation")
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.barh(range(len(t))[::-1], t.rmse_mean, xerr=t.rmse_std, capsize=4,
                color=[COLORS[P] if v == "full" else "#999999" for v in t.variant])
        ax.set_yticks(range(len(t))[::-1]); ax.set_yticklabels(t.variant_name)
        ax.set_xlabel("Mean test RMSE")
        save(fig, out, "fig_ablation_paper")
    sw = first(base, ["sweep_lambda_final", "sweep4_final"], "sweep_table.csv")
    if sw:
        t = pd.read_csv(sw)
        t.to_csv(os.path.join(out, "table_sweep.csv"), index=False, float_format="%.5f")
        fig, ax = plt.subplots(figsize=(7, 4.8))
        ax.errorbar(t.lam, t["mean"], yerr=t["std"], color=COLORS[P], marker="o", linewidth=2.2, capsize=4,
                    label="FedPhase")
        for m in ["fedavg", "fedprox", "ditto", "ifca"]:
            if m in R:
                ax.axhline(R[m].mean(), color=COLORS[m], linestyle="--", linewidth=1.4, label=LABELS[m])
        ax.set_xlabel(r"Shrinkage $\lambda$ (0: data-size weights, 1: pure phase kernel)")
        ax.set_ylabel("Mean test RMSE")
        ax.legend(frameon=False, ncol=2, fontsize=11)
        save(fig, out, "fig_sweep_paper")

    tt = os.path.join(base, "main_final", "timing.csv")
    if os.path.exists(tt):
        t = pd.read_csv(tt)
        keep = [c for c in ["method", "round_wall_s", "compute_max_s", "agg_ms", "MB_per_round", "MB_per_client_round"]
                if c in t]
        t[keep].to_csv(os.path.join(out, "table_cost.csv"), index=False, float_format="%.4f")
        latex_simple(t[keep], os.path.join(out, "table_cost.tex"), "Per-round cost.", "tab:cost", fmt="%.3f")

    extra_tables(base, out, df, methods, seeds, R)

    fig_seeds(R, [m for m in methods if m in FL], seeds, out)
    fig_gain(ts[ts.variant.isin(FL + ["local", "centralized"])], out)
    fig_rank(tm, out)
    fig_scatter(R, ["fedprox", "ifca", "ditto"], out)

    # sentence-ready summary
    rp_ = tm.set_index("variant")
    L = [f"# FedPhase — summary ({a.dataset}, {len(seeds)} seeds, {len(clients)} cells, last-round protocol)", ""]
    L.append(f"FedPhase RMSE {rp_.loc[P, 'rmse_mean']:.4f} ± {rp_.loc[P, 'rmse_std']:.4f}; "
             f"best FL average rank: {tm[tm.variant.isin(FL)].sort_values('avg_rank').iloc[0].method}.")
    for _, r in ts.iterrows():
        sig = "significant" if r.client_p_holm < 0.05 and r.gain_ci_lo > 0 else "not significant"
        L.append(f"- vs {r.baseline}: {r['gain_%']:+.1f}% [{r.gain_ci_lo:+.1f}, {r.gain_ci_hi:+.1f}], "
                 f"better in {r.seed_win} seeds and {100 * r.client_win_rate:.0f}% of cells "
                 f"(p_seed={fmt_p(r.seed_p_holm)}, p_cell={fmt_p(r.client_p_holm)}) -> {sig}")
    L.append("")
    for _, r in sg_fade.iterrows():
        L.append(f"- {r.group}: gain vs best baseline ({r.best_baseline}) {r['gain_vs_best_%']:+.1f}%")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")

    pd.set_option("display.width", 220)
    print(tm[["method", "rmse_mean", "rmse_std", "rmse_p90_mean", "client_std_mean", "avg_rank",
              "seeds_fedphase_better"]].to_string(index=False))
    print(ts[["baseline", "gain_%", "gain_ci_lo", "gain_ci_hi", "seed_win", "seed_p_holm", "client_win_rate",
              "client_p_holm"]].to_string(index=False))
    print(sg[["split", "group", "n_cells", "best_baseline", "gain_vs_best_%"]
             + (["gain_vs_FedProx_%"] if "gain_vs_FedProx_%" in sg else [])].to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
