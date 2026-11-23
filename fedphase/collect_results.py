#!/usr/bin/env python
"""Collect one experiment group into paper-ready tables and figures.

  python collect_results.py --dataset data1 --group main          # main comparison (last round)
  python collect_results.py --dataset data1 --group main --protocol bestval
  python collect_results.py --dataset data1 --group tune          # -> best_hparams.env (tune, tune2, tune3)
  python collect_results.py --dataset data1 --group ablation      # FedPhase ablation
  python collect_results.py --dataset data1 --group sweep_lambda  # sensitivity (sweep_*)
  Groups by prefix: tune* -> tuning; ablation* -> ablation; sweep* -> ablation + sweep;
  everything else (main, smoke, horizon_*, featmode_*, transfer) -> main-type tables.

Reads   client_logs/<dataset>/<group>/<variant>/seed_*/error_client*.csv (+ rounds)
        server_logs/<dataset>/<group>/<variant>/seed_*/...
Writes  results/<dataset>/<group>/

main:     main_table.csv/.tex  stats_vs_fedphase.csv/.tex  fairness in main_table
          per_client.csv  timing.csv/.tex  long_results.csv
          fig_convergence  fig_rmse_box  fig_rmse_cdf  fig_pred_examples
          fig_gain_vs_phase  fig_phase_map  fig_alpha_heatmap  fig_round_time
tune:     tune_table.csv  best_hparams.env   (selection on VALIDATION only)
ablation: ablation_table.csv/.tex  fig_ablation
All metrics are TEST metrics at each client's best-VAL round unless stated.
"""
import argparse
import glob
import os
import re
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
PROPOSED = "fedphase"
METRICS = ["rmse", "mae", "nrmse", "mape"]


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def lab(v):
    base = v.split("__")[0]
    return LABELS.get(v, LABELS.get(base, base) + (" " + v.split("__", 1)[1] if "__" in v else ""))


# ------------------------------------------------------------------ load
def load_long(root):
    rows = []
    for vdir in sorted(glob.glob(os.path.join(root, "*"))):
        v = os.path.basename(vdir)
        for sd in sorted(glob.glob(os.path.join(vdir, "seed_*"))):
            fs = glob.glob(os.path.join(sd, "error_client*.csv"))
            if fs:
                d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
                d["variant"], d["seed"] = v, int(sd.rsplit("_", 1)[1])
                rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def curves(root, variant, col="test_rmse"):
    per_seed = []
    for sd in sorted(glob.glob(os.path.join(root, variant, "seed_*"))):
        fs = glob.glob(os.path.join(sd, "client_*_rounds.csv"))
        if fs:
            d = pd.concat([pd.read_csv(f, usecols=["round", col]) for f in fs])
            per_seed.append(d.groupby("round")[col].mean())
    return pd.concat(per_seed, axis=1) if per_seed else None


# ----------------------------------------------------------- main table
def per_seed_table(df):
    g = df.groupby(["variant", "seed"])
    t = g[METRICS].mean()
    t["rmse_p90"] = g["rmse"].quantile(0.9)                 # worst-10 % client
    t["rmse_client_std"] = g["rmse"].std()                   # fairness across clients
    t["final_rmse"] = g["final_rmse"].mean()
    return t.reset_index()


def main_table(df, variants):
    ps = per_seed_table(df)
    cm = df.groupby(["variant", "partition"])["rmse"].mean().unstack(0)   # seed-avg per client
    cols = METRICS + ["rmse_p90", "rmse_client_std", "final_rmse"]
    rows = []
    for v in variants:
        s = ps[ps.variant == v]
        r = {"method": lab(v), "variant": v, "n_seeds": len(s), "n_clients": int(cm[v].notna().sum())}
        for c in cols:
            r[f"{c}_mean"], r[f"{c}_std"] = s[c].mean(), s[c].std(ddof=1) if len(s) > 1 else np.nan
        if PROPOSED in cm and v != PROPOSED:
            ok = cm[v].notna() & cm[PROPOSED].notna()
            r["fedphase_wins"] = f"{int((cm.loc[ok, PROPOSED] < cm.loc[ok, v]).sum())}/{int(ok.sum())}"
            r["fedphase_gain_%"] = 100 * (cm.loc[ok, v].mean() - cm.loc[ok, PROPOSED].mean()) / cm.loc[ok, v].mean()
        rows.append(r)
    return pd.DataFrame(rows), cm


def to_latex(mt, path, caption, label):
    fl_rows = mt[mt.variant.isin(FL)]
    best = {c: fl_rows[f"{c}_mean"].min() for c in ["rmse", "mae", "nrmse", "mape", "rmse_p90"]}
    L = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Method & RMSE & MAE & NRMSE & MAPE (\%) & RMSE$_{p90}$ \\", r"\midrule"]
    for _, r in mt.iterrows():
        cells = []
        for c, sc in [("rmse", 1), ("mae", 1), ("nrmse", 1), ("mape", 1), ("rmse_p90", 1)]:
            m, s = r[f"{c}_mean"] * sc, r[f"{c}_std"] * sc
            txt = f"{m:.4f}" + (rf" $\pm$ {s:.4f}" if np.isfinite(s) else "")
            if r.variant in FL and np.isclose(r[f"{c}_mean"], best[c]):
                txt = rf"\textbf{{{txt}}}"
            cells.append(txt)
        name = r["method"] + (r"$^\dagger$" if r.variant == "centralized" else "")
        L.append(f"{name} & " + " & ".join(cells) + r" \\")
        if r.variant == "local":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}",
          r"\\[2pt]\footnotesize $^\dagger$Non-private upper bound (pooled data). Mean $\pm$ std over seeds; "
          r"per-client test error at the best-validation round, macro-averaged over clients.",
          r"\end{table}"]
    open(path, "w").write("\n".join(L))


def stats_table(cm):
    rows = []
    for v in [x for x in ORDER if x in cm and x != PROPOSED]:
        ok = cm[v].notna() & cm[PROPOSED].notna()
        a, b = cm.loc[ok, PROPOSED].values, cm.loc[ok, v].values
        d = b - a
        w = stats.wilcoxon(a, b)
        rk = stats.rankdata(np.abs(d[d != 0]))
        rrb = (rk[d[d != 0] > 0].sum() - rk[d[d != 0] < 0].sum()) / rk.sum() if rk.size else 0.0
        rows.append({"comparison": f"FedPhase vs {LABELS[v]}", "n_clients": int(ok.sum()),
                     "rmse_fedphase": a.mean(), "rmse_other": b.mean(), "mean_diff": d.mean(),
                     "median_diff": np.median(d), "win_rate": (d > 0).mean(),
                     "rank_biserial": rrb, "wilcoxon_p": w.pvalue,
                     "paired_t_p": stats.ttest_rel(a, b).pvalue})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    o = np.argsort(t.wilcoxon_p.values)
    k = len(t)
    holm = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * t.wilcoxon_p.values[o]))
    t.loc[t.index[o], "p_holm"] = holm
    t["significant_0.05"] = t.p_holm < 0.05
    return t


def stats_latex(st, path):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired per-client comparison (seed-averaged test RMSE). $\Delta$ = baseline $-$ FedPhase "
         r"($>0$: FedPhase better). Wilcoxon signed-rank, Holm-corrected; $r_{rb}$: rank-biserial effect size.}",
         r"\label{tab:stats}", r"\begin{tabular}{lccccc}", r"\toprule",
         r"Baseline & mean $\Delta$ & win rate & $r_{rb}$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    for _, r in st.iterrows():
        L.append(f"{r.comparison.replace('FedPhase vs ', '')} & {r.mean_diff:.4f} & {100 * r.win_rate:.1f}\\% & "
                 f"{r.rank_biserial:.2f} & {r.p_holm:.2g} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L))


def timing_table(sroot, variants):
    rows = []
    for v in variants:
        rts = glob.glob(os.path.join(sroot, v, "seed_*", "round_times.csv"))
        if not rts:
            continue
        rt = pd.concat([pd.read_csv(f) for f in rts])
        tot = [float(pd.read_csv(f)["total_wall_s"].iloc[0])
               for f in glob.glob(os.path.join(sroot, v, "seed_*", "run_summary.csv"))]
        rows.append({"method": lab(v), "variant": v, "n_seeds": len(rts),
                     "total_wall_s": np.mean(tot) if tot else np.nan,
                     "round_wall_s": rt.round_wall_s.mean(), "fit_wall_s": rt.fit_wall_s.mean(),
                     "eval_wall_s": rt.eval_wall_s.mean(), "compute_max_s": rt.compute_max_s.mean(),
                     "compute_mean_s": rt.compute_mean_s.mean(), "overhead_s": rt.overhead_s.mean(),
                     "agg_ms": rt.agg_ms.mean(),
                     "MB_per_round": (rt.MB_down + rt.MB_up).mean(),
                     "MB_per_client_round": ((rt.MB_down + rt.MB_up) / rt.n_clients).mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plot_box_cdf(cm, variants, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [cm[v].dropna().values for v in variants]
    bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True)
    for patch, v in zip(bp["boxes"], variants):
        patch.set_facecolor(COLORS.get(v, "w")); patch.set_alpha(0.35)
    ax.set_xticks(range(1, len(variants) + 1))
    ax.set_xticklabels([lab(v).replace(" (ours)", "\n(ours)") for v in variants], rotation=0)
    ax.set_ylabel("Per-client test RMSE")
    save(fig, out, "fig_rmse_box")

    fig, ax = plt.subplots(figsize=(7, 5))
    for v in variants:
        x = np.sort(cm[v].dropna().values)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLORS.get(v), label=lab(v),
                linewidth=2.6 if v == PROPOSED else 1.6)
    ax.set_xlabel("Per-client test RMSE"); ax.set_ylabel("CDF over clients")
    ax.set_xscale("log")
    ax.legend(frameon=False)
    save(fig, out, "fig_rmse_cdf")


def plot_convergence(root, variants, out):
    fig, ax = plt.subplots(figsize=(8, 5))
    for v in variants:
        c = curves(root, v)
        if c is None:
            continue
        mu, sd = c.mean(1), c.std(1).fillna(0)
        ax.plot(mu.index, mu.values, color=COLORS.get(v), label=lab(v),
                linewidth=2.6 if v == PROPOSED else 1.6)
        ax.fill_between(mu.index, mu - sd, mu + sd, color=COLORS.get(v), alpha=0.12)
    ax.set_xlabel("Round"); ax.set_ylabel("Mean test RMSE across clients")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=2)
    save(fig, out, "fig_convergence")


def plot_pred_examples(root, df, variants, out):
    if PROPOSED not in variants or "phi_fade_rate_late" not in df:
        return
    seed = sorted(df.seed.unique())[0]
    ph = df[(df.variant == PROPOSED) & (df.seed == seed)].set_index("partition")["phi_fade_rate_late"]
    picks = [ph.index[np.argmin(np.abs(ph.values - ph.quantile(q)))] for q in (0.9, 0.5, 0.1)]
    show = [v for v in ["fedavg", "fedprox", "ditto", "ifca", "fedphase"] if v in variants]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    for ax, pid, name in zip(axes, picks, ["slow fade", "median fade", "fast fade"]):
        truth = None
        for v in show:
            f = os.path.join(root, v, f"seed_{seed}", f"prediction_client{pid}.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            if truth is None:
                truth = d
                ax.plot(d.cycle, d.true, color="k", linewidth=2.4, label="Measured")
                for s, ls in [("val", ":"), ("test", "--")]:
                    if (d.split == s).any():
                        ax.axvline(d.cycle[d.split == s].min(), color="grey", linestyle=ls, linewidth=1.2)
            ax.plot(d.cycle, d.pred, color=COLORS[v], linewidth=2.2 if v == PROPOSED else 1.3,
                    label=lab(v), alpha=0.95)
        ax.set_title(f"Cell {pid} ({name})")
        ax.set_xlabel("Cycle")
    axes[0].set_ylabel("Normalised capacity")
    axes[-1].legend(frameon=False, fontsize=11)
    save(fig, out, "fig_pred_examples")


def plot_gain_vs_phase(df, cm, out):
    if PROPOSED not in cm or "phi_fade_rate_late" not in df:
        return
    ph = df[df.variant == PROPOSED].groupby("partition")["phi_fade_rate_late"].mean()
    others = [v for v in ["fedprox", "ditto", "ifca"] if v in cm]
    if not others:
        return
    fig, axes = plt.subplots(1, len(others), figsize=(5.2 * len(others), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, v in zip(axes, others):
        g = (cm[v] - cm[PROPOSED]).reindex(ph.index)
        ax.scatter(ph.values, g.values, s=22, color=COLORS[PROPOSED], alpha=0.7, edgecolor="none")
        ax.axhline(0, color="k", linewidth=1.2)
        ax.set_xlabel("Capacity fade rate (per 100 cycles)")
        ax.set_title(f"vs {LABELS[v]}")
    axes[0].set_ylabel(r"RMSE$_{\mathrm{baseline}}$ $-$ RMSE$_{\mathrm{FedPhase}}$")
    save(fig, out, "fig_gain_vs_phase")


def plot_phase(sroot, out):
    sd = sorted(glob.glob(os.path.join(sroot, PROPOSED, "seed_*")))
    if not sd or not os.path.exists(os.path.join(sd[0], "phase_descriptors.csv")):
        return
    ph = pd.read_csv(os.path.join(sd[0], "phase_descriptors.csv")).sort_values("partition")
    keys = [c for c in ph.columns if c != "partition"]
    Z = (ph[keys] - ph[keys].mean()) / (ph[keys].std() + 1e-8)
    U, S, Vt = np.linalg.svd(Z.values, full_matrices=False)
    pc = U[:, :2] * S[:2]
    colour = ph["fade_rate_late"] if "fade_rate_late" in ph else ph[keys[0]]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sc = ax.scatter(pc[:, 0], pc[:, 1], c=colour, cmap="viridis", s=36, edgecolor="k", linewidth=0.4)
    cb = fig.colorbar(sc, ax=ax); cb.set_label(colour.name.replace("_", " "))
    ev = S ** 2 / (S ** 2).sum()
    ax.set_xlabel(f"Phase PC1 ({100 * ev[0]:.0f}%)"); ax.set_ylabel(f"Phase PC2 ({100 * ev[1]:.0f}%)")
    save(fig, out, "fig_phase_map")

    ap = os.path.join(sd[0], "alpha.npy")
    if os.path.exists(ap):
        A = np.load(ap)                      # rows/cols sorted by partition id
        o = np.argsort(colour.values)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(A[np.ix_(o, o)], cmap="magma_r", aspect="auto")
        fig.colorbar(im, ax=ax).set_label(r"aggregation weight $a_{ij}$")
        ax.set_xlabel("Client j (sorted by fade rate)"); ax.set_ylabel("Client i (sorted by fade rate)")
        save(fig, out, "fig_alpha_heatmap")


def plot_timing(tt, out):
    if tt.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tt)); bottom = np.zeros(len(tt))
    for col, name, c in [("compute_max_s", "Client compute (slowest)", "#4c72b0"),
                         ("overhead_s", "Communication + waiting", "#dd8452"),
                         ("eval_wall_s", "Evaluation", "#55a868")]:
        ax.bar(x, tt[col].values, 0.6, bottom=bottom, label=name, color=c)
        bottom += tt[col].values
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" (ours)", "\n(ours)") for m in tt.method])
    ax.set_ylabel("Wall time per round (s)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=11)
    save(fig, out, "fig_round_time")


# ----------------------------------------------------------------- groups
def do_main(root, sroot, out, df, dataset="data1", group="main"):
    variants = [v for v in ORDER if v in df.variant.unique()] or sorted(df.variant.unique())
    mt, cm = main_table(df, variants)
    mt.to_csv(os.path.join(out, "main_table.csv"), index=False, float_format="%.5f")
    to_latex(mt, os.path.join(out, "main_table.tex"),
             f"Test capacity-prediction error on {dataset} ({df.partition.nunique()} cells, group {group.replace('_', ' ')}).",
             f"tab:{dataset}_{group}")
    cm.rename(columns=LABELS).to_csv(os.path.join(out, "per_client.csv"), float_format="%.5f")
    st = stats_table(cm)
    if not st.empty:
        st.to_csv(os.path.join(out, "stats_vs_fedphase.csv"), index=False, float_format="%.4g")
        stats_latex(st, os.path.join(out, "stats_vs_fedphase.tex"))
    tt = timing_table(sroot, [v for v in variants if v in FL])
    if not tt.empty:
        tt.to_csv(os.path.join(out, "timing.csv"), index=False, float_format="%.4f")
    plot_box_cdf(cm, variants, out)
    plot_convergence(root, variants, out)
    plot_pred_examples(root, df, variants, out)
    plot_gain_vs_phase(df, cm, out)
    plot_phase(sroot, out)
    plot_timing(tt, out)
    pd.set_option("display.width", 220)
    print(mt[["method", "n_seeds", "rmse_mean", "rmse_std", "mae_mean", "nrmse_mean", "rmse_p90_mean"]
             + [c for c in ["fedphase_wins", "fedphase_gain_%"] if c in mt]].to_string(index=False))
    if not st.empty:
        print(st[["comparison", "mean_diff", "win_rate", "rank_biserial", "p_holm"]].to_string(index=False))
    if not tt.empty:
        print(tt[["method", "round_wall_s", "agg_ms", "MB_per_round"]].to_string(index=False))


def do_tune(out, df, protocol="final"):
    """variant naming: <method>__<KEY>=<value>. Score = mean VAL RMSE at the last round (protocol final,
    same as the reported test protocol) or at the best-val round (protocol bestval)."""
    col = "final_val_rmse" if protocol == "final" and "final_val_rmse" in df else "val_rmse"
    print(f"tuning score: {col}")
    s = df.groupby(["variant", "seed"])[col].mean().groupby("variant").mean().rename("val_rmse")
    t = s.reset_index()
    t["method"] = t.variant.str.split("__").str[0]
    t["param"] = t.variant.str.split("__").str[1]
    t = t.sort_values(["method", "val_rmse"])
    t.to_csv(os.path.join(out, "tune_table.csv"), index=False, float_format="%.5f")
    env = []
    for m, g in t.groupby("method"):
        best = g.iloc[0]
        if isinstance(best.param, str) and "=" in best.param:
            for kv in best.param.split(","):
                k, v = kv.split("=")
                env.append(f"{k}={v}")
    open(os.path.join(out, "best_hparams.env"), "w").write("\n".join(sorted(set(env))) + "\n")
    print(t.to_string(index=False)); print("best ->", env)


def do_ablation(root, out, df):
    variants = sorted(df.variant.unique(), key=lambda v: (v != "full", v))
    ps = per_seed_table(df)
    rows = []
    ref = ps[ps.variant == "full"].rmse.mean() if "full" in variants else np.nan
    for v in variants:
        s = ps[ps.variant == v]
        rows.append({"variant": v, "n_seeds": len(s), "rmse_mean": s.rmse.mean(), "rmse_std": s.rmse.std(ddof=1),
                     "rmse_p90_mean": s.rmse_p90.mean(), "delta_vs_full_%": 100 * (s.rmse.mean() - ref) / ref})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "ablation_table.csv"), index=False, float_format="%.5f")
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Variant & RMSE & RMSE$_{p90}$ & $\Delta$ vs full \\", r"\midrule"]
    for _, r in t.iterrows():
        L.append(f"{r.variant.replace('_', ' ')} & {r.rmse_mean:.4f} $\\pm$ {r.rmse_std:.4f} & "
                 f"{r.rmse_p90_mean:.4f} & {r['delta_vs_full_%']:+.1f}\\% \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "ablation_table.tex"), "w").write("\n".join(L))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(range(len(t)), t.rmse_mean, yerr=t.rmse_std, capsize=4,
           color=[COLORS[PROPOSED] if v == "full" else "#999999" for v in t.variant])
    ax.set_xticks(range(len(t))); ax.set_xticklabels([v.replace("_", "\n") for v in t.variant])
    ax.set_ylabel("Mean test RMSE")
    save(fig, out, "fig_ablation")
    print(t.to_string(index=False))


def do_sweep(out, df, dataset, group="sweep_lambda"):
    """RMSE vs a FedPhase parameter (variant '<name>_<value>', e.g. shrink_0.5 / tau_1.0),
    with FedAvg/FedProx/Ditto/IFCA (main_final) as horizontal references."""
    ps = per_seed_table(df)
    ps["lam"] = ps.variant.str.split("_").str[-1].astype(float)
    g = ps.groupby("lam").rmse.agg(["mean", "std"]).reset_index()
    g.to_csv(os.path.join(out, "sweep_table.csv"), index=False, float_format="%.5f")
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.errorbar(g.lam, g["mean"], yerr=g["std"], color=COLORS[PROPOSED], marker="o", linewidth=2.2, capsize=4,
                label="FedPhase")
    ref = os.path.join("results", dataset, "main_final", "long_results.csv")
    if os.path.exists(ref):
        r = pd.read_csv(ref); r = r[r.variant.isin(["fedavg", "fedprox", "ditto", "ifca"])]
        for v, m in r.groupby("variant").apply(lambda x: x.groupby("seed").rmse.mean().mean()).items():
            ax.axhline(m, color=COLORS[v], linestyle="--", linewidth=1.4, label=LABELS[v])
    if "tau" in group:
        ax.set_xlabel(r"Kernel bandwidth scale $\tau$"); ax.set_xscale("log", base=2)
    else:
        ax.set_xlabel(r"Shrinkage $\lambda$ (0 = FedProx, 1 = pure phase kernel)")
    ax.set_ylabel("Mean test RMSE (last round)")
    ax.legend(frameon=False, ncol=2)
    save(fig, out, "fig_" + group)
    print(g.to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data1")
    p.add_argument("--group", default="main")
    p.add_argument("--protocol", default="final", choices=["final", "bestval"],
                   help="final: report the LAST round (fixed budget, primary); "
                        "bestval: round with the lowest validation MSE (sensitivity analysis)")
    a = p.parse_args()
    root = os.path.join("client_logs", a.dataset, a.group)
    sroot = os.path.join("server_logs", a.dataset, a.group)
    is_tune = a.group.startswith("tune")
    use_final = a.protocol == "final" and not is_tune
    out = os.path.join("results", a.dataset, a.group + ("_final" if use_final else ""))
    os.makedirs(out, exist_ok=True)
    df = load_long(root)
    if df.empty:
        sys.exit(f"no results under {root}")
    if use_final:   # report the last-round test metrics; keep best-val ones as bv_*
        for m in ["mse", "rmse", "mae", "nrmse", "mape", "r2", "maxe"]:
            if f"final_{m}" in df:
                df[f"bv_{m}"] = df[m]
                df[m] = df[f"final_{m}"]
    df.to_csv(os.path.join(out, "long_results.csv"), index=False)
    if is_tune:
        do_tune(out, df, a.protocol)
    elif a.group.startswith("sweep"):
        do_ablation(root, out, df); do_sweep(out, df, a.dataset, a.group)
    elif a.group.startswith("ablation"):
        do_ablation(root, out, df)
    else:
        do_main(root, sroot, out, df, a.dataset, a.group)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
