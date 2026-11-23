#!/usr/bin/env python
"""Compare a new FedPhase variant (15 seeds) with the paper's main table, MIT-140, LAST-round test RMSE.

  python <V4_CODE>/compare_variant.py --group dironly15 --variant fedphase_dironly --name Direction-only
  python <V4_CODE>/compare_variant.py --group eqbudget  --variant fedphase_eb      --name FedPhase-EB

Reads client_logs/<ds>/main/<method>/seed_*/error_client*.csv (the paper's 8 methods) and
      client_logs/<ds>/<group>/<variant>/seed_*/error_client*.csv
Only seeds present for every method are used (paired). Same statistics as paper_results.py:
  seed-level Wilcoxon (primary, Holm across the 7 baselines), hierarchical bootstrap CI of the gain,
  cell-level Wilcoxon on seed-averaged cells (secondary), win rates, rank-biserial; average rank among 9.
  Variant vs full FedPhase is reported separately (mean and RMSE_p90, not part of the Holm family).
-> results/<ds>/<group>/{table_<variant>_main.*, table_<variant>_stats.*, fig_<variant>_seeds.*, summary_<variant>.md}
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from v4common import COLORS, LABELS, ORDER, fmt_p, hier_boot_gain, holm, load_errors, plt, save

REF_5SEED = {"fedphase_dironly": 0.0513}      # paper, ablation (seeds 1-5) -> consistency check


def wilcoxon_p(x, y):
    return stats.wilcoxon(x, y).pvalue if len(x) >= 5 and np.any(x != y) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data2")
    ap.add_argument("--group", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default=None, help="restrict to a seed range, e.g. 6-15 (held-out seeds)")
    ap.add_argument("--col", default="final_rmse", help="final_rmse = last round (primary); rmse = best-val round")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ds, V = a.dataset, a.variant
    name = a.name or V
    out = os.path.join("results", ds, a.group + (f"_seeds{a.seeds}" if a.seeds else "")); os.makedirs(out, exist_ok=True)

    frames = []
    for m in ORDER:
        d = load_errors(os.path.join("client_logs", ds, "main", m))
        if not d.empty:
            frames.append(d.assign(variant=m))
    dv = load_errors(os.path.join("client_logs", ds, a.group, V))
    if dv.empty:
        raise SystemExit(f"no results under client_logs/{ds}/{a.group}/{V}")
    frames.append(dv.assign(variant=V))
    df = pd.concat(frames, ignore_index=True)
    methods = [m for m in ORDER if m in df.variant.unique()] + [V]
    seeds = sorted(set.intersection(*[set(df[df.variant == m].seed) for m in methods]))
    if a.seeds:
        lo, hi = (int(x) for x in a.seeds.split("-"))
        seeds = [x for x in seeds if lo <= x <= hi]
    clients = sorted(set.intersection(*[set(df[df.variant == m].partition) for m in methods]))
    print(f"methods {methods}\nseeds {seeds}  clients {len(clients)}")
    for m in methods:
        ns = df[df.variant == m].groupby("seed").partition.nunique()
        bad = ns[ns < len(clients)]
        if len(bad):
            print(f"WARNING {m}: incomplete seeds {dict(bad)}")
    R = {m: df[df.variant == m].pivot_table(index="seed", columns="partition", values=a.col).loc[seeds, clients].values
         for m in methods}
    lab = {**LABELS, V: name}
    S = len(seeds)

    # ---- main table (9 rows)
    allm = np.stack([R[m] for m in methods])
    ranks = stats.rankdata(allm, axis=0).mean(axis=(1, 2))
    rows = []
    for i, m in enumerate(methods):
        s, p90 = R[m].mean(1), np.quantile(R[m], 0.9, axis=1)
        rows.append({"method": lab[m], "variant": m, "n_seeds": S, "rmse_mean": s.mean(), "rmse_std": s.std(ddof=1),
                     "rmse_p90_mean": p90.mean(), "rmse_p90_std": p90.std(ddof=1), "avg_rank": ranks[i],
                     "seeds_variant_better": "" if m == V else f"{int((R[V].mean(1) < s).sum())}/{S}"})
    tm = pd.DataFrame(rows)
    tm.to_csv(os.path.join(out, f"table_{V}_main.csv"), index=False, float_format="%.5f")

    # ---- paired statistics: variant vs 7 baselines (Holm family) + vs full FedPhase (separate)
    def pair(m):
        A, B = R[V], R[m]
        g = hier_boot_gain(A, B, a.boot, rng)
        sa, sb = A.mean(1), B.mean(1)
        ca, cb = A.mean(0), B.mean(0)
        d = cb - ca; nz = d[d != 0]; rk = stats.rankdata(np.abs(nz))
        pa, pb = np.quantile(A, 0.9, axis=1), np.quantile(B, 0.9, axis=1)
        return {"baseline": lab[m], "variant": m, "rmse_baseline": B.mean(), "rmse_variant": A.mean(),
                "gain_%": 100 * (B.mean() - A.mean()) / B.mean(),
                "gain_ci_lo": np.percentile(g, 2.5), "gain_ci_hi": np.percentile(g, 97.5),
                "seed_win": f"{int((sa < sb).sum())}/{S}", "seed_wilcoxon_p": wilcoxon_p(sa, sb),
                "client_win_rate": (d > 0).mean(), "client_wilcoxon_p": stats.wilcoxon(ca, cb).pvalue,
                "rank_biserial": (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if rk.size else 0.0,
                "p90_baseline": pb.mean(), "p90_variant": pa.mean(),
                "p90_seed_win": f"{int((pa < pb).sum())}/{S}", "p90_seed_wilcoxon_p": wilcoxon_p(pa, pb)}

    base = [m for m in methods if m not in (V, "fedphase")]
    ts = pd.DataFrame([pair(m) for m in base])
    ts["seed_p_holm"] = holm(ts.seed_wilcoxon_p.fillna(1).values)
    ts["client_p_holm"] = holm(ts.client_wilcoxon_p.values)
    if "fedphase" in methods:
        full = pair("fedphase"); full["seed_p_holm"] = full["client_p_holm"] = np.nan
        ts = pd.concat([ts, pd.DataFrame([full])], ignore_index=True)
    ts.to_csv(os.path.join(out, f"table_{V}_stats.csv"), index=False, float_format="%.4g")

    L = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         rf"vs.\ & Gain of {name} (\%) [95\% CI] & Seeds & $p_{{\text{{seed}}}}$ & $p_{{\text{{cell}}}}$ & $r_{{rb}}$ \\",
         r"\midrule"]
    for _, r in ts.iterrows():
        ps = r.seed_p_holm if np.isfinite(r.seed_p_holm) else r.seed_wilcoxon_p
        pc = r.client_p_holm if np.isfinite(r.client_p_holm) else r.client_wilcoxon_p
        sep = r"\midrule" + "\n" if r.variant == "fedphase" else ""
        L.append(f"{sep}{r.baseline} & {r['gain_%']:+.1f} [{r.gain_ci_lo:+.1f}, {r.gain_ci_hi:+.1f}] & {r.seed_win} & "
                 f"{fmt_p(ps, True)} & {fmt_p(pc, True)} & {r.rank_biserial:+.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, f"table_{V}_stats.tex"), "w").write("\n".join(L))

    T = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Method & RMSE & RMSE$_{p90}$ & Rank \\", r"\midrule"]
    fl_best = tm[~tm.variant.isin(["centralized", "local"])].rmse_mean.min()
    for _, r in tm.iterrows():
        x = f"{r.rmse_mean:.4f} $\\pm$ {r.rmse_std:.4f}"
        if np.isclose(r.rmse_mean, fl_best):
            x = rf"\textbf{{{x}}}"
        T.append(f"{r.method} & {x} & {r.rmse_p90_mean:.4f} & {r.avg_rank:.2f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, f"table_{V}_main.tex"), "w").write("\n".join(T))

    # ---- figure: per-seed RMSE, variant vs FedProx and full FedPhase
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(S)
    for m, mk in [("fedprox", "s"), ("fedavg", "^"), ("fedphase", "o")]:
        if m in R:
            ax.plot(x, R[m].mean(1), mk + "-", color=COLORS[m], lw=1.4, ms=6, label=LABELS[m])
    ax.plot(x, R[V].mean(1), "D-", color="black", lw=1.8, ms=6, label=name)
    ax.set_xticks(x); ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed"); ax.set_ylabel("Mean test RMSE (last round)")
    ax.legend(frameon=False, ncol=2)
    save(fig, out, f"fig_{V}_seeds")

    # ---- summary
    fp = ts.set_index("variant")
    lines = [f"# {name} on MIT-140 ({S} seeds {seeds[0]}-{seeds[-1]}, {len(clients)} cells, {a.col})", "",
             tm[["method", "rmse_mean", "rmse_std", "rmse_p90_mean", "avg_rank", "seeds_variant_better"]]
             .to_string(index=False, float_format=lambda v: f"{v:.4f}"), ""]
    for m in [x for x in ("fedprox", "fedavg", "fedphase") if x in fp.index]:
        r = fp.loc[m]
        ps = r.seed_p_holm if np.isfinite(r.seed_p_holm) else r.seed_wilcoxon_p
        lines.append(f"- vs {r.baseline}: RMSE {r.rmse_variant:.4f} vs {r.rmse_baseline:.4f}, gain {r['gain_%']:+.1f}% "
                     f"[{r.gain_ci_lo:+.1f}, {r.gain_ci_hi:+.1f}], seeds {r.seed_win}, p_seed{' (Holm)' if m != 'fedphase' else ''} "
                     f"= {fmt_p(ps)}, p_cell = {fmt_p(r.client_wilcoxon_p)}; p90 {r.p90_variant:.4f} vs {r.p90_baseline:.4f} "
                     f"(p = {fmt_p(r.p90_seed_wilcoxon_p)})")
    if V in REF_5SEED and set(range(1, 6)) <= set(seeds):
        i5 = [seeds.index(s) for s in range(1, 6)]
        v5 = R[V][i5].mean()
        lines += ["", f"Consistency: seeds 1-5 give {v5:.4f} (paper ablation: {REF_5SEED[V]:.4f})"
                  + ("" if abs(v5 - REF_5SEED[V]) < 5e-4 else "  <-- MISMATCH, check")]
    sig = "fedprox" in fp.index and fp.loc["fedprox"].seed_p_holm < 0.05
    lines += ["", f"Significant vs FedProx at seed level (Holm, 0.05): {'YES' if sig else 'NO'}"]
    open(os.path.join(out, f"summary_{V}.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
