#!/usr/bin/env python
"""Reviewer comment 5: equal tuning budget (4 configurations per method), MIT-140, seed-0 VALIDATION only.

FedPhase grid (fixed a priori): kappa = 1, tau in {0.5, 1}, lambda in {0.5, 0.75}, mu = FedProx's.
  tau = 0.5 points: tune5/fedphase5__TAU=0.5,SHRINK=*  if re-run, else tune4/fedphase4__SHRINK=*
  tau = 1   points: tune5/fedphase5__TAU=1.0,SHRINK=*
Baselines: their existing 4-point grids in tune/ (FedProx mu, Ditto lambda, IFCA K).

Selection score (default) = mean over cells of the LAST-round val RMSE (same protocol as the reported test RMSE;
it reproduces the paper's FedProx/Ditto/IFCA/FedPhase choices). The best-round val RMSE is reported too (--score bestround).
The hyper-parameters of every FedPhase run are read back from its server config.csv (no assumption on names).

  python <V4_CODE>/equal_budget.py --dataset data2        (cwd = fedphase)
-> results/<ds>/equal_budget/{table_equal_budget.csv, table_equal_budget.tex, selected_fedphase.env, summary.md}
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from v4common import val_scores

PAPER = {"tau_scale": 0.5, "lookahead": 1.0, "shrink": 0.75}
GRID = [(t, l) for t in (0.5, 1.0) for l in (0.5, 0.75)]


def server_cfg(dataset, group, variant):
    f = os.path.join("server_logs", dataset, group, variant, "seed_0", "config.csv")
    if not os.path.exists(f):
        return {}
    d = pd.read_csv(f, header=None, names=["k", "v"])
    return dict(zip(d.k, d.v))


def run_row(dataset, method, group, variant, param):
    best, last, nc, nr = val_scores(os.path.join("client_logs", dataset, group, variant, "seed_0"))
    return {"method": method, "param": param, "group": group, "variant": variant,
            "val_rmse_bestround": best, "val_rmse_lastround": last, "n_clients": nc, "n_rounds": nr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data2")
    ap.add_argument("--score", default="lastround", choices=["bestround", "lastround"],
                    help="lastround = val RMSE of the last round (same protocol as the reported test RMSE; default)")
    ap.add_argument("--parts", type=int, default=140)
    ap.add_argument("--rounds", type=int, default=100)
    a = ap.parse_args()
    ds = a.dataset
    out = os.path.join("results", ds, "equal_budget"); os.makedirs(out, exist_ok=True)
    score = f"val_rmse_{a.score}"

    rows = []
    for m in ("fedprox", "ditto", "ifca"):
        for vd in sorted(glob.glob(os.path.join("client_logs", ds, "tune", f"{m}__*"))):
            v = os.path.basename(vd)
            rows.append(run_row(ds, m, "tune", v, v.split("__", 1)[1]))

    problems = []
    for t, l in GRID:
        cands = [("tune5", f"fedphase5__TAU={t},SHRINK={l}")]
        if t == 0.5:
            cands.append(("tune4", f"fedphase4__SHRINK={l}"))
        hit = next(((g, v) for g, v in cands if os.path.isdir(os.path.join("client_logs", ds, g, v, "seed_0"))), None)
        if hit is None:
            problems.append(f"missing run for tau={t}, lambda={l} (looked in {cands})")
            rows.append({"method": "fedphase", "param": f"TAU={t},LOOKAHEAD=1.0,SHRINK={l}",
                         "val_rmse_bestround": np.nan, "val_rmse_lastround": np.nan})
            continue
        r = run_row(ds, "fedphase", *hit, f"TAU={t},LOOKAHEAD=1.0,SHRINK={l}")
        cfg = server_cfg(ds, *hit)
        r.update({k: float(cfg[k]) for k in ("mu", "tau_scale", "lookahead", "shrink", "adaptive_mu") if k in cfg})
        if cfg and (abs(r["tau_scale"] - t) > 1e-9 or abs(r["shrink"] - l) > 1e-9 or abs(r["lookahead"] - 1.0) > 1e-9):
            problems.append(f"{hit}: server config {cfg} does not match grid point tau={t}, kappa=1, lambda={l}")
        if not cfg:
            problems.append(f"{hit}: no server config.csv, hyper-parameters not verified")
        rows.append(r)

    t = pd.DataFrame(rows)
    for _, r in t.iterrows():
        if r.get("n_clients", a.parts) != a.parts or r.get("n_rounds", a.rounds) < a.rounds:
            problems.append(f"{r.group}/{r.variant}: {r.n_clients} clients, {r.n_rounds} rounds (incomplete?)")
    t["selected"] = False
    for m, g in t.groupby("method"):
        if g[score].notna().any():
            t.loc[g[score].idxmin(), "selected"] = True
    t = t.sort_values(["method", "param"])
    t.to_csv(os.path.join(out, "table_equal_budget.csv"), index=False, float_format="%.6f")
    print(t[["method", "param", "val_rmse_bestround", "val_rmse_lastround", "selected"]].to_string(index=False))

    fp = t[(t.method == "fedphase") & t.selected]
    if fp.empty or t[t.method == "fedphase"][score].isna().any():
        raise SystemExit("FedPhase grid incomplete:\n  " + "\n  ".join(problems))
    s = fp.iloc[0]
    kv = dict(x.split("=") for x in s.param.split(","))
    same = int(abs(float(kv["TAU"]) - PAPER["tau_scale"]) < 1e-9 and abs(float(kv["SHRINK"]) - PAPER["shrink"]) < 1e-9)
    with open(os.path.join(out, "selected_fedphase.env"), "w") as f:
        f.write(f"TAU={kv['TAU']}\nLOOKAHEAD={kv['LOOKAHEAD']}\nSHRINK={kv['SHRINK']}\nSAME_AS_PAPER={same}\n")

    # ---- LaTeX: FedPhase grid + selection of every method
    fpg = t[t.method == "fedphase"].copy()
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"$\tau$ & $\lambda$ & Val.\ RMSE (best round) & Val.\ RMSE (last round) \\", r"\midrule"]
    for _, r in fpg.iterrows():
        k = dict(x.split("=") for x in r.param.split(","))
        b, l_ = f"{r.val_rmse_bestround:.4f}", f"{r.val_rmse_lastround:.4f}"
        if r.selected:
            b, l_ = (rf"\textbf{{{b}}}", l_) if a.score == "bestround" else (b, rf"\textbf{{{l_}}}")
        L.append(f"{float(k['TAU']):g} & {float(k['SHRINK']):g} & {b} & {l_} \\\\")
    L += [r"\midrule", r"\multicolumn{4}{@{}l}{Selected configuration of every method (4 configurations each)} \\"]
    for m, g in t.groupby("method"):
        r = g[g.selected].iloc[0]
        L.append(f"{m} & \\multicolumn{{2}}{{l}}{{{r.param.replace('_', ' ')}}} & {r[score]:.4f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "table_equal_budget.tex"), "w").write("\n".join(L))

    # ---- summary
    sel = {m: g[g.selected].iloc[0].param for m, g in t.groupby("method")}
    S = ["# Equal tuning budget (MIT-140, seed 0, validation only)", "",
         f"Score: {score}. FedPhase grid: kappa=1, tau in {{0.5,1}}, lambda in {{0.5,0.75}}, mu = FedProx's.", "",
         "Selected: " + "; ".join(f"{m}: {p}" for m, p in sel.items()), ""]
    if same:
        S += ["**Validation selects the paper configuration (tau=0.5, kappa=1, lambda=0.75) -> the 15-seed results "
              "are unchanged.** Text for 6-Evaluation.tex:", "",
              "> With an equal budget of four configurations per method, fixed a priori "
              "($(\\tau,\\lambda)\\in\\{0.5,1\\}\\times\\{0.5,0.75\\}$, $\\kappa=1$), validation selects the same "
              "configuration, so all results below are equal-budget results."]
    else:
        S += [f"**Validation selects {s.param}, NOT the paper configuration.** Run it on seeds 1-15: "
              "`bash submit_equal_budget.sh run15`, and report that result (do not report the grid before it is run)."]
    if problems:
        S += ["", "## Checks that FAILED", ""] + [f"- {p}" for p in problems]
    else:
        S += ["", "All checks passed: every run has all clients and rounds; FedPhase hyper-parameters verified from "
              "server config.csv."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(S) + "\n")
    print("\n".join(S)); print(f"\n-> {out}")


if __name__ == "__main__":
    main()
