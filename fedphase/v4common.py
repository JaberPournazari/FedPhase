"""Shared helpers for the V4 analysis scripts (run from inside fedphase)."""
import glob
import os
import sys

import matplotlib
matplotlib.use("TkAgg" if (sys.platform.startswith("win") or os.environ.get("DISPLAY"))
               and os.environ.get("MPLBACKEND", "").lower() != "agg" else "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
    "axes.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

ORDER = ["centralized", "local", "fedavg", "fedprox", "fedper", "ditto", "ifca", "fedphase"]
LABELS = {"centralized": "Centralised", "local": "Local", "fedavg": "FedAvg", "fedprox": "FedProx",
          "fedper": "FedPer", "ditto": "Ditto", "ifca": "IFCA", "fedphase": "FedPhase"}
COLORS = {"centralized": "#7f7f7f", "local": "#bcbd22", "fedavg": "#1f77b4", "fedprox": "#ff7f0e",
          "fedper": "#9467bd", "ditto": "#2ca02c", "ifca": "#8c564b", "fedphase": "#d62728"}


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def load_errors(vdir):
    """All error_client*.csv under <vdir>/seed_*  -> long DataFrame with a 'seed' column."""
    rows = []
    for sd in sorted(glob.glob(os.path.join(vdir, "seed_*"))):
        fs = glob.glob(os.path.join(sd, "error_client*.csv"))
        if fs:
            d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
            d["seed"] = int(sd.rsplit("_", 1)[1])
            rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def val_scores(sdir):
    """Seed-dir of one tuning run -> (best-round val RMSE, last-round val RMSE, n_clients, n_rounds).
    best-round = round with the lowest val MSE per client (V2 tuning score); both macro-averaged."""
    fs = glob.glob(os.path.join(sdir, "client_*_rounds.csv"))
    if not fs:
        return np.nan, np.nan, 0, 0
    best, last, nr = [], [], []
    for f in fs:
        d = pd.read_csv(f, usecols=["round", "val_mse", "val_rmse"]).sort_values("round")
        best.append(d.val_rmse.iloc[int(np.nanargmin(d.val_mse.values))])
        last.append(d.val_rmse.iloc[-1])
        nr.append(int(d["round"].iloc[-1]))
    return float(np.mean(best)), float(np.mean(last)), len(fs), min(nr)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); k = len(p)
    h = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * p[o]))
    r = np.empty(k); r[o] = h
    return r


def fmt_p(p, tex=False):
    if not np.isfinite(p):
        return "--"
    return (r"$<$0.001" if tex else "<0.001") if p < 1e-3 else f"{p:.3f}"


def hier_boot_gain(a, b, n_boot, rng):
    """Paired hierarchical bootstrap (seeds, then cells) of 100*(B-A)/B; a, b: (S, C)."""
    S, C = a.shape
    g = np.empty(n_boot)
    for k in range(n_boot):
        si = rng.integers(0, S, S); ci = rng.integers(0, C, C)
        A, B = a[np.ix_(si, ci)].mean(), b[np.ix_(si, ci)].mean()
        g[k] = 100 * (B - A) / B
    return g
