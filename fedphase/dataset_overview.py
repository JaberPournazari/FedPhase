#!/usr/bin/env python
"""Dataset section of the paper (no training): per-cell statistics, capacity curves with the
train/val/test boundaries, feature-availability map (data1) and phase descriptors.

  python dataset_overview.py --dataset data1 --train-frac 0.3 --val-frac 0.1 --seq-len 5
-> results/<dataset>/paper/{table_dataset.csv,.tex, fig_dataset_capacity, fig_dataset_features, fig_dataset_phase}
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

import task

plt.rcParams.update({
    "font.family": "serif", "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
    "axes.linewidth": 1.4, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, out, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data1")
    p.add_argument("--seq-len", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.1)
    a = p.parse_args()
    out = os.path.join("results", a.dataset, "paper")
    os.makedirs(out, exist_ok=True)

    rows, curves = [], []
    for pid in range(task.num_cells()):
        df = task.load_full_cell(pid)
        L, meta = task.load_data(pid, a.seq_len, a.train_frac, a.val_frac)
        y = (df[task.TARGET_COLUMN] / df[task.TARGET_COLUMN].iloc[0]).to_numpy(float)
        c = df["cycle"].to_numpy(float)
        r = {"cell": pid, "rows": len(df), "first_cycle": int(c[0]), "last_cycle": int(c[-1]),
             "cycle_step_median": float(np.median(np.diff(c))),
             "Q0": float(df[task.TARGET_COLUMN].iloc[0]), "SOH_end": float(y[-1]),
             "train_samples": len(L["train"].dataset), "val_samples": len(L["val"].dataset),
             "test_samples": len(L["test"].dataset),
             "first_val_cycle": meta["first_val_cycle"], "first_test_cycle": meta["first_test_cycle"]}
        r.update({"phi_" + k: v for k, v in meta["phase"].items()})
        if task.MASK_COLUMNS:
            r.update({m: int(df[m].iloc[0]) for m in task.MASK_COLUMNS})
        rows.append(r)
        curves.append((pid, c, y, meta["first_val_cycle"], meta["first_test_cycle"]))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out, "table_dataset.csv"), index=False, float_format="%.4f")

    cols = ["cell", "rows", "first_cycle", "last_cycle", "Q0", "SOH_end", "train_samples", "test_samples"]
    tex = t[cols].rename(columns={"first_cycle": "first cyc.", "last_cycle": "last cyc.", "Q0": r"$Q_0$ (Ah)",
                                  "SOH_end": r"SOH$_{\text{end}}$", "train_samples": "train", "test_samples": "test"})
    s = tex.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=False)
    open(os.path.join(out, "table_dataset.tex"), "w").write("\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Cells of {a.dataset}: life span, initial capacity, end-of-test SOH and number of samples "
        rf"(train {int(100 * a.train_frac)}\%, val {int(100 * a.val_frac)}\%, sequence length {a.seq_len}).}}",
        rf"\label{{tab:{a.dataset}_cells}}", r"\resizebox{\linewidth}{!}{%", s.strip() + "}", r"\end{table}"]))

    # capacity curves, coloured by end-of-life SOH, train/val boundary marked
    soh = t.set_index("cell").SOH_end
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(soh.min(), soh.max())
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for pid, c, y, fv, ft in curves:
        col = cmap(norm(soh[pid]))
        ax.plot(c, y, color=col, linewidth=1.8)
        for cyc, mk in [(fv, "o"), (ft, "s")]:
            i = int(np.argmin(np.abs(c - cyc)))
            ax.plot(c[i], y[i], mk, color=col, markersize=6, markeredgecolor="k", markeredgewidth=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax).set_label("SOH at end of life")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Normalised capacity")
    ax.plot([], [], "o", color="w", markeredgecolor="k", label="start of validation")
    ax.plot([], [], "s", color="w", markeredgecolor="k", label="start of test")
    ax.legend(frameon=False, loc="lower left")
    save(fig, out, "fig_dataset_capacity")

    if task.MASK_COLUMNS:
        M = t[task.MASK_COLUMNS].values
        fig, ax = plt.subplots(figsize=(9, 5.6))
        ax.imshow(M, cmap="Greys", aspect="auto", vmin=0, vmax=1.4)
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.cell)
        ax.set_xticks(range(len(task.MASK_COLUMNS)))
        ax.set_xticklabels([m.replace("has_Time", "").replace("_", " ") for m in task.MASK_COLUMNS],
                           rotation=45, ha="right", fontsize=11)
        ax.set_ylabel("Cell (client)")
        ax.set_title(f"Selected features per cell ({M.sum(1).mean():.0f} of {M.shape[1]})", fontsize=14)
        save(fig, out, "fig_dataset_features")

    if "phi_fade_rate_late" in t:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        sc = ax.scatter(t.phi_fade_rate_late, t.phi_soh_end, c=t.SOH_end, cmap="viridis", s=70,
                        edgecolor="k", linewidth=0.5)
        for _, r in t.iterrows():
            ax.annotate(int(r.cell), (r.phi_fade_rate_late, r.phi_soh_end), fontsize=10,
                        xytext=(4, 3), textcoords="offset points")
        fig.colorbar(sc, ax=ax).set_label("SOH at end of life")
        ax.set_xlabel("Fade rate, late train window (per 100 cycles)")
        ax.set_ylabel("SOH at end of train window")
        save(fig, out, "fig_dataset_phase")

    pd.set_option("display.width", 200)
    print(t[cols + ["phi_fade_rate_late", "phi_soh_end"]].to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
