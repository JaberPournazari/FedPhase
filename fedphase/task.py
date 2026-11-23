"""Shared data / model / train / eval code for DDA_FedAvg V3 (data1 + data2).

V3 = V2 + a data adapter for data1 (15 cells, DFL-RUL dataset). Model, training,
split protocol and FedPhase are unchanged, so data1 and data2 numbers use the
same pipeline. The format is detected from DATA_DIR:
  data1: PCA_Selected_Features<i>.csv (5 signal statistics) + CapacityInterpolation<i>.csv (Cycle, Capacity)
  data2: Cell_<i>.csv (MIT-style engineered features, target QD)

data1 specifics
  * Every cell has a DIFFERENT 5-column subset (per-cell PCA selection, 10 distinct
    statistics overall). Feeding them by position (DFL-RUL original) gives input slot k
    a different meaning on every client, which breaks any shared model.
    FEAT_MODE=union (default): fixed 10-column space (D1_SIGNALS) + cycle; absent
    statistics are 0 after scaling (= the train mean) and a raw 0/1 availability mask
    is appended -> 21 inputs.  FEAT_MODE=positional: original behaviour (5 + cycle),
    kept only as a scenario.
  * Cells are 34-441 rows long -> default seq_len 5 for data1 (common.sh).
  * Phase descriptor = capacity-curve part only (no IR / charge-time / temperature).

Changes of V2 vs V1 (kept): last-step output (MSE broadcast bug fixed), temporal
train/val/test split with selection on val, scaler fit on train rows only.
"""
from collections import OrderedDict
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Data layout: <ROOT>/data1 or <ROOT>/data2 (set DATA_DIR); default = ../data/data2 relative to fedphase/.
DATA_DIR = os.environ.get("DATA_DIR", os.environ.get("DATA2_DIR", os.path.join("..", "data", "data2")))
FORMAT = "data1" if glob.glob(os.path.join(DATA_DIR, "PCA_Selected_Features*.csv")) else "data2"
FEAT_MODE = os.environ.get("FEAT_MODE", "union")          # data1 only: union | positional
# data2 only: "full" (27 inputs, as in the paper) | "noqc" (drops the 5 charge-capacity inputs and phi^qc; Q^c sensitivity)
FEATURE_SET = os.environ.get("FEATURE_SET", "full")
if FEATURE_SET not in ("full", "noqc"):
    raise ValueError(f"FEATURE_SET={FEATURE_SET!r}: expected 'full' or 'noqc'")

# data2: same 27 inputs as V1/V2 (QD*, cycle_life, RUL excluded: label leakage).
D2_ALL_FEATURES = [
    "IR", "QC", "Tavg", "Tmin", "Tmax", "chargetime", "cycle",
    "IR_rollmean", "IR_rollstd", "IR_slope", "IR_cummax", "IR_cummin",
    "QC_rollmean", "QC_rollstd", "QC_slope", "QC_cummax",
    "Tavg_rollmean", "Tavg_slope", "Tavg_cummax",
    "Tmin_rollmean", "Tmin_cummax",
    "Tmax_rollmean", "Tmax_cummax",
    "chargetime_rollmean", "chargetime_rollstd", "chargetime_slope", "chargetime_cummax",
]
# data1: union of the per-cell selected statistics (all 15 cells), fixed order.
_P = "Timetable_sigstats_"
D1_SIGNALS = [f"TimeCurrent{_P}{k}" for k in ("SINAD", "SNR", "Kurtosis", "Skewness", "ClearanceFactor", "Std")] + \
             [f"TimeVoltage{_P}{k}" for k in ("SINAD", "SNR", "Kurtosis", "Skewness")]

if FORMAT == "data1":
    TARGET_COLUMN = "Capacity"
    if FEAT_MODE == "positional":
        FEATURE_COLUMNS, MASK_COLUMNS = [f"pc{k}" for k in range(5)] + ["cycle"], []
    else:
        FEATURE_COLUMNS = D1_SIGNALS + ["cycle"]
        MASK_COLUMNS = ["has_" + c.replace(_P, "_") for c in D1_SIGNALS]
else:
    TARGET_COLUMN = "QD"
    QC_COLUMNS = [c for c in D2_ALL_FEATURES if c == "QC" or c.startswith("QC_")]
    FEATURE_COLUMNS = D2_ALL_FEATURES if FEATURE_SET == "full" else [c for c in D2_ALL_FEATURES if c not in QC_COLUMNS]
    MASK_COLUMNS = []
TRIM_COLUMNS = FEATURE_COLUMNS if FORMAT == "data1" else D2_ALL_FEATURES   # row trimming always uses all 27 (data2): same rows in both feature sets
N_FEATURES = len(FEATURE_COLUMNS) + len(MASK_COLUMNS)

# ---------------------------------------------------------------- phase
# Degradation-phase descriptor, computed by each client on its OWN training
# window only (never val/test). "label" entries use the train-window capacity
# curve (known to the client); "feat" entries use input signals only.
PHASE_LABEL = ["fade_rate_early", "fade_rate_late", "fade_accel", "soh_end"]
PHASE_FEAT = ["ir_growth", "ir_slope_late", "ct_change", "tavg_change", "qc_change"]
if FORMAT == "data1":   # no IR / charge-time / temperature / QC signals in data1
    PHASE_SETS = {"full": PHASE_LABEL, "label": PHASE_LABEL}
else:
    PHASE_SETS = {"full": PHASE_LABEL + PHASE_FEAT, "feat": PHASE_FEAT, "label": PHASE_LABEL}
if FORMAT == "data2" and FEATURE_SET == "noqc":
    PHASE_FEAT = [k for k in PHASE_FEAT if k != "qc_change"]
    PHASE_SETS = {"full": PHASE_LABEL + PHASE_FEAT, "feat": PHASE_FEAT, "label": PHASE_LABEL}
PHASE_ALL = PHASE_SETS["full"]

# FedPer: these layers stay on the client (personal head); the LSTM is shared.
HEAD_PREFIXES = ("fc1.", "fc2.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 50)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]                      # last time step -> (B, hidden)
        return self.fc2(self.dropout(self.act(self.fc1(h))))  # (B, 1)


def num_cells(data_dir: str = None) -> int:
    pat = "PCA_Selected_Features*.csv" if FORMAT == "data1" else "Cell_*.csv"
    return len(glob.glob(os.path.join(data_dir or DATA_DIR, pat)))


def _cell_path(pid: int) -> str:
    """Cell_<pid>.csv (data2); otherwise the pid-th file of sorted Cell_*.csv (e.g. data1)."""
    p = os.path.join(DATA_DIR, f"Cell_{pid}.csv")
    if os.path.exists(p):
        return p
    files = sorted(glob.glob(os.path.join(DATA_DIR, "Cell_*.csv")),
                   key=lambda f: int("".join(ch for ch in os.path.basename(f) if ch.isdigit()) or 0))
    return files[pid]


def _spike_clean(s: pd.Series, spike_rel: float = 0.05, neighbor_tol: float = 0.02):
    """Unchanged from V1: replace single-cycle 'spike and return' QD glitches."""
    x = s.to_numpy(dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        l, r = x[i - 1], x[i + 1]
        base = max(abs(l), abs(r), 1e-9)
        if abs(x[i] - l) > spike_rel * base and abs(x[i] - r) > spike_rel * base \
                and abs(l - r) < neighbor_tol * base:
            out[i] = True
    return s.mask(pd.Series(out, index=s.index)).interpolate(limit_direction="both"), out


def _load_data1_cell(pid: int) -> pd.DataFrame:
    """data1: features + capacity (same row count in every cell), mapped to FEATURE_COLUMNS."""
    f = pd.read_csv(os.path.join(DATA_DIR, f"PCA_Selected_Features{pid}.csv")).select_dtypes(include=[np.number])
    t = pd.read_csv(os.path.join(DATA_DIR, f"CapacityInterpolation{pid}.csv"))
    n = min(len(f), len(t))
    f, t = f.iloc[:n].reset_index(drop=True), t.iloc[:n].reset_index(drop=True)
    if FEAT_MODE == "positional":
        df = pd.DataFrame(f.to_numpy(float)[:, :5], columns=FEATURE_COLUMNS[:5])
    else:
        unknown = set(f.columns) - set(D1_SIGNALS)
        if unknown:
            raise ValueError(f"cell {pid}: columns not in D1_SIGNALS: {sorted(unknown)}")
        df = f.reindex(columns=D1_SIGNALS).astype(float)
        for c, m in zip(D1_SIGNALS, MASK_COLUMNS):
            df[m] = float(c in f.columns)
        df[D1_SIGNALS] = df[D1_SIGNALS].fillna(0.0)       # absent statistic -> constant (0 after scaling)
    df["cycle"] = t["Cycle"].to_numpy(float)
    df[TARGET_COLUMN] = t["Capacity"].to_numpy(float)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].ffill().bfill()
    df[TARGET_COLUMN], _ = _spike_clean(df[TARGET_COLUMN])
    return df


def load_full_cell(pid: int) -> pd.DataFrame:
    """Raw cell, trimmed to first row with target != 0 and all features present."""
    if FORMAT == "data1":
        return _load_data1_cell(pid)
    df = pd.read_csv(_cell_path(pid))
    first_qd = df[TARGET_COLUMN].ne(0).idxmax()
    first_feat = df[TRIM_COLUMNS].notna().all(axis=1).idxmax()
    df = df.iloc[max(first_qd, first_feat):].reset_index(drop=True)
    df[TRIM_COLUMNS] = df[TRIM_COLUMNS].ffill().bfill()
    df[TARGET_COLUMN], _ = _spike_clean(df[TARGET_COLUMN])
    return df


def _slope(x, y):
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def phase_descriptor(df: pd.DataFrame, n_train: int) -> dict:
    """Degradation-phase vector from the TRAIN window (rows [0, n_train)).
    Rates are per 100 cycles so cells of different length are comparable."""
    d = df.iloc[:n_train]
    c = d["cycle"].to_numpy(float)
    y = (d[TARGET_COLUMN] / df[TARGET_COLUMN].iloc[0]).to_numpy(float)
    w = max(5, n_train // 3)
    e, l = slice(0, w), slice(n_train - w, n_train)

    def rel(col):
        a, b = d[col].iloc[e].mean(), d[col].iloc[l].mean()
        return float(b / a - 1.0) if abs(a) > 1e-9 else 0.0

    fe, fl = 100 * _slope(c[e], y[e]), 100 * _slope(c[l], y[l])
    phi = {
        "fade_rate_early": fe,
        "fade_rate_late": fl,
        "fade_accel": fl - fe,                     # knee indicator
        "soh_end": float(y[l].mean()),
    }
    if FORMAT == "data1":
        return {k: float(np.nan_to_num(v)) for k, v in phi.items()}
    phi.update({
        "ir_growth": rel("IR"),
        "ir_slope_late": 100 * _slope(c[l], d["IR"].to_numpy(float)[l]),
        "ct_change": rel("chargetime"),
        "tavg_change": float(d["Tavg"].iloc[l].mean() - d["Tavg"].iloc[e].mean()),
        "qc_change": rel("QC"),
    })
    if FEATURE_SET == "noqc":
        phi.pop("qc_change")
    return {k: float(np.nan_to_num(v)) for k, v in phi.items()}


def load_data(pid: int, seq_len: int = 10, train_frac: float = 0.3, val_frac: float = 0.1,
              batch_size: int = 16):
    """Temporal split by TARGET cycle index t:
         train: t <  n_tr,  val: n_tr <= t < n_tr + n_va,  test: rest.
    A sample at t uses input rows [t-seq_len+1, t] (inputs only, never QD)."""
    df = load_full_cell(pid)
    X = df[FEATURE_COLUMNS].to_numpy(float)
    y = (df[TARGET_COLUMN] / df[TARGET_COLUMN].iloc[0]).to_numpy(float)
    cyc = df["cycle"].to_numpy()
    n = len(y)
    n_tr = max(int(train_frac * n), seq_len + 2)
    n_va = max(int(val_frac * n), 5)

    Xs = StandardScaler().fit(X[:n_tr]).transform(X)
    Xs = np.nan_to_num(Xs)
    if MASK_COLUMNS:                                   # data1 union: raw 0/1 availability flags
        Xs = np.hstack([Xs, df[MASK_COLUMNS].to_numpy(float)])
    ts = np.arange(seq_len - 1, n)
    xs = np.stack([Xs[t - seq_len + 1:t + 1] for t in ts]).astype(np.float32)
    ys = y[ts].astype(np.float32)[:, None]
    cs = cyc[ts].astype(np.int64)[:, None]

    def ds(mask):
        return TensorDataset(torch.from_numpy(xs[mask]), torch.from_numpy(ys[mask]),
                             torch.from_numpy(cs[mask]))

    tr, va, te = ts < n_tr, (ts >= n_tr) & (ts < n_tr + n_va), ts >= n_tr + n_va
    loaders = {
        "train": DataLoader(ds(tr), batch_size=batch_size, shuffle=True),
        "train_eval": DataLoader(ds(tr), batch_size=256),
        "val": DataLoader(ds(va), batch_size=256),
        "test": DataLoader(ds(te), batch_size=256),
        "all": DataLoader(ds(np.ones_like(ts, dtype=bool)), batch_size=256),
    }
    meta = {"n_rows": n, "n_train": n_tr, "n_val": n_va, "n_test_samples": int(te.sum()),
            "first_val_cycle": int(cyc[min(n_tr, n - 1)]),
            "first_test_cycle": int(cyc[min(n_tr + n_va, n - 1)]),
            "phase": phase_descriptor(df, n_tr), "format": FORMAT, "feat_mode": FEAT_MODE}
    if FORMAT == "data2" and FEATURE_SET != "full":     # written to error_client*.csv -> read by analysis/compare_noqc.py
        meta.update({"feature_set": FEATURE_SET, "n_features": N_FEATURES})
    return loaders, meta


def train(net, loader, epochs, device, lr=1e-3, anchor=None, mu=0.0):
    """Local training. mu > 0 adds the FedProx/Ditto proximal term
    (mu/2)*||w - anchor||^2. Returns the mean TASK loss."""
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    anc = None
    if anchor is not None and mu > 0:
        anc = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in anchor]
        names = [k for k, _ in net.named_parameters()]
        keys = list(net.state_dict().keys())
        anc = [anc[keys.index(k)] for k in names]   # state_dict order -> parameter order
    tot, nb = 0.0, 0
    for _ in range(epochs):
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            task = loss_fn(net(x), y)
            loss = task
            if anc is not None:
                loss = task + 0.5 * mu * sum(((p - a) ** 2).sum()
                                             for p, a in zip(net.parameters(), anc))
            loss.backward()
            opt.step()
            tot += task.item()
            nb += 1
    return tot / max(nb, 1)


@torch.no_grad()
def predict(net, loader, device):
    net.to(device).eval()
    P, Y, C = [], [], []
    for x, y, c in loader:
        P.append(net(x.to(device)).cpu().numpy().ravel())
        Y.append(y.numpy().ravel())
        C.append(c.numpy().ravel())
    if not P:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(Y), np.concatenate(P), np.concatenate(C)


def metrics(y, p) -> dict:
    if len(y) == 0:
        return {k: np.nan for k in ["mse", "rmse", "mae", "nrmse", "mape", "r2", "maxe"]}
    e = p - y
    mse = float(np.mean(e ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(e))),
            "nrmse": mse ** 0.5 / (float(np.mean(np.abs(y))) + 1e-8),
            "mape": 100 * float(np.mean(np.abs(e) / (np.abs(y) + 1e-8))),
            "r2": 1 - float(np.sum(e ** 2)) / sst if sst > 1e-10 else np.nan,
            "maxe": float(np.max(np.abs(e)))}


def evaluate(net, loaders, device, splits=("val", "test")) -> dict:
    out = {}
    for s in splits:
        y, p, _ = predict(net, loaders[s], device)
        out.update({f"{s}_{k}": v for k, v in metrics(y, p).items()})
    return out


def train_loss(net, loader, device) -> float:
    y, p, _ = predict(net, loader, device)
    return float(np.mean((p - y) ** 2)) if len(y) else np.inf


def get_weights(net):
    # .copy(): .cpu().numpy() aliases the live tensors, so a later set_weights() would silently
    # overwrite any saved weights (this made FedPer's personal head equal the global head).
    return [v.detach().cpu().numpy().copy() for v in net.state_dict().values()]


def set_weights(net, params, mask=None):
    """mask[i] True -> load params[i]; False -> keep current tensor."""
    sd = net.state_dict()
    new = OrderedDict()
    for i, (k, v) in enumerate(sd.items()):
        new[k] = torch.tensor(params[i]) if (mask is None or mask[i]) else v
    net.load_state_dict(new, strict=True)


def body_mask(net):
    """True for shared (LSTM) tensors, False for the FedPer personal head."""
    return [not k.startswith(HEAD_PREFIXES) for k in net.state_dict().keys()]


def n_tensors() -> int:
    return len(Net().state_dict())
