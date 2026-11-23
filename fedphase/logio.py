"""Per-client result recorder shared by client.py and baselines.py, so every
method (FL, local, centralized) writes identical files:

  <log_dir>/client_<pid>_rounds.csv   one row per round: val_* and test_* metrics
  <log_dir>/error_client<pid>.csv     one row: TEST metrics at the round with the
                                      lowest VAL mse (model selection on val only)
                                      + final-round test (final_*) and val (final_val_*) metrics + phase vector
  <log_dir>/prediction_client<pid>.csv  full life at the selected round:
                                      cycle, true, pred, split(train/val/test)
"""
import csv
import os

import numpy as np
import pandas as pd

MKEYS = ["mse", "rmse", "mae", "nrmse", "mape", "r2", "maxe"]
ROUND_COLS = ["round"] + [f"{s}_{k}" for s in ("val", "test") for k in MKEYS] + ["train_loss", "extra"]


class Recorder:
    def __init__(self, log_dir, pid, meta):
        self.dir, self.pid, self.meta = log_dir, pid, meta
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"client_{pid}_rounds.csv")
        if os.path.exists(self.path):
            os.remove(self.path)
        self.best, self.best_pred, self.last = None, None, None

    def log_round(self, rnd, m, predict_all, train_loss=np.nan, extra=""):
        row = {"round": rnd, **{k: m.get(k, np.nan) for k in ROUND_COLS[1:-2]},
               "train_loss": train_loss, "extra": extra}
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ROUND_COLS)
            if new:
                w.writeheader()
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
        self.last = row
        v = m.get("val_mse", np.nan)
        if self.best is None or (np.isfinite(v) and v < self.best["val_mse"]):
            self.best = row
            self.best_pred = predict_all()

    def finalize(self):
        if self.best is None:
            return
        b, l = self.best, self.last
        out = {"partition": self.pid, "best_round": b["round"], "last_round": l["round"]}
        out.update({k.replace("test_", ""): b[k] for k in b if k.startswith("test_")})
        out.update({k: b[k] for k in b if k.startswith("val_")})
        out.update({"final_" + k.replace("test_", ""): l[k] for k in l if k.startswith("test_")})
        out.update({"final_" + k: l[k] for k in l if k.startswith("val_")})   # V3: last-round val (tuning)
        out.update({k: v for k, v in self.meta.items() if k != "phase"})
        out.update({"phi_" + k: v for k, v in self.meta.get("phase", {}).items()})
        pd.DataFrame([out]).to_csv(os.path.join(self.dir, f"error_client{self.pid}.csv"), index=False)

        y, p, c = self.best_pred
        fv, ft = self.meta["first_val_cycle"], self.meta["first_test_cycle"]
        split = np.where(c < fv, "train", np.where(c < ft, "val", "test"))
        pd.DataFrame({"cycle": c, "true": y, "pred": p, "split": split}).to_csv(
            os.path.join(self.dir, f"prediction_client{self.pid}.csv"), index=False, float_format="%.6g")
