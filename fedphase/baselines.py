#!/usr/bin/env python
"""Non-federated reference points, same model / data / schedule / selection as FL:

  --mode local        each cell trains alone (no sharing)          -> lower bound
  --mode centralized  one model on the pooled train windows        -> upper bound

num_rounds x train(local_epochs) with a fresh Adam per round, evaluate every
round, per-client best round by VAL mse, TEST reported. Output files are
identical to client.py (logio.Recorder).
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from torch.utils.data import ConcatDataset, DataLoader

from logio import Recorder
from task import Net, evaluate, load_data, predict, set_seed, train


def run_local_client(pid, a, log_dir):
    torch.set_num_threads(1)
    set_seed(a.seed * 1000 + pid)
    L, meta = load_data(pid, a.seq_len, a.train_frac, a.val_frac)
    set_seed(a.seed)
    net = Net()                                 # same init as the FL runs of this seed
    rec = Recorder(log_dir, pid, meta)
    t_train = 0.0
    for rnd in range(1, a.num_rounds + 1):
        t0 = time.time()
        tl = train(net, L["train"], a.local_epochs, torch.device("cpu"), a.lr)
        t_train += time.time() - t0
        rec.log_round(rnd, evaluate(net, L, "cpu"), lambda: predict(net, L["all"], "cpu"), tl)
    rec.finalize()
    return pid, t_train


def run_local(a, log_dir):
    t0 = time.time()
    tt = {}
    with ProcessPoolExecutor(max_workers=a.workers or os.cpu_count()) as ex:
        for f in as_completed([ex.submit(run_local_client, pid, a, log_dir) for pid in range(a.num_partitions)]):
            pid, t = f.result()
            tt[pid] = t
    with open(os.path.join(log_dir, "timing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "wall_s", "client_train_sum_s", "client_train_max_s"])
        w.writerow(["local", f"{time.time() - t0:.1f}", f"{sum(tt.values()):.1f}", f"{max(tt.values()):.1f}"])


def run_centralized(a, log_dir):
    torch.set_num_threads(a.workers or os.cpu_count())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [load_data(pid, a.seq_len, a.train_frac, a.val_frac) for pid in range(a.num_partitions)]
    set_seed(a.seed)
    pooled = DataLoader(ConcatDataset([L["train"].dataset for L, _ in data]), batch_size=16, shuffle=True)
    net = Net()
    recs = [Recorder(log_dir, pid, meta) for pid, (_, meta) in enumerate(data)]
    t0, t_train = time.time(), 0.0
    for rnd in range(1, a.num_rounds + 1):
        t1 = time.time()
        tl = train(net, pooled, a.local_epochs, dev, a.lr)
        t_train += time.time() - t1
        for pid, (L, _) in enumerate(data):
            recs[pid].log_round(rnd, evaluate(net, L, dev), lambda L=L: predict(net, L["all"], dev), tl)
        print(f"[centralized] round {rnd}/{a.num_rounds} train_loss={tl:.5f}", flush=True)
    for r in recs:
        r.finalize()
    with open(os.path.join(log_dir, "timing.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "wall_s", "train_s", "device"])
        w.writerow(["centralized", f"{time.time() - t0:.1f}", f"{t_train:.1f}", str(dev)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local", "centralized"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-partitions", type=int, default=140)
    p.add_argument("--num-rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--run-tag", required=True)
    a = p.parse_args()
    log_dir = os.path.join("client_logs", a.run_tag, f"seed_{a.seed}")
    os.makedirs(log_dir, exist_ok=True)
    (run_local if a.mode == "local" else run_centralized)(a, log_dir)


if __name__ == "__main__":
    main()
