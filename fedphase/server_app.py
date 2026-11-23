#!/usr/bin/env python
"""Flower server for V2: FedAvg, FedProx, FedPer, Ditto, IFCA and FedPhase.

FedPhase (proposed) = phase-personalised aggregation + phase-adaptive proximal term
----------------------------------------------------------------------------------
Every client i reports a degradation-phase vector phi_i computed on its own
training window (task.phase_descriptor: capacity-fade rate early/late, fade
acceleration (knee), end-of-window SOH, IR growth, charge-time / temperature /
QC drift). The server z-scores phi across clients and builds a Gaussian
phase-similarity kernel

    K_ij = exp(-||z_i - z_j||^2 / (2 h^2)),   h = tau_scale * median_{i!=j} ||z_i - z_j||

and sends EACH client its own model

    theta~_i = sum_j a_ij theta_j,     a_ij = n_j K_ij / sum_l n_l K_il

(optionally restricted to the knn most similar clients). Client i then trains
theta~_i with a proximal pull (mu_i/2)||w - theta~_i||^2 where

    mu_i = mu * clip(rho_i / mean(rho), 0.25, 4),   rho_i = mean_{j!=i} K_ij

i.e. clients in a dense phase neighbourhood are kept close to it, clients
whose phase is rare are allowed to adapt more (--adaptive-mu 0 -> mu_i = mu).

Shrinkage (--shrink l): a_ij = (1-l) n_j/sum(n) + l a_ij^phase  (l=0 -> FedProx, l=1 -> pure kernel).
Look-ahead (--lookahead s > 0): K_ij is multiplied by sigmoid((soh_i - soh_j)/(s*std(soh))),
where soh = end-of-train-window capacity. Test windows lie in each cell's FUTURE, so cells that are
already further along in degradation are the informative teachers. s -> inf recovers the symmetric kernel.

Special cases (used as sanity checks / ablations):
    tau_scale -> inf, adaptive_mu=0  => FedProx(mu)
    tau_scale -> inf, mu = 0         => FedAvg
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List

import flwr as fl
import numpy as np
from flwr.common import EvaluateIns, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate as weighted_avg

from task import PHASE_SETS, Net, get_weights, n_tensors, set_seed


def _mb(params) -> float:
    return sum(len(t) for t in params.tensors) / 1e6


# ======================================================================
# Base: FedAvg + per-client instructions + full timing breakdown (as V1)
# ======================================================================
class Base(fl.server.strategy.FedAvg):
    def __init__(self, *a, log_dir=".", hp=None, **kw):
        super().__init__(*a, **kw)
        self.log_dir, self.hp, self._info = log_dir, hp or {}, {}
        os.makedirs(log_dir, exist_ok=True)

    # --- per-client hooks (overridden by FedPhase) ---
    def params_for(self, client, parameters, stage):
        return parameters

    def config_for(self, client) -> Dict:
        return {}

    def _base_cfg(self, rnd):
        return {"server_round": rnd, **self.hp}

    def configure_fit(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=self.min_fit_clients,
                                        min_num_clients=self.min_available_clients)
        cfg = self._base_cfg(server_round)
        ins = []
        for c in clients:
            ins.append((c, FitIns(self.params_for(c, parameters, "fit"), {**cfg, **self.config_for(c)})))
        self._mb_down = sum(_mb(i.parameters) for _, i in ins)
        self._t_fit_send = time.time()
        return ins

    def configure_evaluate(self, server_round, parameters, client_manager):
        clients = client_manager.sample(num_clients=self.min_evaluate_clients,
                                        min_num_clients=self.min_available_clients)
        cfg = self._base_cfg(server_round)
        ins = [(c, EvaluateIns(self.params_for(c, parameters, "eval"), cfg)) for c in clients]
        self._t_eval_send = time.time()
        return ins

    def _aggregate(self, rnd, results, failures):
        return fl.server.strategy.FedAvg.aggregate_fit(self, rnd, results, failures)

    def aggregate_fit(self, server_round, results, failures):
        rnd = server_round
        t_recv = time.time()
        ta = time.time()
        out = self._aggregate(rnd, results, failures)
        agg_s = time.time() - ta
        starts, ends, rows = [], [], []
        for _, r in results:
            m = r.metrics
            st, en = float(m["t_fit_start"]), float(m["t_fit_end"])
            starts.append(st); ends.append(en)
            rows.append([rnd, int(m["partition_id"]), m.get("host", ""), st - self._t_fit_send,
                         en - st, en])
        max_end = max(ends) if ends else t_recv
        comp = [e - s for s, e in zip(starts, ends)] or [0.0]
        cpath = os.path.join(self.log_dir, "client_times.csv")
        new = not os.path.exists(cpath)
        with open(cpath, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["round", "partition", "host", "downlink_s", "compute_s", "straggler_wait_s"])
            for r_, pid, host, dl, cs, en in rows:
                w.writerow([r_, pid, host, f"{dl:.4f}", f"{cs:.4f}", f"{max_end - en:.4f}"])
        fit_wall = t_recv - self._t_fit_send
        self._info[rnd] = {
            "n_clients": len(results), "n_failures": len(failures), "fit_wall_s": fit_wall,
            "agg_ms": 1000 * agg_s, "compute_mean_s": float(np.mean(comp)),
            "compute_max_s": max(comp), "overhead_s": fit_wall - max(comp),
            "uplink_last_s": t_recv - max_end,
            "MB_down": self._mb_down,
            "MB_up": sum(_mb(r.parameters) for _, r in results),
        }
        print(f"[Round {rnd}] fit_wall={fit_wall:.2f}s agg={1000 * agg_s:.1f}ms "
              f"failures={len(failures)}", flush=True)
        return out

    def aggregate_evaluate(self, server_round, results, failures):
        rnd = server_round
        info = self._info.pop(rnd, None)
        if info is not None:
            t = time.time()
            info["eval_wall_s"] = t - self._t_eval_send
            info["round_wall_s"] = t - self._t_fit_send
            v = [r.metrics.get("val_rmse", np.nan) for _, r in results]
            te = [r.metrics.get("test_rmse", np.nan) for _, r in results]
            info["mean_val_rmse"], info["mean_test_rmse"] = float(np.nanmean(v)), float(np.nanmean(te))
            path = os.path.join(self.log_dir, "round_times.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                keys = list(info.keys())
                if new:
                    w.writerow(["round"] + keys)
                w.writerow([rnd] + [f"{info[k]:.5g}" if isinstance(info[k], float) else info[k]
                                    for k in keys])
            print(f"[Round {rnd}] mean val RMSE={info['mean_val_rmse']:.4f} "
                  f"test RMSE={info['mean_test_rmse']:.4f}", flush=True)
        if not results:
            return None, {}
        loss = sum(r.num_examples * r.loss for _, r in results) / sum(r.num_examples for _, r in results)
        return float(loss), {}


class FedAvgS(Base):      # also used by FedProx (mu in config), FedPer, Ditto
    pass


# ======================================================================
# IFCA: K cluster models, concatenated into one parameter list
# ======================================================================
class IFCAS(Base):
    def __init__(self, *a, k=3, **kw):
        super().__init__(*a, **kw)
        self.k, self.t = k, n_tensors()

    def _aggregate(self, rnd, results, failures):
        models = parameters_to_ndarrays(self.current)
        models = [models[i * self.t:(i + 1) * self.t] for i in range(self.k)]
        groups = {c: [] for c in range(self.k)}
        assign = []
        for _, r in results:
            c = int(r.metrics["cluster"])
            groups[c].append((parameters_to_ndarrays(r.parameters), r.num_examples))
            assign.append((int(r.metrics["partition_id"]), c))
        for c, g in groups.items():
            if g:
                models[c] = weighted_avg(g)
        path = os.path.join(self.log_dir, "ifca_assign.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["round", "partition", "cluster"])
            w.writerows([[rnd, p, c] for p, c in sorted(assign)])
        self.current = ndarrays_to_parameters([x for m in models for x in m])
        return self.current, {}


# ======================================================================
# FedPhase (proposed)
# ======================================================================
class FedPhaseS(Base):
    def __init__(self, *a, mu=0.01, tau_scale=1.0, knn=0, adaptive_mu=True,
                 phase_set="full", lookahead=0.0, shrink=1.0, **kw):
        super().__init__(*a, **kw)
        self.mu, self.tau, self.knn, self.adaptive = mu, tau_scale, knn, adaptive_mu
        self.lookahead = lookahead      # >0: asymmetric "look-ahead" kernel (see module docstring)
        self.shrink = shrink            # lambda: a = (1-lambda)*FedAvg weights + lambda*phase kernel weights
        self.keys = PHASE_SETS[phase_set]
        self.personal: Dict[str, object] = {}
        self.mu_i: Dict[str, float] = {}
        self._saved_phi = False

    def params_for(self, client, parameters, stage):
        return self.personal.get(client.cid, parameters)

    def config_for(self, client):
        return {"mu": float(self.mu_i.get(client.cid, self.mu))}

    def kernel(self, Phi):
        Z = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-8)
        D2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        N = len(Z)
        off = np.sqrt(D2[~np.eye(N, dtype=bool)]) if N > 1 else np.array([1.0])
        h = self.tau * max(float(np.median(off)), 1e-8)
        K = np.exp(-D2 / (2 * h * h))
        if self.knn and self.knn < N:
            keep = np.zeros_like(K, dtype=bool)
            idx = np.argsort(-K, axis=1)[:, :self.knn + 1]      # +1: self
            np.put_along_axis(keep, idx, True, axis=1)
            K = np.where(keep, K, 0.0)
        return K, Z

    def _aggregate(self, rnd, results, failures):
        cids = [p.cid for p, _ in results]
        pids = np.array([int(r.metrics["partition_id"]) for _, r in results])
        n = np.array([r.num_examples for _, r in results], dtype=float)
        Phi = np.array([[float(r.metrics["phi_" + k]) for k in self.keys] for _, r in results])
        nds = [parameters_to_ndarrays(r.parameters) for _, r in results]
        shapes = [x.shape for x in nds[0]]
        sizes = [x.size for x in nds[0]]
        Theta = np.stack([np.concatenate([x.ravel() for x in nd]) for nd in nds])   # (N, P)

        K, Z = self.kernel(Phi)
        if self.lookahead > 0:
            # Look-ahead: test windows lie in the FUTURE of each cell, so the useful teachers are
            # cells that are already further along in degradation (lower end-of-window SOH).
            # K_ij *= sigmoid((soh_i - soh_j) / s),  s = lookahead * std(soh)
            soh = np.array([float(r.metrics["phi_soh_end"]) for _, r in results])
            s_ = self.lookahead * (soh.std() + 1e-8)
            K = K / (1.0 + np.exp(-(soh[:, None] - soh[None, :]) / s_))
        A = K * n[None, :]
        A /= A.sum(1, keepdims=True)
        if self.shrink < 1.0:           # shrink toward the global (data-size) average: lambda=0 -> FedProx
            A = (1.0 - self.shrink) * (n / n.sum())[None, :] + self.shrink * A
        Tp = A @ Theta                                                          # (N, P)
        for i, cid in enumerate(cids):
            parts, o = [], 0
            for s, sz in zip(shapes, sizes):
                parts.append(Tp[i, o:o + sz].reshape(s).astype(np.float32)); o += sz
            self.personal[cid] = ndarrays_to_parameters(parts)

        N = len(cids)
        rho = (K.sum(1) - np.diag(K)) / max(N - 1, 1)
        scale = np.clip(rho / (rho.mean() + 1e-12), 0.25, 4.0) if self.adaptive else np.ones(N)
        for cid, s in zip(cids, scale):
            self.mu_i[cid] = self.mu * float(s)

        # --- logs for the paper (phase map, weight matrix, mu_i) ---
        order = np.argsort(pids)
        if not self._saved_phi:
            import pandas as pd
            pd.DataFrame(Phi[order], columns=self.keys).assign(partition=pids[order]).to_csv(
                os.path.join(self.log_dir, "phase_descriptors.csv"), index=False)
            self._saved_phi = True
        np.save(os.path.join(self.log_dir, "alpha.npy"), A[np.ix_(order, order)])
        with open(os.path.join(self.log_dir, "mu_i.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["partition", "mu_i", "rho_i", "self_weight", "eff_neighbours"])
            for i in order:
                w.writerow([pids[i], f"{self.mu * scale[i]:.6g}", f"{rho[i]:.4g}",
                            f"{A[i, i]:.4g}", f"{1.0 / np.sum(A[i] ** 2):.2f}"])

        glob = weighted_avg([(nd, w) for nd, w in zip(nds, n)])  # reference global model
        return ndarrays_to_parameters(glob), {}


# ======================================================================
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["fedavg", "fedprox", "fedper", "ditto", "ifca", "fedphase"])
    p.add_argument("--num-rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--address", default="0.0.0.0:8080")
    p.add_argument("--num-clients", type=int, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-tag", required=True)
    p.add_argument("--mu", type=float, default=0.01, help="fedprox / fedphase proximal mu")
    p.add_argument("--ditto-lambda", type=float, default=0.1)
    p.add_argument("--ifca-k", type=int, default=3)
    p.add_argument("--tau-scale", type=float, default=1.0, help="fedphase kernel bandwidth scale")
    p.add_argument("--knn", type=int, default=0, help="fedphase: keep k nearest phases (0=all)")
    p.add_argument("--adaptive-mu", type=int, default=1)
    p.add_argument("--lookahead", type=float, default=0.0,
                   help="fedphase: >0 enables asymmetric look-ahead kernel, value = sigmoid scale in std(SOH) units")
    p.add_argument("--shrink", type=float, default=1.0,
                   help="fedphase: lambda in [0,1]; a=(1-l)*uniform(data-size)+l*phase kernel (1=pure kernel, 0=FedProx)")
    p.add_argument("--phase-set", default="full", choices=list(PHASE_SETS))
    a = p.parse_args()

    set_seed(a.seed)
    log_dir = os.path.join("server_logs", a.run_tag, f"seed_{a.seed}")
    for f in ("round_times.csv", "client_times.csv", "ifca_assign.csv"):
        if os.path.exists(os.path.join(log_dir, f)):
            os.remove(os.path.join(log_dir, f))

    hp = {"local_epochs": a.local_epochs, "lr": a.lr,
          "mu": a.mu if a.method in ("fedprox", "fedphase") else 0.0,
          "ditto_lambda": a.ditto_lambda, "ifca_k": a.ifca_k}
    init = get_weights(Net())                      # identical init for every method (same seed)
    common = dict(min_fit_clients=a.num_clients, min_evaluate_clients=a.num_clients,
                  min_available_clients=a.num_clients, fraction_fit=1.0, fraction_evaluate=1.0,
                  log_dir=log_dir, hp=hp)
    if a.method == "ifca":
        inits = []
        for k in range(a.ifca_k):
            set_seed(a.seed * 100 + k)
            inits += get_weights(Net())
        s = IFCAS(k=a.ifca_k, initial_parameters=ndarrays_to_parameters(inits), **common)
        s.current = ndarrays_to_parameters(inits)
    elif a.method == "fedphase":
        s = FedPhaseS(mu=a.mu, tau_scale=a.tau_scale, knn=a.knn, adaptive_mu=bool(a.adaptive_mu),
                      phase_set=a.phase_set, lookahead=a.lookahead, shrink=a.shrink,
                      initial_parameters=ndarrays_to_parameters(init), **common)
    else:
        s = FedAvgS(initial_parameters=ndarrays_to_parameters(init), **common)

    with open(os.path.join(log_dir, "config.csv"), "w", newline="") as f:
        csv.writer(f).writerows([[k, v] for k, v in vars(a).items()])
    t0 = time.time()
    fl.server.start_server(server_address=a.address,
                           config=fl.server.ServerConfig(num_rounds=a.num_rounds), strategy=s)
    with open(os.path.join(log_dir, "run_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "num_rounds", "num_clients", "total_wall_s"])
        w.writerow([a.method, a.num_rounds, a.num_clients, f"{time.time() - t0:.2f}"])


if __name__ == "__main__":
    main()
