"""Flower client for all FL methods of V2.

  --method fedavg    plain local training on the received global model
  --method fedprox   + (mu/2)||w - w_global||^2                    (Li et al. 2020)
  --method fedper    LSTM shared, fc head kept on the client       (Arivazhagan et al. 2019)
  --method ditto     global model + personal model v_i pulled to it (Li et al. 2021)
  --method ifca      K cluster models, client picks lowest-loss one (Ghosh et al. 2020)
  --method fedphase  receives a PHASE-PERSONALISED model from the server and
                     trains it with a phase-adaptive proximal term mu_i (ours)

Hyper-parameters (local_epochs, lr, mu, ditto_lambda, ifca_k) come from the
server's round config, so a run has ONE source of truth (server_app.py args).
"""
import argparse
import copy
import os
import socket
import time

import flwr as fl
import numpy as np
import torch

from logio import Recorder
from task import (Net, body_mask, evaluate, get_weights, load_data, n_tensors, predict,
                  set_seed, set_weights, train, train_loss)


class FLClient(fl.client.NumPyClient):
    def __init__(self, method, pid, loaders, meta, rec):
        self.method, self.pid, self.L, self.meta, self.rec = method, pid, loaders, meta, rec
        self.dev = torch.device("cpu")
        self.net = Net()
        self.mask = body_mask(self.net)
        self.personal = None          # ditto: personal model; fedper: own head (full weights)
        self.cluster = -1             # ifca
        self.n_train = len(loaders["train"].dataset)

    # ------------------------------------------------------------ helpers
    def _ifca_models(self, parameters, k):
        t = n_tensors()
        return [parameters[i * t:(i + 1) * t] for i in range(k)]

    def _ifca_pick(self, models):
        losses = []
        for m in models:
            set_weights(self.net, m)
            losses.append(train_loss(self.net, self.L["train_eval"], self.dev))
        return int(np.argmin(losses))

    def _load_for_eval(self, parameters, cfg):
        """Put into self.net the model this method would deploy on this client."""
        if self.method == "ditto":
            if self.personal is None:
                set_weights(self.net, parameters)
            else:
                self.net.load_state_dict(self.personal)
        elif self.method == "fedper":
            set_weights(self.net, parameters)
            if self.personal is not None:
                set_weights(self.net, self.personal, mask=[not b for b in self.mask])
        elif self.method == "ifca":
            models = self._ifca_models(parameters, int(cfg["ifca_k"]))
            self.cluster = self._ifca_pick(models)
            set_weights(self.net, models[self.cluster])
        else:  # fedavg / fedprox / fedphase: the received (global or personalised) model
            set_weights(self.net, parameters)

    # ---------------------------------------------------------------- fit
    def fit(self, parameters, cfg):
        t_start = time.time()
        E, lr = int(cfg["local_epochs"]), float(cfg["lr"])
        mu = float(cfg.get("mu", 0.0))
        extra = {}
        if self.method in ("fedavg", "fedprox", "fedphase"):
            set_weights(self.net, parameters)
            loss = train(self.net, self.L["train"], E, self.dev, lr,
                         anchor=parameters if mu > 0 else None, mu=mu)
            out = get_weights(self.net)
        elif self.method == "fedper":
            set_weights(self.net, parameters)
            if self.personal is not None:
                set_weights(self.net, self.personal, mask=[not b for b in self.mask])
            loss = train(self.net, self.L["train"], E, self.dev, lr)
            out = get_weights(self.net)
            self.personal = out
        elif self.method == "ditto":
            lam = float(cfg["ditto_lambda"])
            set_weights(self.net, parameters)                  # 1) global update
            loss = train(self.net, self.L["train"], E, self.dev, lr)
            out = get_weights(self.net)
            pnet = Net()                                         # 2) personal update
            if self.personal is None:
                set_weights(pnet, parameters)
            else:
                pnet.load_state_dict(self.personal)
            train(pnet, self.L["train"], E, self.dev, lr, anchor=parameters, mu=lam)
            self.personal = copy.deepcopy(pnet.state_dict())
        elif self.method == "ifca":
            models = self._ifca_models(parameters, int(cfg["ifca_k"]))
            self.cluster = self._ifca_pick(models)
            set_weights(self.net, models[self.cluster])
            loss = train(self.net, self.L["train"], E, self.dev, lr)
            out = get_weights(self.net)
            extra["cluster"] = self.cluster
        else:
            raise ValueError(self.method)

        m = {"partition_id": self.pid, "train_loss": float(loss), "host": socket.gethostname(),
             "train_time": time.time() - t_start, "mu_used": mu,
             "t_fit_start": t_start, "t_fit_end": time.time(), **extra}
        m.update({"phi_" + k: v for k, v in self.meta["phase"].items()})
        return out, self.n_train, m

    # ----------------------------------------------------------- evaluate
    def evaluate(self, parameters, cfg):
        rnd = int(cfg.get("server_round", 0))
        self._load_for_eval(parameters, cfg)
        m = evaluate(self.net, self.L, self.dev)
        net_snapshot = copy.deepcopy(self.net)
        self.rec.log_round(rnd, m, lambda: predict(net_snapshot, self.L["all"], self.dev),
                           extra=f"cluster={self.cluster}" if self.method == "ifca" else "")
        n_val = len(self.L["val"].dataset)
        return float(m["val_mse"]), n_val, {"val_rmse": float(m["val_rmse"]),
                                            "test_rmse": float(m["test_rmse"]),
                                            "partition_id": self.pid}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["fedavg", "fedprox", "fedper", "ditto", "ifca", "fedphase"])
    p.add_argument("--partition-id", type=int, required=True)
    p.add_argument("--server-address", default="localhost:8080")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run-tag", required=True, help="e.g. data2/main/fedphase")
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.1)
    a, _ = p.parse_known_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    set_seed(a.seed * 1000 + a.partition_id)   # distinct but reproducible shuffling per client
    log_dir = os.path.join("client_logs", a.run_tag, f"seed_{a.seed}")
    loaders, meta = load_data(a.partition_id, a.seq_len, a.train_frac, a.val_frac)
    rec = Recorder(log_dir, a.partition_id, meta)
    client = FLClient(a.method, a.partition_id, loaders, meta, rec)
    try:
        fl.client.start_client(server_address=a.server_address, client=client.to_client())
    except (AttributeError, TypeError):
        fl.client.start_numpy_client(server_address=a.server_address, client=client)
    rec.finalize()


if __name__ == "__main__":
    main()
