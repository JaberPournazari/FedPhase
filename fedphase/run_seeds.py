#!/usr/bin/env python
"""Single-node launcher: 1 server + N clients on this machine, once per seed.
Used by the smoke test and for small runs without multi-node SLURM.

  python run_seeds.py --method fedphase --seeds 1 --num-partitions 8 --num-rounds 3 \
      --run-tag data2/smoke/fedphase -- --tau-scale 1.0 --mu 0.01

Arguments after `--` go to server_app.py (method hyper-parameters).
Client-side data args (--seq-len/--train-frac/--val-frac) go to both.
"""
import argparse
import os
import socket
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--num-partitions", type=int, required=True)
    p.add_argument("--num-rounds", type=int, default=100)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--port", type=int, default=8080, help="0 = pick a free port (several jobs per node)")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--seq-len", type=int, default=10)
    p.add_argument("--train-frac", type=float, default=0.3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--quiet", action="store_true")
    argv = sys.argv[1:]
    srv_extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, srv_extra = argv[:i], argv[i + 1:]
    a = p.parse_args(argv)
    data = ["--seq-len", str(a.seq_len), "--train-frac", str(a.train_frac), "--val-frac", str(a.val_frac)]

    if a.port == 0:
        with socket.socket() as s_:
            s_.bind(("", 0)); a.port = s_.getsockname()[1]
    print(f"port {a.port}", flush=True)
    for seed in [int(s) for s in a.seeds.split(",")]:
        srv = subprocess.Popen([sys.executable, "server_app.py", "--method", a.method,
                                "--num-rounds", str(a.num_rounds), "--local-epochs", str(a.local_epochs),
                                "--address", f"0.0.0.0:{a.port}", "--num-clients", str(a.num_partitions),
                                "--seed", str(seed), "--run-tag", a.run_tag] + srv_extra)
        rc = subprocess.call([sys.executable, "run_clients_node.py", "--start", "0",
                              "--end", str(a.num_partitions), "--server-address", f"localhost:{a.port}",
                              "--seed", str(seed), "--run-tag", a.run_tag, "--method", a.method]
                             + data + (["--quiet"] if a.quiet else []))
        try:
            srv.wait(timeout=300)
        except subprocess.TimeoutExpired:
            srv.terminate()
        print(f"=== {a.method} seed {seed} done (clients rc={rc}, server rc={srv.returncode}) ===")
        if rc or srv.returncode:
            sys.exit(1)


if __name__ == "__main__":
    main()
