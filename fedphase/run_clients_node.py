"""Run clients [start, end) on THIS node against a (possibly remote) server.
Waits for the server port first. Every unknown argument (--method, --run-tag,
--seq-len, ...) is forwarded unchanged to client.py."""
import argparse
import os
import socket
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--start", type=int, required=True)
p.add_argument("--end", type=int, required=True)
p.add_argument("--server-address", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--run-tag", required=True)
p.add_argument("--quiet", action="store_true")
p.add_argument("--wait", type=int, default=900)
a, fwd = p.parse_known_args()

host, port = a.server_address.rsplit(":", 1)
t0 = time.time()
while True:
    try:
        socket.create_connection((host, int(port)), timeout=3).close()
        break
    except OSError:
        if time.time() - t0 > a.wait:
            sys.exit(f"server {a.server_address} not reachable")
        time.sleep(3)

err_path = os.path.join("client_logs", a.run_tag, f"seed_{a.seed}", f"clients_stderr_{socket.gethostname()}.log")
os.makedirs(os.path.dirname(err_path), exist_ok=True)
err_f = open(err_path, "a")
io = dict(stdout=subprocess.DEVNULL if a.quiet else None, stderr=err_f)
procs = [subprocess.Popen([sys.executable, "client.py", "--partition-id", str(i),
                           "--server-address", a.server_address, "--seed", str(a.seed),
                           "--run-tag", a.run_tag] + fwd, **io)
         for i in range(a.start, a.end)]
bad = [i for i, q in zip(range(a.start, a.end), procs) if q.wait() != 0]
err_f.close()
if bad:
    print(f"[{socket.gethostname()}] clients failed: {bad} (see {err_path})")
    sys.exit(1)
os.remove(err_path)
