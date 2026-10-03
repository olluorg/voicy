"""Run a command; kill its process tree if it goes over LIMIT_MB of RAM.
Reports the peak and wall time. WSL here has ~16 GB, and running out takes
down every session on the machine, not just the offending job.

Counted is anonymous and shared memory (RssAnon + RssShmem), not VmRSS:
weights read through mmap show up in VmRSS as file pages the kernel can drop
at will; they do not bring WSL down.

    LIMIT_MB=11000 python scripts/rss_watch.py COMMAND ...
"""
import os
import subprocess
import sys
import time

LIMIT_MB = int(os.environ.get("LIMIT_MB", "6000"))


def tree_rss(pid: int) -> int:
    pids, rss = [pid], 0
    while pids:
        p = pids.pop()
        try:
            with open(f"/proc/{p}/status") as f:
                for line in f:
                    if line.startswith(("RssAnon:", "RssShmem:")):
                        rss += int(line.split()[1])
            with open(f"/proc/{p}/task/{p}/children") as f:
                pids += [int(c) for c in f.read().split()]
        except OSError:
            pass
    return rss // 1024


start = time.monotonic()
proc = subprocess.Popen(sys.argv[1:], start_new_session=True)
peak = 0
while proc.poll() is None:
    peak = max(peak, tree_rss(proc.pid))
    if peak > LIMIT_MB:
        os.killpg(proc.pid, 9)
        proc.wait()
        print(f"watch: killed at {peak} MB (limit {LIMIT_MB})", file=sys.stderr)
        sys.exit(137)
    time.sleep(0.2)
print(f"watch: exit {proc.returncode}, wall {time.monotonic() - start:.1f} s, peak RSS {peak} MB", file=sys.stderr)
sys.exit(proc.returncode)
