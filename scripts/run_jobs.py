"""Run shell commands from a file, one per GPU slot (CUDA_VISIBLE_DEVICES set per job).

  python scripts/run_jobs.py jobs.txt --gpus 8 --log-dir logs/v3_eval
Lines starting with '#' are skipped. A job whose --out file already exists is skipped, so a
relaunch resumes. Exit code 1 if any job failed (all jobs still run).
"""
import argparse
import os
import re
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--log-dir", default="logs/jobs")
    a = ap.parse_args()
    jobs = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    os.makedirs(a.log_dir, exist_ok=True)
    todo = []
    for i, j in enumerate(jobs):
        m = re.search(r"--out\s+(\S+)", j)
        if m and os.path.exists(m.group(1)):
            continue
        todo.append((i, j))
    print(f"[jobs] {len(todo)} of {len(jobs)} to run on {a.gpus} GPUs", flush=True)
    free, running, failed = list(range(a.gpus)), {}, 0
    while todo or running:
        while todo and free:
            i, j = todo.pop(0)
            g = free.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
            log = open(os.path.join(a.log_dir, f"job{i:03d}.log"), "w")
            running[subprocess.Popen(j, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT)] = (g, i, j, log)
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            g, i, j, log = running.pop(p)
            log.close()
            free.append(g)
            if p.returncode:
                failed += 1
                print(f"[jobs] FAILED job{i:03d} (rc {p.returncode}): {j[:160]}", flush=True)
    print(f"[jobs] done, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
