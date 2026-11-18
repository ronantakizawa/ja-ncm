"""Fail fast if the node's GPUs cannot communicate (run under torchrun before training).

Does a 256 MB all-reduce on every rank and checks the result, so NCCL/NVLink/driver problems
(like the NVLS multicast bind failure seen on an 8x H100 node) surface in seconds instead of
after a stage has been scheduled.
"""
import os
import time

import torch
import torch.distributed as dist

rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
local = int(os.environ.get("LOCAL_RANK", "0"))
torch.cuda.set_device(local)
dist.init_process_group("nccl")
x = torch.full((64 * 1024 * 1024,), float(rank + 1), device="cuda")
dist.all_reduce(x)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(5):
    dist.all_reduce(x)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 5
want = world * (world + 1) / 2 * world ** 5
ok = torch.allclose(x[:8], torch.full((8,), want, device="cuda"), rtol=1e-3)
gbps = 2 * (world - 1) / world * x.numel() * 4 / dt / 1e9
if rank == 0:
    print(f"[nccl] world={world} all-reduce 256MB: {dt * 1000:.1f} ms, bus {gbps:.0f} GB/s, correct={ok}", flush=True)
dist.destroy_process_group()
raise SystemExit(0 if ok else 1)
