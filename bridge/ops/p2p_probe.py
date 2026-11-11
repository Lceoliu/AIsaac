"""Lane-GPU probe (a): page copies between two GPUs of this host (measurement only, no training).

For the lane-GPU design (cpp-actor-driver-split-plan.md, 2b): a lane restore reads pages that live on the actor GPU,
and every lane step's new pages go back to it. This measures, for two visible GPUs (cuda:0 = actor, cuda:1 = lane):
  * whether peer access is available (torch.cuda.can_device_access_peer) and the link (printed by the caller's
    nvidia-smi topo -m);
  * the latency of one copy of N bytes, synchronised, for sizes from 16 KiB to 8 MiB: direct device-to-device
    (tensor.to, which uses P2P when enabled, else stages through the host inside the driver), and explicitly through
    pinned host memory (two copies);
  * the same with the copies queued back to back on one stream (the throughput a batch of restores would see).
Each figure: the median of R repetitions after a warm-up, in microseconds.
  usage: python p2p_probe.py [repetitions=200]"""
import statistics as st
import sys
import time

import torch


def timed(fn, reps):
    for _ in range(10):
        fn()
    torch.cuda.synchronize(0); torch.cuda.synchronize(1)
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        out.append((time.perf_counter()-t)*1e6)
    return st.median(out)


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    assert torch.cuda.device_count() >= 2, 'two visible GPUs needed'
    print('devices:', [torch.cuda.get_device_name(i) for i in range(2)])
    print('peer access 0->1:', torch.cuda.can_device_access_peer(0, 1), ' 1->0:', torch.cuda.can_device_access_peer(1, 0))
    print(f'{"bytes":>10s} {"d2d 0->1 us":>12s} {"d2d 1->0 us":>12s} {"host 0->1 us":>13s} {"GB/s d2d":>9s} {"8 queued d2d us":>16s}')
    for kib in (16, 64, 256, 1024, 4096, 8192):
        n = kib*1024
        a = torch.empty(n, dtype=torch.uint8, device='cuda:0').random_(0, 255)
        b = torch.empty(n, dtype=torch.uint8, device='cuda:1')
        h = torch.empty(n, dtype=torch.uint8).pin_memory()
        d01 = timed(lambda: b.copy_(a, non_blocking=True), reps)
        d10 = timed(lambda: a.copy_(b, non_blocking=True), reps)

        def via_host():
            h.copy_(a, non_blocking=True)
            torch.cuda.current_stream(0).synchronize()
            b.copy_(h, non_blocking=True)
        hst = timed(via_host, reps)

        def queued():
            for _ in range(8):
                b.copy_(a, non_blocking=True)
        q8 = timed(queued, max(20, reps//8))
        assert torch.equal(a.cpu(), b.cpu())
        print(f'{n:10d} {d01:12.1f} {d10:12.1f} {hst:13.1f} {n/d01/1e3:9.2f} {q8:16.1f}')


if __name__ == '__main__':
    main()
