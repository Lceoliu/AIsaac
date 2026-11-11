"""B62 / BM1 decision 1, step 1: is capture_error_mode='thread_local' safe for our captures -- no error AND no silent
corruption -- while other threads of the process make CUDA calls, on the capture's GPU or on another one?

  cd bridge/python && python ../ops/b62_capture_probe2.py [--mode thread_local|global] [--reps 5] [--replays 128]
                                                          [case ...]          (two visible GPUs for the dev1 cases)

Each (case, repetition) runs in its own subprocess. In it, the main thread first captures a reference graph quietly
(no other thread active), then captures the same body again while a side thread acts during the capture; then both
graphs are replayed `replays` times on the same inputs (4 fixed inputs in turn) and every output compared bit for
bit. The body allocates intermediates from the graph's private pool (a matmul, an activation, a reduction, a
division), so a pool shared or corrupted by the side thread's work would show. A case passes when the capture ends,
the side call ends, every replay of the interfered graph equals the quiet one bit for bit, and (two-capture cases)
the side thread's own graph equals its own quiet reference too, with both graphs' pools distinct.

Side actions (on cuda:0, the capture's GPU, or with the dev1_ prefix on cuda:1):
  query, sync           Event.query / Event.synchronize of an event recorded before
  stream_sync           Stream.synchronize of another idle stream
  event_destroy         the last reference to a recorded torch.cuda.Event dropped (cudaEventDestroy)
  pin_alloc             a new pinned host buffer of a size not cached (cudaHostAlloc)
  pin_free_used         the last reference to a pinned buffer that served a non_blocking copy dropped
  dev_alloc             a new 64 MiB device tensor on a side stream (the caching allocator: a new segment)
  replay                a graph captured before, replayed on a side stream
  h2d                   a non_blocking pinned -> device copy on a side stream
  staging_acquire       block_learner.Staging().acquire of a larger size than its one buffer (the learner prefetch
                        worker's real call: event query, the wait for the last copy, the free, the pinned allocation),
                        SIDE_CUDA gate off
  peer_copy             (dev1 only) a cuda:1 -> cuda:0 copy on a cuda:1 stream (take_weights' peer copy) -- into a
                        tensor on the capture's GPU outside the capture
  job_events            a learner-job-like dict holding CUDA events of recorded copies: queried, synchronised and
                        dropped (learner_loop's jobs.get with split devices)
  capture               the side thread captures a graph of its own at the same time (same GPU, another stream; or
                        dev1_: on cuda:1), in the same mode
Prints one JSON line per (case, repetition) and a summary line per case."""
import json
import os
import subprocess
import sys
import threading


CASES = ('none', 'query', 'sync', 'stream_sync', 'event_destroy', 'pin_alloc', 'pin_free_used', 'dev_alloc', 'replay',
         'h2d', 'staging_acquire', 'job_events', 'capture',
         'dev1_query', 'dev1_sync', 'dev1_stream_sync', 'dev1_event_destroy', 'dev1_pin_alloc', 'dev1_dev_alloc',
         'dev1_replay', 'dev1_h2d', 'dev1_peer_copy', 'dev1_job_events', 'dev1_capture')


def _err(e):
    return (type(e).__name__+': '+' | '.join(x.strip() for x in str(e).splitlines() if x.strip()))[:400]


def child(case, mode, replays):
    import torch
    sys.path.insert(0, os.getcwd())
    from isaac_bridge import block_training
    block_training.SideCuda.ENABLED = False
    dev1 = case.startswith('dev1_')
    base = case[5:] if dev1 else case
    sdev = 1 if dev1 else 0
    res = dict(case=case, mode=mode)
    g = torch.Generator(device='cpu').manual_seed(7)
    W = torch.randn(512, 512, generator=g).cuda(0)
    ins = [torch.randn(256, 512, generator=g).cuda(0) for _ in range(4)]

    def body(x):
        a = x @ W
        b = torch.relu(a)*0.5+x
        c = b.sum(dim=1, keepdim=True)
        return b/(c.abs()+1.0)

    def capture(dev, x, stream):
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.device(dev), torch.cuda.stream(stream):
            for _ in range(2):   # (warm-up on the side stream, as torch.cuda.graph does)
                body(x)
            torch.cuda.current_stream().synchronize()
            gr.capture_begin(capture_error_mode=mode)
            try:
                y = body(x)
            finally:
                gr.capture_end()
        return gr, y

    # the quiet reference (nothing else running)
    x_ref = ins[0].clone()
    s_ref = torch.cuda.Stream(device=0)
    g_ref, y_ref = capture(0, x_ref, s_ref)
    torch.cuda.synchronize(0)
    # side material, made before the interfered capture, on the side device
    held = {}
    with torch.cuda.device(sdev):
        other = torch.cuda.Stream()
        ev = torch.cuda.Event()
        ev.record()
        if base == 'event_destroy':
            e2 = torch.cuda.Event()
            e2.record()
            held['ev'] = e2
        if base == 'pin_free_used':
            p = torch.empty(1 << 20, dtype=torch.uint8, pin_memory=True)
            with torch.cuda.stream(other):
                torch.empty(1 << 20, dtype=torch.uint8, device=f'cuda:{sdev}').copy_(p, non_blocking=True)
            held['p'] = p
            del p
        if base == 'replay':
            gr0, yr0 = torch.cuda.CUDAGraph(), torch.zeros(1 << 16, device=f'cuda:{sdev}')
            torch.cuda.synchronize(sdev)
            with torch.cuda.graph(gr0, stream=other):
                yr0.add_(1)
            held['g'] = gr0
        if base == 'h2d':
            held['src'] = torch.ones(1 << 20, dtype=torch.uint8, pin_memory=True)
            held['dst'] = torch.empty(1 << 20, dtype=torch.uint8, device=f'cuda:{sdev}')
        if base == 'staging_acquire':
            from isaac_bridge.block_learner import Staging
            st = Staging()
            k, _ = st.acquire(1000)
            st.release(k)
            held['st'] = st
        if base == 'peer_copy':
            held['src'] = torch.randn(1 << 20, device='cuda:1')
            held['dst'] = torch.empty(1 << 20, device='cuda:0')
        if base == 'job_events':
            evs = []
            for _ in range(4):
                t_ = torch.empty(1 << 18, device=f'cuda:{sdev}')
                with torch.cuda.stream(other):
                    t_.add_(1)
                    e_ = torch.cuda.Event()
                    e_.record()
                evs.append((t_, e_))
            held['job'] = dict(events=evs)
        if base == 'capture':
            x_side = ins[1].clone().cuda(sdev)
            W_side = W.cuda(sdev)
            held['x_side'], held['W_side'] = x_side, W_side
            s_q = torch.cuda.Stream(device=sdev)

            def body_s(x):
                a = x @ W_side
                b = torch.relu(a)*0.5+x
                return b/(b.sum(dim=1, keepdim=True).abs()+1.0)
            held['body_s'] = body_s
            # the side graph's own quiet reference
            gq = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s_q):
                for _ in range(2):
                    body_s(x_side)
                torch.cuda.current_stream().synchronize()
                gq.capture_begin(capture_error_mode=mode)
                try:
                    yq = body_s(x_side)
                finally:
                    gq.capture_end()
            held['gq'], held['yq'] = gq, yq
        torch.cuda.synchronize(sdev)
    go, done, warmed, side = threading.Event(), threading.Event(), threading.Event(), {}

    def worker():
        torch.cuda.set_device(sdev)
        if base == 'capture':   # (this thread's own cuBLAS handle / workspace and warm-up before any capture, as the
            s_s = torch.cuda.Stream(device=sdev)   # actor thread has before its captures)
            with torch.cuda.stream(s_s):
                for _ in range(2):
                    held['body_s'](held['x_side'])
            torch.cuda.synchronize(sdev)
        warmed.set()
        go.wait()
        try:
            if base == 'query':
                side['value'] = bool(ev.query())
            elif base == 'sync':
                ev.synchronize()
            elif base == 'stream_sync':
                other.synchronize()
            elif base == 'event_destroy':
                held.pop('ev')
            elif base == 'pin_alloc':
                side['n'] = torch.empty(16 << 20, dtype=torch.uint8, pin_memory=True).numel()
            elif base == 'pin_free_used':
                held.pop('p')
            elif base == 'dev_alloc':
                with torch.cuda.stream(other):
                    t_ = torch.ones(16 << 20, device=f'cuda:{sdev}')   # (64 MiB float32)
                    side['sum'] = None
                    held['big'] = t_
            elif base == 'replay':
                with torch.cuda.stream(other):
                    held['g'].replay()
            elif base == 'h2d':
                with torch.cuda.stream(other):
                    held['dst'].copy_(held['src'], non_blocking=True)
            elif base == 'staging_acquire':
                i, _ = held['st'].acquire(16 << 20)
                held['st'].drop(i)
            elif base == 'peer_copy':
                with torch.cuda.stream(other):
                    held['dst'].copy_(held['src'], non_blocking=True)
            elif base == 'job_events':
                job = held.pop('job')
                side['queried'] = [bool(e.query()) for _, e in job['events']]
                for _, e in job['events']:
                    e.synchronize()
                del job
            elif base == 'capture':
                gs = torch.cuda.CUDAGraph()
                with torch.cuda.stream(s_s):
                    gs.capture_begin(capture_error_mode=mode)
                    try:
                        ys = held['body_s'](held['x_side'])
                    finally:
                        gs.capture_end()
                held['gs'], held['ys'] = gs, ys
            side['outcome'] = 'ok'
        except BaseException as e:
            side['outcome'] = 'error: '+_err(e)
        done.set()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    warmed.wait(60)
    x_i = ins[0].clone()
    s_i = torch.cuda.Stream(device=0)
    try:
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.device(0), torch.cuda.stream(s_i):
            for _ in range(2):
                body(x_i)
            torch.cuda.current_stream().synchronize()
            gr.capture_begin(capture_error_mode=mode)
            try:
                y_i = body(x_i)
                go.set()
                res['side_during_capture'] = done.wait(2.0)
                for _ in range(4):   # (the capture stays open a little longer)
                    y_i = y_i*1.0
            finally:
                gr.capture_end()
        res['capture'] = 'ok'
    except BaseException as e:
        res['capture'] = 'error: '+_err(e)
        gr = None
    go.set()
    t.join(30)
    res['side'] = side.get('outcome', 'not run')
    # bit-for-bit replays: the interfered graph vs the quiet one, same inputs
    if gr is not None:
        try:
            bad = 0
            for r in range(replays):
                src = ins[r % 4]
                x_ref.copy_(src)
                x_i.copy_(src)
                g_ref.replay()
                gr.replay()
                torch.cuda.synchronize(0)
                if not torch.equal(y_ref, y_i):
                    bad += 1
            res['replays'] = replays
            res['replay_mismatches'] = bad
            if 'gs' in held:   # the side thread's graph vs its own quiet reference, interleaved with ours
                bad_s = 0
                for r in range(replays):
                    held['gs'].replay()
                    held['gq'].replay()
                    gr.replay()
                    torch.cuda.synchronize()
                    if not torch.equal(held['ys'], held['yq']):
                        bad_s += 1
                res['side_replay_mismatches'] = bad_s
                res['pools_distinct'] = len({gr.pool(), held['gs'].pool(), g_ref.pool(), held['gq'].pool()}) == 4
        except BaseException as e:
            res['replay'] = 'error: '+_err(e)
    print('RESULT', json.dumps(res), flush=True)


def ok(d):
    good = d.get('capture') == 'ok' and d.get('side') == 'ok' and d.get('replay_mismatches') == 0
    if d['case'].endswith('capture') and d['case'] != 'none':
        good = good and d.get('side_replay_mismatches') == 0 and d.get('pools_distinct', False)
    return good


def main():
    args = sys.argv[1:]
    mode = args[args.index('--mode')+1] if '--mode' in args else 'thread_local'
    reps = int(args[args.index('--reps')+1]) if '--reps' in args else 5
    replays = int(args[args.index('--replays')+1]) if '--replays' in args else 128
    skip = {i for k in ('--mode', '--reps', '--replays') if k in args for i in (args.index(k), args.index(k)+1)}
    cases = [a for i, a in enumerate(args) if i not in skip] or list(CASES)
    for c in cases:
        n_ok = 0
        for r in range(reps):
            p = subprocess.run([sys.executable, os.path.abspath(__file__), '--child', c, mode, str(replays)],
                               capture_output=True, text=True, timeout=600)
            line = next((x for x in p.stdout.splitlines() if x.startswith('RESULT ')), None)
            d = json.loads(line[7:]) if line else dict(case=c, mode=mode, crashed=p.returncode,
                                                       tail=(p.stdout+p.stderr)[-300:])
            d['rep'] = r
            n_ok += ok(d)
            print(json.dumps(d), flush=True)
        print(f'SUMMARY {c} mode {mode}: {n_ok}/{reps} pass', flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 4 and sys.argv[1] == '--child':
        child(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        main()
