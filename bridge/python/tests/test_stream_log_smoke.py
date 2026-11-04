"""CPU tests of the decision stream log (train_tok --log-stream, isaac_bridge/tok_stream.py, rl/docs/STREAM_LOG_FORMAT.md). No game,
no GPU. Plain script:  cd rl/bridge/python && python -X utf8 tests/test_stream_log_smoke.py
Checks:
  (a) the sampling: rates, whole episodes, stable, the ROW set a subset of the stream set;
  (b) the writer: records back byte for byte, per worker, over parts; a part cut off by a crash reads to its last whole
      block; a full queue drops and counts, the caller never waits; a write failure stops the log, not the caller;
  (c) the real learn() under the CPU shim: the numbers trained on are the same with the log on (rollout, weights, Adam
      bit for bit), the actor's bytes too; what the log holds == the rollout: z, g == the policy's own z, gn at the update's
      start (bfloat16), state, action, reward, events, close_*, every closed decision of a sampled episode and no other, the
      ROW records byte for byte; the columns; the log's formats readable through format.json alone.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_temporal_smoke as TS   # noqa: E402,F401  (its path set-up, gymnasium stand-in)

import numpy as np   # noqa: E402
import torch   # noqa: E402

from isaac_bridge import tok_aux as A   # noqa: E402
from isaac_bridge import tok_stream as L   # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def case_a():
    n = 200000
    w = np.arange(n) % 64
    seed = 2147490000 + (np.arange(n) // 7)
    ep = np.arange(n) // 3
    h = L.episode_hash(w, seed, ep)
    r = float((h < 2500).mean())
    r2 = float((h < 200).mean())
    check('(a) rates close to 1/4 and 1/50', abs(r - 0.25) < 0.01 and abs(r2 - 0.02) < 0.003, f'{r:.4f}, {r2:.4f}')
    check('(a) stable', np.array_equal(h, L.episode_hash(w, seed, ep)) and int(L.episode_hash(3, 5, 7)) == int(L.episode_hash(3, 5, 7)))
    check('(a) the ROW set is a subset of the stream set', bool(((h < 200) <= (h < 2500)).all()))
    # an episode's decisions all alike
    e = L.episode_hash(np.full(50, 5), np.full(50, 123), np.full(50, 9))
    check('(a) one episode: one verdict', len(set(e.tolist())) == 1)


def case_b(tmp):
    root = tmp / 'unit'
    sl = L.StreamLog(root, 8, 3, rate=1.0, row_rate=0.5)
    rec = np.zeros(10, sl.dt)
    rec['worker'] = np.arange(10) % 2
    rec['t'] = np.arange(10)
    rec['z'] = np.arange(80).reshape(10, 8)
    full = np.zeros(4, sl.dt_row)
    full['worker'] = [0, 0, 1, 1]
    full['update'] = 7
    full['row']['t'] = [1, 2, 3, 4]
    plan = dict(rec=rec.copy(), full=full.copy())
    sl.stage = torch.zeros((16, 16), dtype=torch.int16)
    sl.stage[:10, :8] = torch.from_numpy(rec['z'].astype(np.int16))
    sl.submit(plan)
    sl.close()
    got = {w_: r_ for w_, r_ in L.iter_records(root / 'stream', 'stream')}
    check('(b) stream records back per worker', set(got) == {0, 1} and np.array_equal(got[0]['t'], [0, 2, 4, 6, 8]) and
          np.array_equal(got[1]['z'], rec['z'][1::2]))
    got_r = {w_: r_ for w_, r_ in L.iter_records(root / 'stream', 'row')}
    check('(b) ROW records back', np.array_equal(got_r[1]['row']['t'], [3, 4]) and got_r[0]['update'][0] == 7)
    # a crash: a part cut short
    p = sorted((root / 'stream').glob('stream-w000-*.zst'))[0]
    root2 = tmp / 'cut'
    sl2 = L.StreamLog(root2, 8, 3, rate=1.0, row_rate=0.0)
    sl2.stage = torch.zeros((2048, 16), dtype=torch.int16)
    big = np.zeros(1000, sl2.dt)
    big['t'] = np.arange(1000)
    for k_ in range(3):
        sl2.submit(dict(rec=big.copy(), full=np.zeros(0, sl2.dt_row)))
        for _ in range(100):
            if sl2.q.empty() and sl2.rows_written == 1000 * (k_ + 1):
                break
            time.sleep(0.05)
    # (read the file while the writer holds it open: its blocks flushed per update, no end of frame)
    pc = sorted((root2 / 'stream').glob('stream-w000-*.zst'))[0]
    n_open = sum(len(r_) for _, r_ in L.iter_records(root2 / 'stream', 'stream'))
    check('(b) a live or cut-off part reads to its last flushed block', n_open == 3000, f'{n_open} of 3000')
    data = pc.read_bytes()
    pc2 = root2 / 'stream' / 'stream-w001-p0000.zst'
    pc2.write_bytes(data[:len(data) // 2])
    got2 = dict(L.iter_records(root2 / 'stream', 'stream', worker=1))
    check('(b) half a file: whole records only', 0 <= len(got2[1]) < 3000 and (len(got2[1]) == 0 or got2[1]['t'][0] == 0), f'{len(got2[1])} records')
    sl2.close()
    # a full queue: dropped, counted, no wait
    sl3 = L.StreamLog(tmp / 'full', 8, 3, rate=1.0, row_rate=0.0, depth=1)
    orig = sl3.stream.write

    def slow(*a_):
        time.sleep(0.5)
        return orig(*a_)
    sl3.stream.write = slow
    sl3.stage = torch.zeros((16, 16), dtype=torch.int16)
    t0 = time.perf_counter()
    for _ in range(5):
        sl3.submit(dict(rec=rec.copy(), full=np.zeros(0, sl3.dt_row)))
    dt = time.perf_counter() - t0
    check('(b) a full queue: the caller does not wait, the drop is counted', dt < 0.2 and sl3.dropped_jobs >= 1 and sl3.columns()['log_dropped'] >= 10,
          f'{dt * 1000:.0f} ms, {sl3.dropped_jobs} jobs')
    sl3.close()
    # a write failure
    sl4 = L.StreamLog(tmp / 'bad', 8, 3, rate=1.0, row_rate=0.0)

    def boom(*a_):
        raise OSError('No space left on device')
    sl4.stream.write = boom
    sl4.stage = torch.zeros((16, 16), dtype=torch.int16)
    sl4.submit(dict(rec=rec.copy(), full=np.zeros(0, sl4.dt_row)))
    for _ in range(100):
        if sl4.error:
            break
        time.sleep(0.02)
    sl4.submit(dict(rec=rec.copy(), full=np.zeros(0, sl4.dt_row)))
    check('(b) a write failure stops the log, not the caller; it shows', 'No space' in sl4.columns()['log_error'])
    sl4.close()


def driver(tree, out, *extra):
    cmd = [sys.executable, '-X', 'utf8', str(HERE / 'aux_learn_driver.py'), '--tree', str(tree), '--out', str(out)] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=dict(os.environ, OMP_NUM_THREADS='2'), timeout=900)
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-3000:])
        raise RuntimeError(f'driver failed: {extra}')
    return torch.load(Path(out) / 'result.pt', weights_only=False), p.stdout


def same_tensors(a, b):
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


def same_opt(a, b):
    return len(a['state']) == len(b['state']) and all(
        (not torch.is_tensor(a['state'][k][n])) or torch.equal(a['state'][k][n], b['state'][k][n]) for k in a['state'] for n in a['state'][k])


def case_c(tmp):
    tree = HERE.parent.parent.parent
    rates = '--log-stream,1,--log-stream-rate,0.5,--log-row-rate,0.25,--overlap'
    for fast in (1, 2):
        off, _ = driver(tree, tmp / f'off{fast}', '--updates', '2', '--fast', str(fast), '--extra=--overlap')
        on, _ = driver(tree, tmp / f'on{fast}', '--updates', '2', '--fast', str(fast), f'--extra={rates}', '--dump-z', '1')
        common = [k for k in off['rollout'] if k in on['rollout']]
        check(f'(c) [--learner-fast {fast}] the rollout as off', all(np.array_equal(off['rollout'][k], on['rollout'][k]) for k in common))
        check(f'(c) [--learner-fast {fast}] the actor\'s bytes as off', off['actor_hash'] == on['actor_hash'])
        check(f'(c) [--learner-fast {fast}] the weights after two updates as off, bit for bit', same_tensors(off['weights'], on['weights']))
        check(f'(c) [--learner-fast {fast}] Adam as off', same_opt(off['opt'], on['opt']))
        row = on['progress'][-1]
        print(f"info: [--learner-fast {fast}] tiny model: u_log {float(row['u_log']) * 1000:.1f} ms of {float(row['update_s']) * 1000:.0f} ms; rows {row['log_rows']}, ROW {row['log_row_rows']}")
        root = Path(on['last']).parent.parent / 'stream'
        fmt = json.loads((root / 'format.json').read_text())
        check('(c) format.json says the record', fmt['record'] == L.stream_dtype(fmt['width'], fmt['heads']).itemsize and fmt['rate'] == 0.5 and fmt['state'] == list(A.STATE))
        rec = np.concatenate([r_ for _, r_ in L.iter_records(root, 'stream')]) if list(root.glob('stream-w*')) else None
        full = np.concatenate([r_ for _, r_ in L.iter_records(root, 'row')])
        ru = rec[rec['update'] == rec['update'].min()]   # the first update's job
        rl = on['rollout']
        W, T = rl['rows'].shape[0], rl['length']
        want = []
        for w in range(W):
            usable = int(T[w]) - int(rl['open'][w] == T[w] - 1)
            for t in range(usable):
                r = rl['rows'][w, t]
                want.append((w, t, int(L.episode_hash(w, r['seed'], r['episode']))))
        sel = [(w, t) for w, t, h in want if h < 5000]
        got = sorted((int(a), int(b)) for a, b in zip(ru['worker'], ru['t']))
        rec_t = [int(rl['rows'][w, t]['t']) for w, t in sel]
        check(f'(c) [--learner-fast {fast}] exactly the closed decisions of sampled episodes, none else', got == sorted((w, int(rl['rows'][w, t]['t'])) for w, t in sel),
              f'{len(got)} of {len(want)} closed decisions')
        ok = True
        by_key = {(w, int(rl['rows'][w, t]['t']), int(rl['rows'][w, t]['episode'])): (w, t) for w, t in sel}
        for r in ru:
            w, t = by_key[(int(r['worker']), int(r['t']), int(r['episode']))]
            src = rl['rows'][w:w + 1, t:t + 1].reshape(1)
            ok = ok and np.array_equal(r['state'], A.state_rows(src)[0]) and r['reward'] == rl['reward'][w, t] and \
                np.array_equal(r['action'], rl['actions'][w, t]) and bool(r['terminal']) == bool(rl['terminal'][w, t]) and \
                r['close_done'] == rl['close_done'][w, t] and r['close_ev'] == rl['close_ev'][w, t] and r['close_hurt'] == rl['close_hurt'][w, t] and \
                r['events'] == src['events'][0] and r['first'] == src['first'][0] and r['stage'] == src['stage'][0] and r['seed'] == src['seed'][0]
        check(f'(c) [--learner-fast {fast}] every record\'s fields == the rollout\'s', bool(ok))
        zb = torch.from_numpy(ru['z'].copy()).view(torch.bfloat16 if False else torch.int16)
        check(f'(c) [--learner-fast {fast}] z, g hold values', float(np.abs(ru['z']).sum()) > 0 and float(np.abs(ru['g']).sum()) > 0 and zb.shape[1] == fmt['width'])
        fu = full[full['update'] == full['update'].min()]
        ok_r, n_r = True, 0
        for r in fu:
            key = (int(r['worker']), int(r['row']['t']), int(r['row']['episode']))
            w, t = by_key[key]
            n_r += 1
            ok_r = ok_r and r['row'].tobytes() == rl['rows'][w, t].tobytes() and np.array_equal(r['action'], rl['actions'][w, t])
        check(f'(c) [--learner-fast {fast}] the ROW records, byte for byte, of the 1/4 episodes (a subset)', bool(ok_r) and n_r == sum(h < 2500 for _, _, h in want), f'{n_r} records')
        del rec_t
        # z, g against the policy's own at the update's start (padded and packed paths: float rounding, then bfloat16's)
        mz, mg, n_z = 0.0, 0.0, 0
        for r in ru:
            ref = on['z_ref'].get((int(r['worker']), int(r['t']), int(r['episode'])))
            zr, gr = ref
            za = torch.from_numpy(r['z'].astype(np.int16).copy()).view(torch.bfloat16).float().numpy()
            ga = torch.from_numpy(r['g'].astype(np.int16).copy()).view(torch.bfloat16).float().numpy()
            sz, sg = np.maximum(np.abs(zr), 1e-2), np.maximum(np.abs(gr), 1e-2)
            mz, mg, n_z = max(mz, float((np.abs(za - zr) / sz).max())), max(mg, float((np.abs(ga - gr) / sg).max())), n_z + 1
        check(f'(c) [--learner-fast {fast}] z, g == the policy\'s own at the update\'s start, within bfloat16 (relative)',
              n_z > 0 and mz < 0.02 and mg < 0.02, f'{n_z} records, max rel z {mz:.1e}, g {mg:.1e}')


def main():
    tmp = Path(tempfile.mkdtemp(prefix='stream-test-', dir=os.environ.get('AUX_TEST_TMP') or None))
    case_a()
    case_b(tmp)
    case_c(tmp)
    print('info: test directory', tmp)
    print('RESULT', 'all checks passed' if not FAIL else f'{len(FAIL)} FAILED: {FAIL}')
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
