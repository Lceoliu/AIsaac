"""Decision stream log (train_tok --log-stream 1; rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 5.1, the format in
rl/docs/STREAM_LOG_FORMAT.md): what the offline world model trains on. Learner side only: the actor sees nothing; a run with
the log on trains on the same numbers as one without (tests/test_stream_log_smoke.py).

Per update, in the learner: an episode is logged when a hash of (worker, seed, episode number) falls under the rate, so its
decisions stay contiguous over rollouts; only closed decisions (a rollout's open one comes as decision 0 of the next). The
spatial representations z_t, g_t come from the update's start-of-update pass (the learner's weights then, inference mode,
the actor's weights up to one update), in bfloat16; everything else from the rollout's host arrays. The learner thread only
gathers: the GPU's rows go to a pinned buffer with its other work, are read after the update's last synchronisation, and
one job goes to a bounded queue (full: the job is dropped and counted, the training never waits). A writer thread
compresses per worker (zstd, one stream per worker and kind, a block flushed per update, a new part every PART_BYTES raw).
"""
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .tok_aux import STATE, state_rows
from .tok_obs import ROW

PART_BYTES = 1 << 30
SCALE = 10000
LEVEL = 3
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def stream_dtype(width, heads):
    """One decision of the stream log, packed. z, g: bfloat16's raw bits."""
    return np.dtype([('z', '<i2', (width,)), ('g', '<i2', (width,)), ('state', '<f4', (len(STATE),)), ('action', 'i1', (heads,)),
                     ('pending', 'i1', (heads,)), ('reward', '<f4'), ('hurt', '<f4'), ('close_hurt', '<f4'),
                     ('events', 'u1'), ('close_ev', 'u1'), ('close_done', 'u1'), ('terminal', 'u1'), ('first', 'u1'),
                     ('stage', 'u1'), ('room_type', 'u1'), ('room', 'u1'), ('worker', '<u2'), ('t', '<i4'),
                     ('episode', '<i8'), ('seed', '<i8'), ('update', '<u4')])


def row_dtype(heads):
    """One decision of the full-record log, packed (ROW's own layout inside)."""
    return np.dtype([('row', ROW), ('action', 'i1', (heads,)), ('pending', 'i1', (heads,)), ('worker', '<u2'), ('update', '<u4')])


def episode_hash(worker, seed, episode):
    """int64 arrays -> uint64 in [0, SCALE): stable over runs and rollouts (splitmix64's finaliser)."""
    x = (np.asarray(worker, np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (np.asarray(seed).astype(np.uint64) * np.uint64(_M1)) \
        ^ (np.asarray(episode).astype(np.uint64) * np.uint64(_M2))
    x = (x ^ (x >> np.uint64(30))) * np.uint64(_M1)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(_M2)
    return (x ^ (x >> np.uint64(31))) % np.uint64(SCALE)


class _Shards:
    """One zstd stream per worker for one kind of record; parts of PART_BYTES raw."""

    def __init__(self, root, kind):
        import zstandard
        self.zstd = zstandard
        self.root, self.kind = Path(root), kind
        self.w = {}
        self.parts = {}
        self.bytes = 0   # raw

    def _open(self, worker):
        part = self.parts.get(worker, -1) + 1
        path = self.root / f'{self.kind}-w{worker:03d}-p{part:04d}.zst'
        while path.exists():   # (a resumed run: after its parts)
            part += 1
            path = self.root / f'{self.kind}-w{worker:03d}-p{part:04d}.zst'
        self.parts[worker] = part
        f = open(path, 'wb')
        # (a compressor per stream: one ZstdCompressor holds one context, two open streams would mix)
        self.w[worker] = [f, self.zstd.ZstdCompressor(level=LEVEL).stream_writer(f, closefd=False), 0]

    def write(self, worker, data):
        if worker not in self.w or self.w[worker][2] >= PART_BYTES:
            self.close_one(worker)
            self._open(worker)
        s = self.w[worker]
        s[1].write(data)
        s[2] += len(data)
        self.bytes += len(data)

    def flush(self):
        for f, z, _ in self.w.values():
            z.flush(self.zstd.FLUSH_BLOCK)
            f.flush()

    def close_one(self, worker):
        s = self.w.pop(worker, None)
        if s is not None:
            s[1].close()
            s[0].close()

    def close(self):
        for w in list(self.w):
            self.close_one(w)


class StreamLog:
    def __init__(self, out, width, heads, rate=0.25, row_rate=0.02, depth=8,
                 input_schema='legacy-tok-row-v7'):
        import zstandard  # noqa: F401  (refused at the start, not in the writer thread)
        self.root = Path(out) / 'stream'
        self.root.mkdir(parents=True, exist_ok=True)
        self.width, self.heads = int(width), int(heads)
        self.dt, self.dt_row = stream_dtype(width, heads), row_dtype(heads)
        self.cut, self.cut_row = int(round(rate * SCALE)), int(round(row_rate * SCALE))
        fmt = dict(version=2, input_schema=input_schema,
                   feature_semantics='temporal-zr-zg' if input_schema!='legacy-tok-row-v7' else 'legacy-spatial-z-g',
                   diagnostic_state_semantics='legacy public/reward diagnostics; not actor input tokens',
                   width=width, heads=heads, rate=rate, row_rate=row_rate, state=list(STATE), scale=SCALE,
                   stream=[list(map(str, f)) if len(f) == 2 else [f[0], str(f[1]), list(f[2])] for f in self.dt.descr],
                   record=self.dt.itemsize, record_row=self.dt_row.itemsize, row_itemsize=ROW.itemsize,
                   row_fields=ROW.descr.__repr__())
        (self.root / 'format.json').write_text(json.dumps(fmt, indent=1),encoding='utf-8')
        self.stream, self.rows = _Shards(self.root, 'stream'), _Shards(self.root, 'row')
        self.q = queue.Queue(maxsize=depth)
        self.dropped_jobs = self.dropped_rows = 0
        self.rows_written = self.row_rows_written = 0
        self.write_s = 0.0
        self.error = None
        self.stage = None
        self.t = threading.Thread(target=self._writer, name='stream-log', daemon=True)
        self.t.start()

    def plan(self, roll, flat, device):
        """The update's selection among the start-of-update pass's rows `flat` (closed decisions of sampled episodes).
        Returns a plan dict: pos (positions in flat), the host copies job() will need are taken here (before `fresh`)."""
        import torch
        w, t = flat // roll.cap, flat % roll.cap
        rows = roll.rows
        h = episode_hash(w, rows['seed'][w, t], rows['episode'][w, t])
        usable = roll.length[w] - (roll.open[w] == roll.length[w] - 1)
        closed = t < usable
        pos = np.flatnonzero(closed & (h < self.cut))
        pos_row = np.flatnonzero(closed & (h < self.cut_row))
        n = len(pos)
        if self.stage is None or self.stage.shape[0] < n:
            self.stage = None
            self.stage = torch.empty((int(n * 1.25) + 64, 2 * self.width), dtype=torch.int16).pin_memory()
        return dict(pos=pos, pos_row=pos_row, w=w, t=t, device=device, parts=[], lo=0, n=n)

    def part(self, plan, lo, hi, z, g):
        """A pass part's rows lo .. hi of flat: its sampled rows' z, g [hi - lo, width] (float32, device) into the plan."""
        import torch
        a, b = np.searchsorted(plan['pos'], (lo, hi))
        if b > a:
            idx = torch.from_numpy(plan['pos'][a:b] - lo).to(z.device)
            x = torch.cat([z.index_select(0, idx), g.index_select(0, idx)], 1).to(torch.bfloat16)
            plan['parts'].append(x.view(torch.int16))

    def gather(self, plan, roll, update):
        """The host records (the rollout may be reused after the learner's `fresh`) and the GPU's rows on their way to the
        pinned buffer (issued on the current stream; read in submit after the caller's synchronisation)."""
        import torch
        p, w, t = plan['pos'], plan['w'], plan['t']
        n = len(p)
        if plan['parts']:
            on = torch.cat(plan['parts'])
            self.stage[:n].copy_(on, non_blocking=True)
        ws, ts = w[p], t[p]
        rec = np.empty(n, self.dt)
        if n:
            rows = roll.rows
            sub = {k: rows[k][ws, ts] for k in ('player', 'phearts', 'n_ent', 'room_type', 'stage', 'hurt', 'events', 'first', 't',
                                                'episode', 'seed')}
            st = _state_of(rows, ws, ts, sub)
            rec['state'] = st
            rec['action'] = roll.actions[ws, ts]
            rec['pending'] = roll.pending[ws, ts]
            rec['reward'] = roll.reward[ws, ts]
            rec['hurt'] = sub['hurt']
            rec['close_hurt'] = roll.close_hurt[ws, ts]
            rec['events'], rec['close_ev'], rec['close_done'] = sub['events'], roll.close_ev[ws, ts], roll.close_done[ws, ts]
            rec['terminal'] = roll.terminal[ws, ts]
            rec['first'], rec['stage'], rec['room_type'] = sub['first'], sub['stage'], sub['room_type']
            m = rows['map'][ws, ts, 3].reshape(n, -1)
            rec['room'] = np.where(m.any(1), m.argmax(1), 255)
            rec['worker'], rec['t'], rec['episode'], rec['seed'] = ws, sub['t'], sub['episode'], sub['seed']
            rec['update'] = update
        pr = plan['pos_row']
        # Structured assignment copies declared fields, not NumPy alignment gaps.
        # Canonical zero padding avoids emitting unrelated allocator contents.
        full = np.zeros(len(pr), self.dt_row)
        if len(pr):
            wr, tr = w[pr], t[pr]
            full['row'] = roll.rows[wr, tr]
            full['action'], full['pending'] = roll.actions[wr, tr], roll.pending[wr, tr]
            full['worker'], full['update'] = wr, update
        plan['rec'], plan['full'] = rec, full

    def submit(self, plan):
        """After the caller's synchronisation: the job to the writer; a full queue drops it (counted)."""
        rec, full = plan['rec'], plan['full']
        n = len(rec)
        if n:
            x = self.stage[:n].numpy()
            rec['z'] = x[:, :self.width]
            rec['g'] = x[:, self.width:]
        try:
            self.q.put_nowait((rec, full))
        except queue.Full:
            self.dropped_jobs += 1
            self.dropped_rows += n

    def _writer(self):
        try:
            while True:
                job = self.q.get()
                if job is None:
                    return
                t0 = time.perf_counter()
                for shards, arr, key in ((self.stream, job[0], job[0]['worker'] if len(job[0]) else None),
                                         (self.rows, job[1], job[1]['worker'] if len(job[1]) else None)):
                    if key is None:
                        continue
                    for w in np.unique(key):
                        shards.write(int(w), arr[key == w].tobytes())
                    shards.flush()
                self.rows_written += len(job[0])
                self.row_rows_written += len(job[1])
                self.write_s += time.perf_counter() - t0
        except Exception as exc:   # (a full disk: the log stops, the training goes on; the columns show it)
            self.error = repr(exc)[:300]
            print('stream log: stopped:', self.error, flush=True)
            while True:
                if self.q.get() is None:
                    return

    def columns(self):
        return dict(log_rows=self.rows_written, log_row_rows=self.row_rows_written, log_raw_gb=(self.stream.bytes + self.rows.bytes) / 2 ** 30,
                    log_dropped=self.dropped_rows, log_queue=self.q.qsize(), log_write_s=self.write_s, log_error=self.error or '')

    def close(self, timeout=300):
        """The queue drained, every stream closed (valid to its end)."""
        self.q.put(None)
        self.t.join(timeout)
        self.stream.close()
        self.rows.close()


def _state_of(rows, ws, ts, sub):
    """state_rows' columns for the rows at (ws, ts), gathering only what they read."""
    view = np.zeros(len(ws), np.dtype([('player', '<f4', rows['player'].shape[2:]), ('phearts', '<f4', (3,)),
                                       ('ent', '<f4', (rows['ent'].shape[2], 1)), ('n_ent', '<i4'), ('room_type', 'u1'), ('stage', 'u1')]))
    view['player'], view['phearts'], view['n_ent'] = sub['player'], sub['phearts'], sub['n_ent']
    view['room_type'], view['stage'] = sub['room_type'], sub['stage']
    from .tok_aux import NPC_COLUMN
    view['ent'][..., 0] = rows['ent'][ws, ts, :, NPC_COLUMN]
    return state_rows(view, npc_column=0)


def iter_records(root, kind='stream', worker=None):
    """(the reader) (worker, numpy records) per part file of a log directory, in order. Tolerates a part cut off by a
    crash: its whole records up to there."""
    import zstandard
    root = Path(root)
    fmt = json.loads((root / 'format.json').read_text(encoding='utf-8'))
    from .structured_input import SCHEMA
    if fmt.get('version')!=2 or fmt.get('input_schema') not in (SCHEMA,'legacy-tok-row-v7'):
        raise ValueError('stream input schema is missing or unsupported; implicit legacy conversion forbidden')
    if (fmt.get('row_itemsize')!=ROW.itemsize or fmt.get('row_fields')!=repr(ROW.descr)
            or fmt.get('record_row')!=row_dtype(fmt['heads']).itemsize):
        raise ValueError('stream ROW layout does not match this decoder')
    dt = stream_dtype(fmt['width'], fmt['heads']) if kind == 'stream' else row_dtype(fmt['heads'])
    pat = f'{kind}-w{worker:03d}-*.zst' if worker is not None else f'{kind}-w*.zst'
    for path in sorted(root.glob(pat)):
        with open(path, 'rb') as f:
            r = zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True)
            chunks = []
            try:
                while True:
                    b = r.read(1 << 24)
                    if not b:
                        break
                    chunks.append(b)
            except zstandard.ZstdError:
                pass
        data = b''.join(chunks)
        n = len(data) // dt.itemsize
        records=np.frombuffer(data,dt,n)
        if kind=='row' and fmt['input_schema']==SCHEMA:
            from .structured_input import VERSION
            if not np.all(records['row']['visible_schema']==VERSION):
                raise ValueError('new stream contains a missing/old visible frame')
        yield int(path.name.split('-w')[1][:3]), records
