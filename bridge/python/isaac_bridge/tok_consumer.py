"""P package phase 1 (2026-11-19): a collect through the C++ driver (abp_tok_thread.h TokDriver) -- the Python consumer.

The driver thread answers the main workers (P-driver-design 11); this module, on the actor's thread, takes its records
in sequence order and does what the Python loop did after an answer: Episodes.arrive (book: rewards, the episode's
bookkeeping) over each call's records, then store (the rollout's rows / actions / values / version / temporal counters,
and each decision's K / V, s and o copied from the driver's output ring into the rollout's rows, on the actor's stream
-- the driver's own: the copy is ordered before the ring rows can be written again, P-driver-design 11.2). Checked per
record (11.6): the go mask equals Episodes.arrive's, the pending the call used equals the consumer's mirror after
arrive. The rollout's switch is the consumer's: it collects until the committed usable count reaches the target, then
pauses the driver and drains it (11.1). Worker ends / EOFs reach the sampler's bookkeeping; a driver fatal is raised
here (a visible-v3 one as VisibleV3Fatal).
"""
import time

import numpy as np

from .tok_obs import ROW
from .tok_obs import DONE_BRANCH


def supported(args, *, t_on, mem_on, frame_v3, block_on, proc, n_act, snaps, priv_on, pf_on, lab, aux, plr, lanes_gpu):
    """(P-driver-design 11.3) None when phase 1 supports this run, else why not."""
    why = [w for w, bad in (('not temporal', not t_on), ('a GRU', mem_on), ('not --frame v3', not frame_v3),
                            ('block memory', block_on), ('a learner process', proc), ('more actor processes', n_act != 1),
                            ('branch snapshots', bool(snaps)), ('pickup force', pf_on),
                            ('the lab', lab), ('branch aux', aux), ('archive PLR', plr),
                            ('the teacher on its own GPU', lanes_gpu)) if bad]
    return ', '.join(why) or None


class DriverCollect:
    """The driver, its GraphActor(fold=True), the staging blocks and the output ring for one training run."""

    def __init__(self, mod, model, sampler, n, g_ents, gkw, t_state, bank_tstate, stream, version, cap_t, trunc,
                 blocks=64, ring=8192, priv_dtype=None):
        import torch
        from .tok_driver import Middle
        from .tok_policy import GraphActor
        kw = {k: v for k, v in gkw.items() if k in ('bf16', 'compile_mode', 'sdpa_math', 'row_sizes')}
        self.ga = GraphActor(model, n, g_ents, stream=stream, crn=False, memory=None, temporal=t_state, fold=True, **kw)
        ta = model.tatt
        dev = t_state.kv.device
        self.ring = dict(kv=torch.zeros((ring, ta.L, 2, ta.D), dtype=t_state.dtype, device=dev),
                         s=torch.zeros((ring, ta.D), dtype=t_state.dtype, device=dev),
                         o=torch.zeros((ring, ta.D), device=dev), obase=0)
        self.mid = Middle(self.ga, ring=self.ring)
        if bank_tstate is not None:
            self.mid.set_cf(t_state, bank_tstate)
        self.blocks = np.zeros((blocks, n), ROW)
        self.K, self.sampler, self.stream, self.nh = ring, sampler, stream, self.ga.nh
        d = self.d = mod.TokDriver(self.mid.m)
        ptr = lambda a: 0 if a is None else int(a.ctypes.data)   # noqa: E731
        d.set_sampler(int(sampler.rows.ctypes.data), n, list(sampler._fds), int(sampler.actions.ctypes.data),
                      int(sampler.actions.shape[1]), ptr(getattr(sampler, 'main_value', None)),
                      ptr(getattr(sampler, 'main_vg', None)), ptr(getattr(sampler, 'cf_snap', None)))
        d.set_rules(ROW.fields['done'][1], ROW.fields['first'][1], ROW.fields['t'][1], DONE_BRANCH, int(cap_t), int(trunc))
        d.set_blocks(int(self.blocks.ctypes.data), blocks, n, 0)
        self.pblocks = None
        if priv_dtype is not None:   # (tok_priv: each record's PRIV record copied by the driver beside its row)
            self.pblocks = np.zeros((blocks, n), priv_dtype)
            d.set_priv(int(sampler.priv.ctypes.data), priv_dtype.itemsize, int(self.pblocks.ctypes.data))
        d.set_ring_rows(ring)
        d.set_version(int(version))
        self.started = False
        self.partial = {}   # call -> records taken so far (a take may end inside a call)
        self.stat = dict(records=0, calls=0, ends=0, gone=0, take_s=0.0, waits=0)

    # ---- the collect
    def collect(self, roll, book, pending, target, prof, store, tick=None, weights=None, stopped=lambda: False):
        """Records into the rollout until its committed usable count reaches target (or the workers end / a stop);
        tick(): the lanes' round (each loop); weights(): None, or a callable that swaps the weights (called with the
        driver paused and drained; returns the new version)."""
        c0 = self.d.counters()
        if not self.started:
            self.d.start()
            self.started = True
        else:
            self.d.resume()
        while roll.usable() < target and self.sampler.live and not stopped():
            if tick is not None:
                tick(prof)
            if weights is not None and weights.ready():
                self.drain(roll, book, pending, prof, store)
                self.d.set_version(int(weights()))
                self.d.resume()
            self.step(roll, book, pending, prof, store, 0.002)
        self.drain(roll, book, pending, prof, store)
        c1 = self.d.counters()   # (the driver's ready -> reply clock of this collect, as the Python loop's)
        prof['rr_s'] = prof.get('rr_s', 0.0) + c1['rr_s'] - c0['rr_s']
        prof['rr_n'] = prof.get('rr_n', 0) + c1['rr_n'] - c0['rr_n']
        prof['drv_calls'] = prof.get('drv_calls', 0) + c1['calls'] - c0['calls']
        prof['drv_credit_s'] = prof.get('drv_credit_s', 0.0) + c1['t_credit_s'] - c0['t_credit_s']
        prof['drv_call_s'] = prof.get('drv_call_s', 0.0) + c1['t_call_s'] - c0['t_call_s']

    def drain(self, roll, book, pending, prof, store):
        """Pause the driver; every record it accepted committed (P-driver-design 11.1)."""
        self.d.pause()
        while True:
            got = self.step(roll, book, pending, prof, store, 0.01)
            if not got and self.d.is_paused() and self.d.counters()['queue'] == 0 and not self.partial:
                return

    def step(self, roll, book, pending, prof, store, timeout):
        t0 = time.perf_counter()
        d = self.d.take(256, timeout)
        self.stat['take_s'] += time.perf_counter() - t0
        code, w, text = self.d.fatal()
        if code:
            msg = text.decode('utf-8', 'replace')
            if code == 86:
                from .visible_producer import VisibleV3Fatal
                raise VisibleV3Fatal(f'tok_driver: worker {w}: {msg}')
            raise RuntimeError(f'tok_driver: fatal {code} (worker {w}): {msg}')
        m = len(d['seq'])
        if not m:
            return 0
        for i in range(m):
            kind = int(d['kind'][i])
            if kind:   # a worker's end ('x') / EOF: the sampler's bookkeeping, as poll_ready's _drop
                w_ = int(d['worker'][i])
                self.stat['ends' if kind == 1 else 'gone'] += 1
                c = self.sampler.conns[w_]
                self.sampler._drop(c)
                continue
            cid = int(d['call'][i])
            rec = {k: v[i] for k, v in d.items()}
            self.partial.setdefault(cid, []).append(rec)
            if len(self.partial[cid]) == int(rec['n_call']):
                self.commit(self.partial.pop(cid), roll, book, pending, prof, store)
        return m

    def commit(self, recs, roll, book, pending, prof, store):
        """One call's records: Episodes.arrive over all of them, store of the go ones (11.6's checks)."""
        import torch
        k = len(recs)
        b = int(recs[0]['block'])
        rows = self.blocks[b, :k]
        idx = np.array([int(r['worker']) for r in recs], np.int64)
        go_d = np.array([bool(r['go']) for r in recs])
        tb = time.perf_counter()
        go = book.arrive(roll, rows, idx, pending)
        t1 = time.perf_counter()
        prof['book'] += t1 - tb
        if not np.array_equal(go, go_d):
            raise RuntimeError(f'tok_driver: the go mask differs from Episodes.arrive ({go_d.astype(int)} vs '
                               f'{go.astype(int)}, workers {idx})')
        if go.any():
            g = np.flatnonzero(go)
            a_idx, sub = idx[g], rows[g]
            under_way = pending[a_idx]
            pend_d = np.stack([recs[j]['pending'] for j in g])
            if not np.array_equal(under_way, pend_d):
                raise RuntimeError(f'tok_driver: the pending of workers {a_idx} differs from the driver\'s')
            actions = np.stack([recs[j]['actions'] for j in g]).astype(np.int64)
            logp = np.array([recs[j]['logp'] for j in g], np.float32)
            value = np.array([recs[j]['value'] for j in g], np.float32)
            vg = np.array([recs[j]['vg'] for j in g], np.float32)
            ver = int(recs[g[0]]['version'])
            ring_seq = np.array([int(recs[j]['ring']) for j in g], np.int64)
            kg = len(g)
            r_ = ring_seq % self.K   # (the ring wraps: a call's rows are its sequence numbers mod K)
            dev = self.ring['kv'].device
            # (index tensors through pinned memory, non-blocking: a pageable H2D copy would hold this thread until the
            #  driver's queued replays on the stream were done -- the consumer then as slow as the GPU)
            to_dev = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, np.int64)).pin_memory().to(  # noqa: E731
                dev, non_blocking=True)
            if r_[-1] - r_[0] == kg - 1:
                sel = lambda x: x[int(r_[0]):int(r_[0]) + kg]   # noqa: E731
            else:
                ix = to_dev(r_)
                sel = lambda x: x.index_select(0, ix)   # noqa: E731
            tc = np.stack([recs[j]['tc'] for j in g])
            t = dict(create=tc[:, 0] > 0, nroom=tc[:, 1], steps=tc[:, 2], stage_prev=tc[:, 3])
            priv = self.pblocks[b, :k][g] if self.pblocks is not None else None
            with torch.cuda.stream(self.stream):   # (the driver's stream: the copy before the rows' next write)
                # the decisions' rollout rows (as GraphActor's tdest: Rollout.t_dest before store), K / V, s and o into
                # them from the ring, then store with the counters only (store_temporal's GraphActor branch)
                rows_t = to_dev(roll.t_assign(a_idx))
                roll.t_kv.index_copy_(0, rows_t, sel(self.ring['kv']).to(roll.t_kv.dtype))
                roll.t_s.index_copy_(0, rows_t, sel(self.ring['s']).to(roll.t_s.dtype))
                roll.t_o.index_copy_(0, rows_t + int(roll.t_obase), sel(self.ring['o']).to(roll.t_o.dtype))
                store(roll, a_idx, sub, under_way, actions, logp, value, ver, None, vg, t, priv)
            pending[a_idx] = actions
            self.d.release_ring(int(ring_seq[-1]) + 1)
            prof['calls'] += 1
            prof['rows'] += kg
        self.d.release_block(b)
        prof['store'] += time.perf_counter() - t1
        self.stat['records'] += k
        self.stat['calls'] += 1

    def close(self):
        if self.started:
            self.d.stop()
