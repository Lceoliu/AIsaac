"""B36 (2026-10-16): the lanes' service in train_tok (teacher v2 / the counterfactual teacher's branches), batched,
with a wait window, and asynchronous.

Before: the actor thread answered the lane requests it had seen at each pass of its loop with one lane GraphActor call
and waited for it (train_tok serve_lanes). C99 at 10 minutes (HPC160, 64 workers, --cf 1): 198,819 lane calls for
1,757,944 lane rows (8.8 rows per call: the calls were batched already; cf_lane_calls counts the children's answers,
one per branch decision, not the actor's calls), 670 s inside them = 3.4 ms per call, about 60% of the actor thread's
wall time, during which the workers' own records waited (infer_ms 2.5 -> 6.5-7.6, ~900x -> ~620x).

A service round now (LaneService.tick at every pass of the actor's loop, serve_wait while it waits for the learner):
  1. the requests are read into the sampler's lane_ready (its poll sets as before; poll_lanes reads without waiting),
     with the time the oldest / newest was read (lane_first_t / lane_last_t);
  2. the wait window (window_due): a round starts when the oldest waiting request is --cf-lane-wait old (default
     2 ms), or no new request came for --cf-lane-quiet (0.5 ms), or every lane active lately (a request in the last
     ACTIVE_S) is waiting; --cf-lane-wait 0: at once (as before);
  3. ONE call for all the waiting lanes: the lane GraphActor's smallest row bucket that holds them (--cf-lane-buckets,
     default the actor's --actor-row-buckets), each row's temporal slot / GRU table row = its lane, its uniforms
     keyed by the request (lane_uniforms: the request's sequence number and the lane, crn_uniforms; --cf-lane-ukeys 1),
     so an answer does not depend on which other lanes share the call or on its row in it;
  4. --cf-lane-async 1: the call (the branch begins' state copies, the reads of o_{t-1} / has before the step and of
     o_t after it: all on the lane actor's stream, through pinned buffers, one event) is only submitted; the actor
     thread goes on with the workers' records and answers the lanes when the event has completed (checked at every
     pass). One lane call in flight at a time; quiesce() before the actor's weights change (take_weights);
  5. the echo (B34's sequence numbers): LANE_OUT [4] gets the number read when the request was TAKEN, not at the
     answer: a search child killed while its call ran, whose lane already carries the next child's request, gets an
     answer that child skips (LanePolicy.answer); a lane whose GRU begin is pending at the answer (a new branch
     started there: the requester is gone) keeps the begin state.
No checkpoint or config change: the same model and actor; the lane answers have the policy's distribution as before
(the uniforms are counter-based instead of the graph's generator)."""
import time

import numpy as np
import torch

from .tok_policy import crn_uniforms, to_batch

ACTIVE_S = 0.05   # s a lane counts as active after its last request (the window's "every active lane is waiting")


def lane_uniforms(seqs, lanes, columns):
    """The sampling uniforms of lane requests (float32 [n, columns]): crn_uniforms keyed by each request's sequence
    number (LANE_IO [10]) with t = its lane: a request's draw is the same in whatever call it is answered."""
    return crn_uniforms(np.asarray(seqs, np.int64), np.asarray(lanes, np.int64), columns)


def window_due(n_ready, first_t, last_t, now, wait, quiet, active=None):
    """The wait window (pure): True when a round should start now. n_ready: distinct lanes waiting; first_t / last_t:
    when the oldest / newest request was read; wait: the window (<= 0: at once); quiet: no new request for that long
    ends it early (<= 0: off); active: the lanes active lately, the waiting ones among them (None: not used): all of
    them waiting ends it too."""
    if n_ready <= 0:
        return False
    if wait <= 0 or now - first_t >= wait:
        return True
    if quiet > 0 and now - last_t >= quiet:
        return True
    return active is not None and n_ready >= active


class LaneService:
    """The lanes' answers (module docstring). sampler: the TokSampler (lane_ready, lane_first_t / lane_last_t,
    take_lanes, poll_lanes, wait_lanes, reply_lanes, the lane blocks lane_rows / lane_io / lane_out, lane_h (memory),
    lane_t (temporal)); model: the actor's TokPolicy; actor: the lane GraphActor (None: the eager path, e.g. CPU);
    table: the lanes' GRU table (memory models); tstate: the lanes' TemporalState, bank: the choice points' snapshot
    bank, tdest: the lane steps' device rows (dict kv, s, o, obase) (temporal models); gtok: value_global; version():
    the actor's weights; stat: the dict the counters go to (calls, rows, s = actor-thread seconds, fly_s = submit to
    answer, held_s = the window's waits, eager = calls on the eager path)."""

    def __init__(self, sampler, model, device, nh, actor=None, table=None, tstate=None, bank=None, tdest=None,
                 gtok=False, wait=0.002, quiet=0.0005, ukeys=True, async_=True, version=None, stat=None):
        self.s, self.model, self.device, self.nh = sampler, model, torch.device(device), int(nh)
        self.actor, self.table, self.tstate, self.bank, self.tdest = actor, table, tstate, bank, tdest
        self.gtok = bool(gtok)
        self.wait, self.quiet = float(wait), float(quiet)
        self.ukeys = bool(ukeys)
        self.async_ = bool(async_) and actor is not None
        self.version = version if version is not None else (lambda: 0)
        self.stat = stat if stat is not None else {}
        for k in ('calls', 'rows', 's', 'fly_s', 'held_s', 'eager'):
            self.stat.setdefault(k, 0)
        n = len(sampler.lane_io)
        self.seen = np.full(n, -1e9)   # when each lane's last request was taken
        self.cols = int(sum(model.heads))
        self.fly = None
        if actor is not None:
            self._ev = torch.cuda.Event()
            if tstate is not None:   # (the reads around the step, on the lane stream: pinned buffers, a device index)
                D = tstate.D
                self._sl_pin = torch.zeros(n, dtype=torch.int64).pin_memory()
                self._sl = torch.zeros(n, dtype=torch.int64, device=self.device)
                self._pre_pin = torch.zeros((n, D + 1), dtype=torch.float32).pin_memory()
                self._post_pin = torch.zeros((n, D), dtype=torch.float32).pin_memory()

    # ---------------------------------------------------------------- the round's timing
    @property
    def in_flight(self):
        return self.fly is not None

    def idle(self, default):
        """The workers' poll wait while a lane call is in flight or a window is open (neither wakes the poll)."""
        if self.fly is not None:
            return min(default, 0.001)
        if self.s.lane_ready:
            return min(default, max(0.0, self.wait - (time.perf_counter() - self.s.lane_first_t)))
        return default

    def due(self, now=None):
        """The waiting requests' window has ended (window_due)."""
        ready = self.s.lane_ready
        if not ready:
            return False
        now = time.perf_counter() if now is None else now
        first_t = getattr(self.s, 'lane_first_t', 0.0)
        last_t = getattr(self.s, 'lane_last_t', 0.0)
        if window_due(1, first_t, last_t, now, self.wait, self.quiet):   # (the cheap tests first)
            return True
        rd = np.unique(np.asarray(ready, np.int64))
        recent = self.seen >= now - ACTIVE_S
        recent[rd] = True
        return window_due(len(rd), first_t, last_t, now, self.wait, self.quiet, int(recent.sum()))

    def tick(self):
        """One pass of the actor's loop (never waits): a finished call answered, a due round started (and answered at
        once without --cf-lane-async). True when something was done."""
        did = False
        if self.fly is not None:
            if not self.fly['ready'] and not self._ev.query():
                return False
            self.finish()
            did = True
        self.s.poll_lanes()   # (the newest arrivals: the window's quiet test needs them)
        if self.s.lane_ready and self.due():
            self.submit()
            if not self.async_ or self.fly['ready']:
                self.finish()
            did = True
        return did

    def serve_wait(self, timeout):
        """While the actor has nothing else to do (it waits for the learner): a call in flight answered, else up to
        timeout s for requests, the window waited out, one round served to its answer. True when something was done."""
        if self.fly is not None:
            self.finish()
            return True
        if not self.s.wait_lanes(timeout):
            return False
        while not self.due():
            k = len(self.s.lane_ready)
            if self.s.poll_lanes() == k:   # (nothing new: no busy loop beside the learner's thread)
                time.sleep(0.0002)
        self.submit()
        self.finish()
        return True

    def quiesce(self):
        """The call in flight answered (before the actor's weights change)."""
        if self.fly is not None:
            self.finish()

    # ---------------------------------------------------------------- a round
    def _begin(self, idx):
        """The branches' first requests: the lane's GRU state from the worker (memory, LanePolicy.begin), its temporal
        state from the point's snapshot or reset (LanePolicy.tbegin); on the current stream."""
        s, dev = self.s, self.device
        lane_h = getattr(s, 'lane_h', None) if self.table is not None else None
        if lane_h is not None:
            dim = lane_h.shape[1] - 1
            new_b = lane_h[idx, dim] > 0.5
            if new_b.any():
                j = idx[new_b]
                self.table.index_copy_(0, torch.from_numpy(np.ascontiguousarray(j)).to(dev),
                                       torch.from_numpy(np.ascontiguousarray(lane_h[j, :dim])).to(dev))
                lane_h[j, dim] = 0.0
        if self.tstate is not None and s.lane_io.shape[1] > 11:
            beg = s.lane_io[idx, 11].astype(np.int64)
            if (beg != 0).any():
                pos, neg = beg > 0, beg < 0
                if pos.any() and self.bank is not None:
                    self.tstate.copy_slot_(beg[pos] - 1, idx[pos], other=self.bank)
                elif pos.any():
                    self.tstate.reset_slot_(idx[pos])
                if neg.any():
                    self.tstate.reset_slot_(idx[neg])
                s.lane_io[idx[beg != 0], 11] = 0

    def submit(self):
        """Take every waiting request and start one call for them. True when there were any."""
        if self.fly is not None:   # (one call in flight at a time)
            self.finish()
        s = self.s
        idx = s.take_lanes()
        if not len(idx):
            return False
        t0 = time.perf_counter()
        n = len(idx)
        self.stat['held_s'] += max(0.0, t0 - getattr(s, 'lane_first_t', t0))
        self.seen[idx] = t0
        rows = s.lane_rows[idx]
        pend = s.lane_io[idx, :self.nh].astype(np.int64)
        seq = s.lane_io[idx, 10].astype(np.int64) if s.lane_io.shape[1] > 10 else np.zeros(n, np.int64)
        ukeys = (seq, idx) if self.ukeys else None
        f = dict(idx=idx, rows=rows, seq=seq, version=self.version(), t0=t0, ready=False)
        self.fly = f
        got = False
        if self.actor is not None:
            cur = torch.cuda.current_stream()
            st = self.actor.stream if self.actor.stream is not None else cur
            with torch.cuda.stream(st):
                st.wait_stream(cur)   # (e.g. the choice points' snapshots copied into the bank on the actor's stream)
                self._begin(idx)
                if self.tstate is not None:   # o_{t-1} and has before the step
                    self._sl_pin.numpy()[:n] = idx
                    sl = self._sl[:n]
                    sl.copy_(self._sl_pin[:n], non_blocking=True)
                    pre = torch.cat([self.tstate.prev_o.index_select(0, sl).float(),
                                     (self.tstate.steps.index_select(0, sl) > 0).float()[:, None]], 1)
                    self._pre_pin[:n].copy_(pre, non_blocking=True)
            t_on, m_on = self.tstate is not None, self.table is not None
            got = self.actor.submit(rows, pend, widx=idx if (t_on or m_on) else None, reset=None,
                                    tdest=dict(self.tdest, rows=idx) if t_on else None, ukeys=ukeys)
            with torch.cuda.stream(st):
                if got and t_on:   # o_t after it
                    o_rows = self._sl[:n] + int(self.tdest.get('obase', 0))
                    self._post_pin[:n].copy_(self.tdest['o'].index_select(0, o_rows).float(), non_blocking=True)
                self._ev.record(st)
            if not got:   # (more lanes than the graphs hold: the eager path, after the reads above)
                st.synchronize()
                if t_on:
                    pre = self._pre_pin[:n].numpy()
                    f['op'], f['has'] = pre[:, :-1].copy(), (pre[:, -1] > 0.5) & (rows['first'] == 0)
        else:
            self._begin(idx)
        if not got:
            self._eager(f, rows, pend, idx, ukeys)
            f['ready'] = True
            self.stat['eager'] += 1
        self.stat['s'] += time.perf_counter() - t0
        return True

    def _eager(self, f, rows, pend, idx, ukeys):
        """The call on the eager path (TokPolicy.act with the lanes' uniforms), its outputs into f."""
        dev = self.device
        sl = torch.from_numpy(np.ascontiguousarray(idx, np.int64)).to(dev)
        tk = {}
        if self.tstate is not None:
            if 'op' not in f:
                f['op'] = self.tstate.prev_o.index_select(0, sl).float().cpu().numpy()
                f['has'] = (self.tstate.steps.index_select(0, sl) > 0).cpu().numpy() & (rows['first'] == 0)
            tk = dict(temporal=dict(mode='step', state=self.tstate, slot=sl))
        u = torch.from_numpy(lane_uniforms(ukeys[0], ukeys[1], self.cols)).to(dev) if ukeys is not None else None
        h_in = self.table.index_select(0, sl) if self.table is not None else None
        with torch.inference_mode():
            b = to_batch(rows, pend, dev)
            if h_in is not None:
                out = self.model.act(b, u=u, mem=h_in, need_vg=self.gtok, **tk)
            else:
                out = self.model.act(b, u=u, need_vg=self.gtok, **tk)
        if h_in is not None:
            self.table.index_copy_(0, sl, out[3])
            f['h'] = h_in.cpu().numpy()
        f['a'], f['lp'], f['v'] = (x.cpu().numpy() for x in out[:3])
        f['vg'] = out[-1].float().cpu().numpy() if self.gtok else None
        if tk:
            f['o'] = tk['temporal']['out']['o'].float().cpu().numpy()

    def finish(self):
        """The call in flight (waited for), its answers into the lanes' blocks, the lanes woken."""
        f = self.fly
        if f is None:
            return False
        self.fly = None
        t0 = time.perf_counter()
        s, idx = self.s, f['idx']
        n = len(idx)
        if not f['ready']:
            self._ev.synchronize()
            a, lp, v = self.actor.result()
            h = self.actor.h_used if self.table is not None else None
            vg = self.actor.vg if self.gtok else None
            if self.tstate is not None:
                pre = self._pre_pin[:n].numpy()
                op, has = pre[:, :-1].copy(), (pre[:, -1] > 0.5) & (f['rows']['first'] == 0)
                o = self._post_pin[:n].numpy().copy()
        else:
            a, lp, v, h, vg = f['a'], f['lp'], f['v'], f.get('h'), f['vg']
            op, has, o = f.get('op'), f.get('has'), f.get('o')
        lane_h = getattr(s, 'lane_h', None) if self.table is not None else None
        if lane_h is not None and h is not None:   # the input state of each answer (imitation records); a lane whose
            dim = lane_h.shape[1] - 1              # begin is pending (a new branch: the requester is gone) keeps it
            keep = lane_h[idx, dim] <= 0.5
            lane_h[idx[keep], :dim] = h[keep]
        if self.tstate is not None and getattr(s, 'lane_t', None) is not None:   # the lane step's outputs
            from .tok_cf import t_pack
            s.lane_t[idx] = t_pack(op, has, o)
        s.lane_io[idx, 5:5 + self.nh] = a
        s.lane_out[idx, 0], s.lane_out[idx, 1], s.lane_out[idx, 2] = v, lp, f['version']
        if s.lane_out.shape[1] > 3:
            s.lane_out[idx, 3] = v if vg is None else vg   # value_global (the value without one)
        if s.lane_out.shape[1] > 4:
            s.lane_out[idx, 4] = f['seq']   # the requests' numbers as taken, written last
        s.reply_lanes(idx)
        t1 = time.perf_counter()
        self.stat['calls'] += 1
        self.stat['rows'] += n
        self.stat['s'] += t1 - t0
        self.stat['fly_s'] += t1 - f['t0']
        return True
