"""Truncated memory training (train_tok --memory-train truncated, 2026-10-30).

Inference is unchanged: the actor's Reader reads the whole episode (the most recent W source values u and every
closed Writer block). Training: a loss reaches only the frames and Writer blocks of the rollout being trained, per
worker. Older values are the actor's own numbers, constants (stop-gradient):

  - the source values u of the W decisions before a worker's rollout: the rollout's carried context rows (P = W rows
    per worker, Rollout.carry), t_s holding exactly the u the actor wrote;
  - the Writer tokens of the blocks the episode closed before the rollout: LearnerCarry, built from the rollouts
    themselves (t_m: the tokens of the block a decision closes, as the actor computed them).

A worker's rollout is its decisions 0 .. length - 1; its open last decision (reward still to come) is decision 0 of
the next rollout, so the context of the next rollout ends right before it (LearnerCarry.advance).
"""
import numpy as np
import torch


class LearnerCarry:
    """Per worker the current episode's Writer tokens of the blocks closed before the next rollout to train: the
    actor's values (float32 [blocks * tokens, D] on the learner's device), the episode serial (Rollout.t_ep) they
    belong to, and how many blocks. Audit counters: mismatches (a context that could not be rebuilt exactly; must stay
    0), constants with a gradient (must stay 0)."""

    def __init__(self, workers, block, tokens, width, device):
        self.block, self.tokens, self.D = int(block), int(tokens), int(width)
        self.device = torch.device(device)
        self.ep = np.full(workers, -10, np.int64)
        self.blocks = np.zeros(workers, np.int64)
        self.mem = [None] * workers
        self.mismatch = 0

    def _empty(self):
        return torch.zeros((0, self.D), dtype=torch.float32, device=self.device)

    def memory(self, w, ep, t0):
        """The constant Writer tokens [t0 // block * tokens, D] of worker w's episode ep before its decision at episode
        index t0 (None counted as a mismatch when the carry does not hold exactly those blocks)."""
        need = t0 // self.block
        if need == 0:
            return self._empty()
        if self.ep[w] != ep or self.blocks[w] < need:
            self.mismatch += 1
            return None
        return self.mem[w][:need * self.tokens]

    def advance(self, rl):
        """After training on rollout rl: every worker's blocks closed by its decisions before the open one (or all of
        them when none is open) appended (their episode's tokens replace another episode's)."""
        P, B = rl.tP, self.block
        rows_all, ws = [], []
        for w in range(rl.n):
            n = int(rl.open[w]) if rl.open[w] >= 0 else int(rl.length[w])
            if n <= 0:
                continue
            ep = rl.t_ep[w, P:P + n]
            t = rl.t_step[w, P:P + n] - 1
            # Only the last episode in the rollout goes on: the carry continues it or starts it afresh (an episode the
            # carry never saw from its start is a mismatch only when one of its blocks is needed: memory())
            at = int(np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])[-1])
            if ep[at] != self.ep[w]:
                self.ep[w], self.blocks[w], self.mem[w] = ep[at], 0, None
                if t[at] != 0:
                    self.blocks[w] = -1   # (started mid-episode: its earlier blocks are unknown)
            close = at + np.flatnonzero((t[at:] + 1) % B == 0)
            if not len(close):
                continue
            if self.blocks[w] < 0 or self.blocks[w] != (t[close[0]] + 1) // B - 1:
                self.blocks[w] = -1   # (a gap: never served)
                continue
            rows_all.append(rl.t_row[w, P + close])
            ws.append((w, len(close)))
            self.blocks[w] += len(close)
        if not rows_all:
            return
        rows = torch.from_numpy(np.concatenate(rows_all)).to(rl.t_m.device)
        tokens = rl.t_m.index_select(0, rows).to(self.device, torch.float32)
        at = 0
        for w, k in ws:
            new = tokens[at:at + k].reshape(k * self.tokens, self.D)
            at += k
            self.mem[w] = new if self.mem[w] is None else torch.cat([self.mem[w], new])


def worker_context(rl, w, window):
    """Worker w's carried context of rollout rl: the episode serial and index of its decision 0 and the constant
    source values before it (float32 [m, D], m = min(t0, window), episode indices t0 - m .. t0 - 1), from the carried
    rows. Returns (ep0, t0, values, ok): ok False when the carried rows do not hold exactly those decisions."""
    P = rl.tP
    ep0, t0 = int(rl.t_ep[w, P]), int(rl.t_step[w, P]) - 1
    m = min(t0, window)
    if m == 0:
        return ep0, t0, rl.t_s.new_zeros((0, rl.t_s.shape[1])), True
    ep, t, rows = rl.t_ep[w, P - m:P], rl.t_step[w, P - m:P] - 1, rl.t_row[w, P - m:P]
    ok = P >= m and bool((ep == ep0).all()) and bool((t == np.arange(t0 - m, t0)).all()) and bool((rows >= 0).all())
    if not ok:
        return ep0, t0, None, False
    return ep0, t0, rl.t_s.index_select(0, torch.from_numpy(rows).to(rl.t_s.device)).float(), True
