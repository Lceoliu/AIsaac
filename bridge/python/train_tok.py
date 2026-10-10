"""Training of the token policy on the fork sampler (EXPERIMENTS.md C55 on): PPO, one policy for every group.

One process owns the GPU. Its main thread is the actor: it answers the workers' records (isaac_bridge/tok_sampler.py)
with sampled actions (GraphActor, the actor's own copy of the weights) and stores the decisions in a rollout; workers
advance at their own pace, so each has its own number of decisions. A learner thread does the update when a rollout
holds `--rollout` decisions: generalised advantage estimates per worker, then `--epochs` passes of clipped PPO over
the stored decisions in minibatches, on the learner's copy of the weights; the new weights are then copied into the
actor's (between two of its calls). Two modes:
  synchronous (default)  the actor waits while the learner updates (the workers finish the step they are in and wait;
                         queued hindsight searches run then) and continues with the new weights: plain PPO, as before.
  --overlap              the actor keeps answering the workers with the weights it has while the learner updates, and
                         takes the new weights when the update is done. The decisions collected meanwhile belong to the
                         next rollout and were sampled with the weights before that update: one update of policy lag
                         for that part of a rollout (at most). Their stored log-probabilities are the behaviour
                         policy's (the one that sampled them), which is what PPO's ratio pi_new / pi_behaviour needs:
                         the clipped objective is then taken around the behaviour policy (its share of the rollout is
                         the column stale_share, the update is on the newest weights as always). The values that the
                         advantages need are recomputed with the learner's weights at the start of each update
                         (--recompute-values, on by default with --overlap), so the advantages are what the
                         synchronous mode computes from the same decisions. If a rollout is full before the update is
                         done, the actor waits for it (the workers too; wait_learner_s): the lag never exceeds one
                         update. The hindsight teacher (--teacher) then gets its time from --teacher-share (default
                         0.15 with --overlap), since the workers no longer pause during updates.
                         With the plain ratio the stale part starts the update with ratios away from 1 and is clipped
                         earlier (higher clip_frac, slower entropy decrease); --decoupled 1 clips around the weights
                         at the update's start instead and weights by pi_start / pi_behaviour (see its help).
Other throughput settings (same results, other speed): --actor-slots (calls in flight), --poll-want / --poll-wait
(batching of records), --sort-micro (minibatch rows ordered by entity count before the --micro parts), CUDA streams
(the actor's calls on high-priority streams, the learner's kernels on a low-priority one), --fused-adam; and for the
learner (2026-10-04, on by default, the same gradient up to summation order): --packed (the Transformer's per-token
layers on the real tokens only), --teach-merge (a step's teacher records join its last PPO part: one forward and
backward instead of two), --teach-gpu (the teacher records also held on the GPU), the minibatches planned on the host
(no wait for the GPU inside an epoch). ISAAC_RL_PROFILE=1 times the update's parts (u_* columns) with GPU syncs.
Instance defaults (2026-10-05, tok_sampler "Instance defaults"; each proven exact): stub list stub_render_h.txt
(--stub-list <file> for another), ABP_FAST=3, ISAAC_RL_PU_SKIP=1, ABP_FORK_LITE=1 (a variable already set in the
environment wins, e.g. ABP_FAST=0), --teacher-pair 1; --build-timeout (60 s, 120 before 2026-10-06) limits a worker's
first start-state build after a launch, --reset-timeout (20 s, 2026-10-06) a later one; a stalled or failed build is
retried once with the same seed on the relaunched instance. --root-anon-mib replaces a root above that RssAnon at a state
boundary (2026-10-06, off by default).
Mixed precision: --amp bf16 runs the learner's forward passes under bfloat16 autocast (the heads, the losses and the
input MLPs over raw features stay float32; the actor and the stored log-probabilities stay float32): close to, not
exactly, the float32 update.
Actions take effect one decision late (tok_sampler): the action under way is part of the state the policy and the
value see, and a stored decision's reward is what happens during the step after it.

Reward of a decision (the terms the record carries; no shaping beyond them):
  + damage   x the share of the room's starting monster HP removed
  - hurt     x half hearts lost
  + clear    when the room is cleared;  - death when the player dies;  - timeout at the time limit
  - time     every decision
Whole floors (--mode floor, tok_floor.py; episodes run through the doors to the trapdoor after the boss): instead of
clear, + room for a room with monsters cleared, + explore for a room entered for the first time, + boss on top for the
boss room, + exit when the next floor is reached; the minimap and the doors' "seen" flags are part of the input.
--init starts from another run's weights (a room-mode run's: the door input's new columns start at zero).
Phase A (2026-10-06, both off by default; nothing else changes when they are off):
  --items        the item-aware observation and the item / pill heads (TokPolicy items, tok_obs ROW's item fields;
                 the workers' roots run the bridge's lean_items). --init from an items-free checkpoint pads the new
                 inputs with zeros (load_compatible); --resume of an items checkpoint turns it on by itself.
  --mode run     whole runs from Basement I (tok_floor with run = True): the floor's exit is an event (EV_EXIT, reward
                 exit) and the episode goes on on the next floor until death, --run-seconds of game time, or a floor's
                 own --floor-stall-seconds without progress. Rewards as floor mode. Counters: progress.csv
                 run/stage_max (deepest stage per episode, mean), run/exits, items_taken, active_uses, pills_used (and
                 per episode in episodes.jsonl: stage0, stage, exits, items, uses, pills); with --items in floor mode the
                 item counters too.
Counterfactual branches (2026-10-07, isaac_bridge/tok_branch.py; floor / run modes, off unless --branch-share > 0):
  at decision points (an item taken or left behind, a door taken while others lead to unvisited rooms) the workers
  restore the state, run each option as a branch episode with the current policy (records tagged ROW point / branch,
  actions from the actor) from common random numbers, and write a choice record (the record before the decision, the
  options' outcomes) that lands in a host ring (L['choice_*']), choices.jsonl and choices-last.npz. A branch episode
  ending for its own reasons (cap, floor exit, boss cleared) sends done DONE_BRANCH: that record is evaluated and kept
  as a boot decision (Rollout.boot) the decision before bootstraps from; it is never trained on. --branch-ppo 0 keeps
  all branch decisions out of PPO. Branch episodes go to branch_episodes.jsonl, not to the episode statistics. The
  PPO update is unchanged by the choice records unless --choice-adv-coef > 0 (Phase B2 below).
  Phase B2 (2026-10-07; tok_branch's second docstring): the branches of a pair share the policy's sampling noise
  (ROW crn, GraphActor) and are answered with a weight snapshot the point pinned (--branch-snap-every; the actor keeps
  two); a branch is scored as prefix reward + gamma^prefix (discounted return + gamma^T V(end)) with 60 s branches;
  pairs are replicated (--branch-noise unconditionally for the noise floor, --branch-replicate-max while the label is
  ambiguous). The choice heads (TokPolicy choice, K = --choice-heads) learn D = score(take) - score(skip) from the
  records' pairs on detached player tokens with an optimiser of their own after each PPO update (train_choice; held-out
  points: --choice-holdout), the trainer pools the noise variance per kind for the workers (aux block), the actor
  hands the heads' spread at every record to the workers (the point priority's uncertainty, --branch-uncert), and
  --choice-adv-coef > 0 adds a PPO-clipped correction with the measured own - alternative difference on the choice
  records' decision records. With --branch-share 0 none of this exists and the run is as before.
The last stored decision of a worker has no reward yet when the rollout ends: it is kept out of the update, its value
bootstraps the one before, and it opens the next rollout.

progress.csv, besides the learning columns: x_real_time = game time trained per wall second (decisions of the update x
frames per decision / 30 over the wall time since the update before ended); collect_s = the actor's time for the
rollout, update_s = the learner's; infer_ms = a network call's time from its start to its result (infer_upd_ms: calls
made while the learner was working); p_* = where the actor loop's time went (poll = waiting for records, book =
rewards and episode bookkeeping, act = starting calls and waiting for their results, reply = actions out, store = the
rollout; with --actor-slots 2 a call runs on the GPU while the loop reads and books the next records, so infer_ms can
exceed the loop's own act time per call); idle_share = the workers' time spent waiting for answers over their wall
time; w_* = the workers' seconds (summed) in stepping, waiting, encoding, forking and teacher searches over the same
span; stale_share = share of the update's decisions sampled with weights older than the update's start; ent_tokens =
entity tokens per micro part. Added 2026-10-04: cpu_collect / cpu_update = busy share of all the machine's CPUs during
the collect / the update (/proc/stat); c_trainer_cores = CPUs this process used over the cycle; w_overrun = worker
seconds of searches that ran past the end of the synchronous hold; q_teach = hurts queued for a search; w_state /
w_close / w_total = worker seconds taking up start states / closing episodes / in all; c_dec_min / c_dec_max / dec_w =
decisions per worker in the cycle (a worker at 0 is stuck); u_* = seconds of the update's parts (see ISAAC_RL_PROFILE).

usage (from the bridge's python dir, PYTHONPATH=.):
  python train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --groups normal:8,boss:5,normal_big:3 \
      --out <dir> [--game-hours 2000] [--resume <checkpoint.pt>] [--overlap]
"""
import argparse
import collections
import csv
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

from goexplore_abplus import default_bridge_lua, default_preload, load_spec
from isaac_bridge.tok_obs import DONE_BRANCH, EV_ACTIVE, EV_EXIT, EV_ITEM, EV_PILL, ROW
from isaac_bridge.tok_sampler import (ACTION_F, ST, TokSampler, TokSamplerConfig, apply_instance_defaults,
                                      default_stub_list)


ROOM_REWARD = 'damage=1,hurt=0.75,clear=1,death=2,timeout=1,time=0.002'
# whole floors: room = a room with monsters cleared, explore = a room entered for the first time, boss = the boss
# room cleared (on top of room), exit = the next floor reached
FLOOR_REWARD = 'damage=1,hurt=0.75,room=1,explore=0.5,boss=2,exit=3,death=2,timeout=1,time=0.002'
# 2026-10-08 (SCALING_THESIS.md §3, off by default): heal = per half heart of health units gained between two records
# (red + soul + black + eternal + 2 x bone, the player token's units column; gains only, so a hurt stays the hurt term),
# resource = per coin / bomb / key gained (the player token's columns, capped there at 50 / 10 / 10); e.g.
# --reward heal=0.5,resource=0.25 (Episodes.cur_gains; floor / run modes only)
OVERLAP_TEACHER_SHARE = 0.15
# ISAAC_RL_PROFILE=1: the learner synchronises the GPU between the parts of an update and logs their seconds (u_*
# columns); a measurement aid that costs a little speed. Without it the u_* columns hold only host-side times.
PROFILE = os.environ.get('ISAAC_RL_PROFILE', '0') == '1'


def cpu_ticks():
    """(busy, total) jiffies of all CPUs from /proc/stat ((0, 0) where there is none)."""
    try:
        with open('/proc/stat') as f:
            v = [int(x) for x in f.readline().split()[1:]]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return sum(v) - idle, sum(v)
    except (OSError, ValueError, IndexError):
        return 0, 0


def cpu_share(a, b):
    """Busy share of all CPUs between two cpu_ticks() readings."""
    return (b[0] - a[0]) / max(b[1] - a[1], 1)


def parse_groups(text, workers):
    """'normal:8,boss:5' -> ([name], [worker -> group index]); the counts are scaled to the number of workers."""
    names, counts = [], []
    for part in text.split(','):
        name, _, count = part.partition(':')
        names.append(name)
        counts.append(float(count or 1))
    share = np.array(counts) / sum(counts)
    assign = []
    for w in range(workers):   # largest remainder, spread over the worker indices
        have = np.bincount(assign, minlength=len(names)) if assign else np.zeros(len(names))
        assign.append(int(np.argmax(share * (w + 1) - have)))
    return names, assign


class Rollout:
    """Decisions per worker, in arrival order."""

    def __init__(self, workers, capacity, n_heads=3):
        self.n, self.cap, self.nh = workers, capacity, n_heads
        self.rows = np.zeros((workers, capacity), ROW)
        self.pending = np.zeros((workers, capacity, n_heads), np.int64)
        self.actions = np.zeros((workers, capacity, n_heads), np.int64)
        self.logp = np.zeros((workers, capacity), np.float32)
        self.value = np.zeros((workers, capacity), np.float32)
        self.reward = np.zeros((workers, capacity), np.float32)
        self.terminal = np.zeros((workers, capacity), bool)
        self.version = np.zeros((workers, capacity), np.int64)   # the actor's weights (updates done) that sampled it
        # 2026-10-07 (branches): a branch episode's truncated last record (done DONE_BRANCH), stored only for its value:
        # the decision before bootstraps from it, it is not trained on (gae)
        self.boot = np.zeros((workers, capacity), bool)
        self.length = np.zeros(workers, np.int64)
        self.open = np.full(workers, -1, np.int64)   # the stored decision still waiting for its reward (the last one)

    def total(self):
        return int(self.length.sum())

    def usable(self):
        """Decisions that have their reward (an open decision is always a worker's last one)."""
        return int(self.length.sum() - (self.open >= 0).sum())

    def reset(self):
        self.length[:] = 0
        self.open[:] = -1

    def carry(self, other):
        """Start this (empty) rollout with the other's open decisions (their rewards are still to come)."""
        self.reset()
        i = np.flatnonzero(other.open >= 0)
        k = other.open[i]
        for name in ('rows', 'pending', 'actions', 'logp', 'value', 'version', 'boot'):
            getattr(self, name)[i, 0] = getattr(other, name)[i, k]
        self.length[i], self.open[i] = 1, 0


class Episodes:
    """The actor's bookkeeping of the records that arrive (vectorised over a poll's records): the reward of each
    worker's open decision, episode ends and their statistics, the action under way."""

    def __init__(self, workers, rw, floor, run=False, items=False):
        self.rw, self.floor, self.run, self.items = rw, floor or run, run, items
        self.end_bonus = rw['exit'] if self.floor else rw['clear']
        self.ret, self.hurt, self.damage = np.zeros(workers), np.zeros(workers), np.zeros(workers)
        self.rooms, self.boss = np.zeros(workers, np.int64), np.zeros(workers, np.int64)
        # run / items (2026-10-06): floors reached, collectibles gained, active item and pill / card uses, the stage
        # at the episode's first record and the deepest one
        self.extra = np.zeros((workers, 6), np.int64)   # exits, items, uses, pills, stage0, stage_max
        self.finished = []                       # episodes ended since the last hand-over
        self.branch_finished = []                # 2026-10-07: branch episodes ended since then (tok_branch)
        self.lab_finished = []                   # 2026-10-08: lab episodes ended since then (tok_lab)
        self.seen_inv = None                     # 2026-10-08 (lab on): held collectibles at ordinary episodes' ends
        self.inv0 = {}                           # 2026-10-08: worker -> collectibles held at the episode's first record
        self.stats0 = None                       # 2026-10-08 (stat aug on): worker -> the stats at the first record
        self.char0 = None                        # 2026-10-08 (characters on): worker -> ROW pchar at the first record
        # 2026-10-08 (heal / resource terms, floor modes): per worker the health units and coins + bombs + keys of the
        # last record (the deltas' base) and the episode's gains so far
        self.gains = np.zeros((workers, 2)) if self.floor and (rw.get('heal') or rw.get('resource')) else None
        self.healed, self.gathered = np.zeros(workers), np.zeros(workers)

    @staticmethod
    def cur_gains(rows):
        """[health units, coins + bombs + keys] of each record, from the player token (tok_obs / abp_row_encode
        columns 7 units / 12, 8 min(bombs, 10) / 10, 9 min(keys, 10) / 10, 10 min(coins, 50) / 50)."""
        pl = rows['player']
        return np.stack([np.rint(pl[:, 7] * 12.0), np.rint(pl[:, 8] * 10.0) + np.rint(pl[:, 9] * 10.0)
                         + np.rint(pl[:, 10] * 50.0)], 1).astype(np.float64)

    def arrive(self, roll, rows, idx, pending):
        """Records `rows` of workers `idx` (distinct): close the decisions before them, end episodes, reset the action
        under way where an episode starts or ends. Returns the mask of the records that need an action."""
        rw = self.rw
        first = rows['first'] != 0
        done = rows['done'].astype(np.int64)
        k = roll.open[idx]
        broke = first & (k >= 0)   # the episode before broke off (a worker error): end it there
        if broke.any():
            roll.reward[idx[broke], k[broke]], roll.terminal[idx[broke], k[broke]] = 0.0, True
            roll.open[idx[broke]] = -1
            k = np.where(broke, -1, k)
        has = k >= 0
        if has.any():   # the reward of the decision before
            w, kk, d = idx[has], k[has], done[has]
            dmg = rows['damage'][has].astype(np.float64)
            hu = rows['hurt'][has].astype(np.float64)
            reward = rw['damage'] * dmg - rw['hurt'] * hu - rw['time'] + np.where(
                d == 1, self.end_bonus, np.where(d == 2, -rw['death'], np.where(d == 3, -rw['timeout'], 0.0)))
            if self.floor:
                ev = rows['events'][has].astype(np.int64)
                reward = reward + (rw['room'] * ((ev & 1) > 0) + rw['explore'] * ((ev & 2) > 0)
                                   + rw['boss'] * ((ev & 4) > 0))
                if self.run:   # the next floor reached: the exit reward, and the episode goes on
                    reward = reward + rw['exit'] * ((ev & EV_EXIT) > 0)
                self.rooms[w] += (ev & 1) > 0
                self.boss[w] += (ev & 4) > 0
                if self.gains is not None:   # 2026-10-08: health units and resources gained since the record before
                    dl = self.cur_gains(rows)[has] - self.gains[w]
                    reward = reward + rw['heal'] * np.maximum(dl[:, 0], 0) + rw['resource'] * np.maximum(dl[:, 1], 0)
                    self.healed[w] += np.maximum(dl[:, 0], 0)
                    self.gathered[w] += np.maximum(dl[:, 1], 0)
            if self.run or self.items:
                ev = rows['events'][has].astype(np.int64)
                x = self.extra
                x[w, 0] += (ev & EV_EXIT) > 0
                x[w, 1] += (ev & EV_ITEM) > 0
                x[w, 2] += (ev & EV_ACTIVE) > 0
                x[w, 3] += (ev & EV_PILL) > 0
                x[w, 5] = np.maximum(x[w, 5], rows['stage'][has])
            # (a branch's truncated end, DONE_BRANCH: not terminal, the decision bootstraps from the end's value)
            roll.reward[w, kk], roll.terminal[w, kk] = reward, (d != 0) & (d != DONE_BRANCH)
            roll.open[w] = -1
            self.ret[w] += reward
            self.hurt[w] += hu
            self.damage[w] += dmg
        if self.gains is not None:   # every record (the first of an episode included): the next deltas' base
            self.gains[idx] = self.cur_gains(rows)
        if self.run or self.items:   # an episode's first record: its stage, and the build it starts with
            st0 = first
            if st0.any():
                self.extra[idx[st0], 4] = rows['stage'][st0]
                self.extra[idx[st0], 5] = rows['stage'][st0]
                if self.items:
                    for j in np.flatnonzero(st0):
                        self.inv0[int(idx[j])] = [int(v) for v, c in rows[j]['inv'] if v > 0 and c > 0]
                if self.stats0 is not None:   # 2026-10-08 (stat aug): damage, fire delay, shot speed, range, speed
                    for j in np.flatnonzero(st0):
                        pl = rows[j]['player']
                        self.stats0[int(idx[j])] = [round(float(pl[11]) * 10, 3), round(float(pl[12]) * 20, 3),
                                                    round(float(pl[13]) * 2, 3), round(float(pl[14]) * 500, 1),
                                                    round(float(pl[15]) * 2, 3)]
        if self.char0 is not None and first.any():   # 2026-10-08 (characters): the episode's character
            for j in np.flatnonzero(first):
                self.char0[int(idx[j])] = int(rows[j]['pchar'])
        ended = done != 0
        if ended.any():
            for j in np.flatnonzero(ended):
                i, r = idx[j], rows[j]
                fin = dict(group=int(r['group']), seed=int(r['seed']), done=int(r['done']),
                           decisions=int(r['t']), ret=float(self.ret[i]), hurt=float(self.hurt[i]),
                           damage=float(self.damage[i]), rooms=int(self.rooms[i]), boss=int(self.boss[i]))
                if self.run or self.items:
                    x = self.extra[i]
                    fin.update(exits=int(x[0]), items=int(x[1]), uses=int(x[2]), pills=int(x[3]),
                               stage0=int(x[4]), stage=int(max(x[5], r['stage'])))
                    if self.inv0.get(int(i)):   # 2026-10-08: the collectibles held at the first record (start builds)
                        fin['build0'] = self.inv0[int(i)]
                if self.stats0 is not None and int(i) in self.stats0:   # 2026-10-08 (stat aug): the start stats
                    fin['stats0'] = self.stats0[int(i)]
                if self.char0 is not None and int(i) in self.char0:   # 2026-10-08 (characters): its PlayerType
                    fin['char0'] = self.char0[int(i)]
                if self.gains is not None:   # 2026-10-08: health units healed, coins + bombs + keys gathered
                    fin.update(healed=float(self.healed[i]), gathered=float(self.gathered[i]))
                if r['lab'] != 0:   # 2026-10-08: a lab episode (tok_lab): kept apart from the episode stats
                    fin.update(lab=int(r['lab']), lab_s=int(r['lab_s']))
                    self.lab_finished.append(fin)
                elif r['branch'] != 0:   # 2026-10-07: a branch episode (tok_branch): kept apart from the episode stats
                    fin.update(point=int(r['point']), branch=int(r['branch']))
                    self.branch_finished.append(fin)
                else:
                    self.finished.append(fin)
                    if self.seen_inv is not None and self.items:   # the build it ended with
                        held = [int(v) for v, c in r['inv'] if v > 0 and c > 0]
                        if held:
                            self.seen_inv.append(held)
            e = idx[ended]
            self.ret[e], self.hurt[e], self.damage[e], self.rooms[e], self.boss[e] = 0.0, 0.0, 0.0, 0, 0
            self.extra[e] = 0
            self.healed[e], self.gathered[e] = 0.0, 0.0
            boot = done == DONE_BRANCH
            pending[idx[ended & ~boot]] = 0   # (a truncated branch end keeps it: its record is evaluated, see below)
        else:
            boot = np.zeros(len(idx), bool)
        # a branch episode's truncated end (DONE_BRANCH, 2026-10-07) is answered like a running record: the actor's
        # value of it is stored (Rollout.boot) for the decision before to bootstrap from; the next record of the
        # worker is the next episode's first, which closes it with no reward as a terminal (broke, above)
        go = ~ended | boot
        pending[idx[first & go]] = 0
        return go


def store(roll, a_idx, rows, under_way, actions, logp, value, version):
    """Store the decisions just answered (workers a_idx, distinct) as their workers' open ones."""
    kk = roll.length[a_idx]
    ok = kk < roll.cap   # cannot fail with the trainer's capacity; drop rather than overrun
    if not ok.all():
        a_idx, kk, rows, under_way = a_idx[ok], kk[ok], rows[ok], under_way[ok]
        actions, logp, value = actions[ok], logp[ok], value[ok]
    roll.rows[a_idx, kk] = rows
    roll.boot[a_idx, kk] = rows['done'] == DONE_BRANCH   # 2026-10-07 (branches; always False without them)
    roll.pending[a_idx, kk], roll.actions[a_idx, kk] = under_way, actions
    roll.logp[a_idx, kk], roll.value[a_idx, kk], roll.version[a_idx, kk] = logp, value, version
    roll.length[a_idx], roll.open[a_idx] = kk + 1, kk


def gae(rollout, gamma, lam, value=None):
    """(advantages, returns, mask) [workers, capacity] over each worker's decisions that have their reward. value:
    the values to use (default: the stored ones)."""
    value = rollout.value if value is None else value
    adv = np.zeros_like(rollout.reward)
    mask = np.zeros_like(rollout.terminal)
    for i in range(rollout.n):
        last = rollout.length[i] - 1
        if last < 0:
            continue
        usable = last if rollout.open[i] == last else last + 1   # an open decision only bootstraps
        running, nxt = 0.0, value[i, last] if rollout.open[i] == last else 0.0
        boot = rollout.boot[i]
        for t in range(usable - 1, -1, -1):
            if boot[t]:   # a truncated branch end (2026-10-07): only its value, for the decision before; not trained
                running, nxt = 0.0, value[i, t]
                continue
            live = 0.0 if rollout.terminal[i, t] else 1.0
            delta = rollout.reward[i, t] + gamma * nxt * live - value[i, t]
            running = delta + gamma * lam * live * running
            adv[i, t] = running
            nxt = value[i, t]
        mask[i, :usable] = True
        mask[i, :usable] &= ~boot[:usable]
    return adv, adv + value, mask


def main():
    import signal
    # started in the background of a non-interactive shell, SIGINT arrives ignored: both signals end the run
    # through the finally block below (last.pt is saved, the sampler closed)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    # GPU sharing hazard (2026-10-04): on a GPU time-shared with another CUDA process (a second trainer, eval_tok) this
    # process died now and then with "CUDA error: an illegal memory access" (Xid 31, GPCCLIENT_GCC FAULT_PDE read). The
    # faulting address lies in the OTHER process's address space (checked against its /proc/<pid>/maps): state of the
    # other context used by ours, a driver / GPU context-switch fault, not a buffer of ours (535 / 3090 and
    # 580 / 3080 Ti). It needs the other process (alone on the GPU: no crash in 754 overlap updates) and the actor's
    # replayed CUDA graphs (eager actor, --graph-entities 0: none in 461 shared updates, at about half the throughput);
    # --overlap makes it far more frequent than synchronous runs. Mitigation, not a cure: one hardware work queue for
    # this process's streams (read when the context is created, so set before torch touches the GPU; an explicit
    # CUDA_DEVICE_MAX_CONNECTIONS in the environment wins) cut the crash rate of the shared stress test about tenfold.
    # The cure is not sharing the GPU with another CUDA process while training (or NVIDIA MPS).
    os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
    # torch is imported here, not at the top: the sampler's workers are spawned and import this module too
    import torch
    from isaac_bridge.tok_obs import ENT_CAP
    from isaac_bridge.tok_policy import (ENTITY_FIELDS, ROW_BYTES, GraphActor, TokPolicy, cat_batch, crn_uniforms,
                                         DEDUP, decode_rows, distinct_ids, first_of, learner_plan, load_compatible,
                                         packed_index, plan_tokens, rows_to_device, take, to_batch, used_entities)
    p = argparse.ArgumentParser()
    p.add_argument('--groups-file', required=True)
    p.add_argument('--groups', default='normal:8,boss:5,normal_big:3')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--out', required=True)
    p.add_argument('--resume', default='')
    p.add_argument('--game-hours', type=float, default=1000.0)
    p.add_argument('--frames-per-decision', type=int, default=4)
    p.add_argument('--episodes-per-state', type=int, default=16)
    p.add_argument('--episode-seconds', type=float, default=90.0)
    p.add_argument('--start-hp', default='2:6', help='half hearts at an episode start, lo:hi, uniform per start state')
    p.add_argument('--rollout', type=int, default=8192, help='decisions per update')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--minibatch', type=int, default=2048)
    p.add_argument('--micro', type=int, default=0,
                   help="rows per forward / backward pass: a minibatch's gradient is accumulated over parts of this "
                        'size (0: the whole minibatch at once). The step is the same, the GPU memory smaller')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-final', type=float, default=1e-4)
    p.add_argument('--gamma', type=float, default=0.995)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--reward', default='',
                   help='terms as name=weight,...; the default is ROOM_REWARD, in floor mode FLOOR_REWARD')
    p.add_argument('--mode', default='room', choices=('room', 'floor', 'run'),
                   help='floor: whole Basement I floors (isaac_bridge/tok_floor.py); one group, named floor. run '
                        '(2026-10-06): from Basement I on through the trapdoors (one group, named run)')
    p.add_argument('--run-seconds', type=float, default=1800.0,
                   help='run mode: game seconds a run may last (the stall rule of each floor as in floor mode)')
    p.add_argument('--items', action='store_true',
                   help='2026-10-06: item-aware observation and the item / pill heads (see the module docstring)')
    p.add_argument('--charge', type=int, default=1,
                   help="2026-10-07: the player's charge state (ROW pcharge) as policy input; a resumed checkpoint keeps "
                        'its own setting')
    p.add_argument('--ent-ext', type=int, default=1,
                   help='2026-10-08: the entity flag and laser columns (tok_obs ENT_F0 .. ENT_F) as policy input; a '
                        'resumed checkpoint keeps its own setting, --init pads them with zeros')
    p.add_argument('--floor-seconds', type=float, default=480.0)
    p.add_argument('--floor-stall-seconds', type=float, default=60.0)
    p.add_argument('--archive-size', type=int, default=12)
    p.add_argument('--archive-prob', type=float, default=0.7)
    p.add_argument('--snapshot-prob', type=float, default=0.5)
    p.add_argument('--archive-boss', type=float, default=0.0,
                   help='floor mode: share of the archive starts taken from boss-room entries (they are always parked '
                        'and evicted last); 0: the archive as before')
    p.add_argument('--archive-deep', type=float, default=1.0,
                   help='run mode: archive entries drawn with weight archive_deep^(floor - 1), the shallowest floor '
                        'evicted first (1: uniform, as before)')
    p.add_argument('--init', default='', help='checkpoint whose weights start the run (shapes that differ are padded)')
    p.add_argument('--width', type=int, default=192)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--checkpoint-every', type=int, default=25)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--port', type=int, default=28300)
    p.add_argument('--name', default='tk')
    p.add_argument('--stub-list', default='')
    p.add_argument('--tf32', action='store_true', default=True)
    p.add_argument('--teacher', action='store_true',
                   help='hindsight search teacher (tok_sampler): the searches run while the workers are held back')
    p.add_argument('--teacher-share', type=float, default=None,
                   help="and up to this share of a worker's sampling time right after an episode (default 0; "
                        f'{OVERLAP_TEACHER_SHARE} with --overlap, where the workers are no longer held back)')
    p.add_argument('--teach-coef', type=float, default=1.0, help='weight of -log(policy mass on the safe moves)')
    p.add_argument('--danger-coef', type=float, default=1.0, help='weight of the danger head (which moves got hurt)')
    p.add_argument('--teach-batch', type=int, default=512, help='teacher records per minibatch step at most')
    p.add_argument('--teach-reuse', type=float, default=8.0,
                   help='how often a teacher record is used in all: an update draws this many times the records that '
                        'arrive per update (a running mean), spread over its minibatch steps')
    p.add_argument('--teach-weight', type=float, default=8.0,
                   help='decisions one teacher record counts for: a step with b teacher records and a minibatch of m '
                        'decisions weights the teacher losses by teach-weight x b / m')
    p.add_argument('--teach-buffer', type=int, default=60000, help='teacher records kept (the newest)')
    p.add_argument('--teacher-queue', type=int, default=24, help='hurts waiting to be searched per worker')
    p.add_argument('--teacher-slots', type=int, default=0,
                   help='searches running at once over all workers at most (0: no limit; tok_sampler.SearchSlots)')
    p.add_argument('--teacher-skip-known', type=int, default=0,
                   help='1: no clone for the move the episode itself held up to the hurt (its danger bit is known to '
                        'be 1; the teacher record is the same, only that move\'s damage total is not measured)')
    p.add_argument('--teacher-pair', type=int, default=1,
                   help="1 (default since 2026-10-05): the queued hurts of one episode are searched together, one "
                        'restore replay for their first depth (tok_sampler.hindsight_shared; the same records and '
                        'lost[]); 0: one restore per hurt')
    # 2026-10-09 (death teacher, SCALING_THESIS.md S5): the fatal hurt always searched, deeper, with its own margin
    p.add_argument('--teacher-death-depths', default='',
                   help='floor / run modes: depths (decisions before the fatal step) for the episode-ending hurt, e.g. '
                        '"2,4,8,16,32"; it is then always searched (never dropped by --teacher-per-episode). '
                        'Default: off (the fatal hurt is an ordinary hurt)')
    p.add_argument('--teacher-death-margin', type=int, default=15,
                   help='decisions after the fatal step a held move must stay unhurt (2 s); 0: --teacher-margin')
    p.add_argument('--teacher-thread', type=int, default=0,
                   help='2026-10-10, floor / run modes: 1 = the searches run in a thread of the worker (its episode '
                        'never waits for a search; --teacher-share bounds the search time outside the updates); 0 = as '
                        'before (searches during the updates, and within --teacher-share in the worker loop)')
    # 2026-10-07: counterfactual branches at decision points (isaac_bridge/tok_branch.py; floor / run modes)
    p.add_argument('--branch-share', type=float, default=0.0,
                   help="share of a worker's wall time spent on branch points (0: off, nothing changes)")
    p.add_argument('--branch-seconds', type=float, default=60.0,
                   help="a branch episode's game-time cap (Phase B2: 60, was 180; the rest is the value bootstrap)")
    p.add_argument('--branch-max', type=int, default=3, help='options run per point at most (progressive widening)')
    p.add_argument('--branch-close', type=float, default=0.5,
                   help="one more option while every |outcome - the episode's own option's| is below this")
    p.add_argument('--branch-noise', type=float, default=0.0,
                   help='probability of an unconditional replicate pair (both options again with a new reseed and '
                        'policy-noise key: the noise floor of a label, pooled by the trainer per kind)')
    p.add_argument('--branch-queue', type=int, default=4, help='points waiting per worker')
    p.add_argument('--branch-kinds', default='item,item_left',
                   help='kinds branched (exit is only counted; door on request, Phase B2)')
    p.add_argument('--branch-item-back', type=int, default=1,
                   help='item points: the restore decision is this many decisions before the pickup step')
    p.add_argument('--branch-walk-seconds', type=float, default=10.0, help='scripted prefix walks at most')
    p.add_argument('--branch-weights', default='', help='outcome weights (tok_branch.DEFAULT_WEIGHTS when empty)')
    p.add_argument('--branch-uncert', type=float, default=0.0,
                   help="weight of the choice heads' spread at a point's decision record in its queue priority")
    # Phase B2 (2026-10-07, tok_branch's second docstring)
    p.add_argument('--branch-crn', type=int, default=1,
                   help='1: common random numbers for the policy in a branch pair (ROW crn, GraphActor); 0: fresh noise')
    p.add_argument('--branch-twin', type=float, default=0.0,
                   help='probability of a twin of the first branch (same option, reseed and noise key): the CRN check')
    p.add_argument('--branch-replicate-max', type=int, default=2,
                   help='conditional replicate pairs per point at most (while |mean D| < 2 noise s.e.)')
    p.add_argument('--branch-noise-prior', type=float, default=1.0,
                   help="a label's noise variance before the trainer has an estimate for the kind")
    p.add_argument('--branch-snap-every', type=int, default=20,
                   help='the actor answers branch records with weight snapshots refreshed every this many updates (two '
                        'kept; a point pins the newest at its start: its branches are played by one policy); 0: the '
                        'current weights')
    p.add_argument('--branch-left-back', type=int, default=4,
                   help='item_left points: restored this many decisions before the step that left the room (Phase B2; '
                        'at the step itself the player may already be in the door transition)')
    p.add_argument('--branch-archive', type=int, default=0,
                   help="1: the take branch's state after the take and its end state (alive) into the archive")
    p.add_argument('--choice-heads', type=int, default=-1,
                   help='K choice-value heads (TokPolicy choice; -1: 4 with --branch-share > 0, else 0)')
    p.add_argument('--choice-lr', type=float, default=1e-3)
    p.add_argument('--choice-label', default='score', choices=('score', 'score10', 'score20', 'score30'),
                   help="the branches' score the labels D are taken from: the full branch (score) or bootstrapped "
                        'after 10 / 20 / 30 s of it (tok_branch.SCORE_HORIZONS; the replicate-widening of the workers '
                        'always uses score)')
    p.add_argument('--choice-steps', type=int, default=40, help='choice-head minibatch steps per update')
    p.add_argument('--choice-batch', type=int, default=256)
    p.add_argument('--choice-holdout', type=int, default=5,
                   help='a point whose floor seed hash is 0 mod this is held out of the choice heads training '
                        '(0: none)')
    p.add_argument('--choice-adv-coef', type=float, default=0.0,
                   help="weight of the choice records' advantage correction (0: off until validated)")
    p.add_argument('--choice-adv-min', type=float, default=0.5,
                   help='|own - alternative| a choice record needs (and 2 noise s.e.) to correct its decision')
    p.add_argument('--choice-adv-max', type=float, default=5.0, help='the correction is clipped to this')
    p.add_argument('--branch-ppo', type=int, default=1,
                   help='1: the branch episodes\' decisions are PPO samples too (on-policy, legitimate states); 0: masked')
    p.add_argument('--choice-buffer', type=int, default=5000, help='choice records kept (the newest)')
    # 2026-10-08: the build lab (isaac_bridge/tok_lab.py; floor / run modes, off unless --lab-share > 0)
    p.add_argument('--lab-share', type=float, default=0.0,
                   help="share of a worker's wall time on lab jobs (0: off, nothing changes; >= 1: whenever one waits)")
    p.add_argument('--lab-panel', default='normal:24,boss:6,normal_big:6',
                   help='panel states per catalog group (room-mode resets of held-out seeds, tok_lab.parse_panel)')
    p.add_argument('--lab-seed', type=int, default=2147700000, help='seed of the panel states (tok_lab.parse_panel)')
    p.add_argument('--lab-seconds', type=float, default=90.0, help="a lab episode's game-time cap")
    p.add_argument('--lab-costs', default='',
                   help="an earlier run's lab.jsonl (same panel): its states' mean game seconds balance the states over "
                        'the workers (tok_lab.assign_panel); default: by group')
    p.add_argument('--lab-ppo', type=int, default=1, help="1: the lab episodes' decisions are PPO samples; 0: masked")
    p.add_argument('--lab-items', default='', help='eligible collectibles (default ../abplus/catalog/lab_items.json)')
    p.add_argument('--lab-mix', default='base=0.08,model=0.15,random=0.1,seen=0.05',
                   help='shares of the new jobs by kind (the rest: single items, each eligible one in turn)')
    p.add_argument('--lab-builds', default='', help='a JSON list of builds (lists of ids) benchmarked first')
    p.add_argument('--lab-noise', type=float, default=0.0,
                   help='share of the jobs that replicate a recent job (half the same rep: the twin check, half a new '
                        'rep: the seed noise)')
    p.add_argument('--lab-inflight', type=int, default=24, help='jobs posted and not yet complete at most')
    p.add_argument('--lab-timeout', type=float, default=900.0, help='s after which a job with states missing is closed')
    p.add_argument('--lab-max-items', type=int, default=3, help='collectibles of a model / random build at most')
    p.add_argument('--lab-model-k', type=int, default=5, help='heads of the build model (ensemble)')
    p.add_argument('--lab-model-steps', type=int, default=50, help='build-model Adam steps per update')
    p.add_argument('--lab-pool', type=int, default=512, help='random candidate builds the model ranks per update')
    # 2026-10-08: start builds (floor / run modes): random builds transplanted into ordinary episodes' start states
    p.add_argument('--start-build-prob', type=float, default=0.0,
                   help='share of the floor-start training episodes that begin with a random build (0: off; archive '
                        'starts keep their build)')
    p.add_argument('--start-build-max', type=int, default=3, help='collectibles of a start build at most (1 .. max)')
    # 2026-10-08: stat augmentation (floor / run modes): random multipliers of the base stats at floor starts
    p.add_argument('--stat-aug-prob', type=float, default=0.0,
                   help='share of the floor-start training episodes whose base stats get random multipliers (0: off)')
    p.add_argument('--stat-aug', default='',
                   help='multiplier ranges, log-uniform: "damage=0.7:2.5,tears=0.7:2,range=0.8:1.5,shot_speed=0.8:1.3,'
                        'speed=0.9:1.3" (the default; keys speed, damage, shot_speed, tears, range)')
    # 2026-10-08: character randomisation (floor / run modes): floor starts restart the run as a drawn character
    p.add_argument('--characters', default='',
                   help='PlayerTypes floor starts are built as, with optional weights: "0,1,2,3,5,6,7,8,13,15", '
                        '"0:4,7:1" or "all" (tok_floor.CHARACTERS_ALL); refused: 10 The Lost, 11 Lazarus II, 12 Black '
                        'Judas, 16 The Forgotten, 17 The Soul (tok_floor.CHARACTERS_UNSUPPORTED). Default: Isaac only')
    p.add_argument('--pchar', type=int, default=-1,
                   help="2026-10-08: the player's PlayerType (ROW pchar, TokPolicy's char_emb table) as policy input; "
                        '-1: on with --characters; a resumed checkpoint keeps its own setting, --init starts the table '
                        'at zero (the same outputs as the checkpoint)')
    p.add_argument('--build-timeout', type=float, default=60.0,
                   help="s a worker's root instance may take for the first start state after its launch (launch, "
                        'connect, reset): past it the instance is killed and relaunched and the same seed built once '
                        'more, then the next (counted in errors and stalls; tok_sampler.StartBuilder). 0: no limit. '
                        '120 before 2026-10-06')
    p.add_argument('--reset-timeout', type=float, default=20.0,
                   help='s a later start state (a reset of a running root, about 0.1 s) may take, the same handling '
                        '(2026-10-06); 0: --build-timeout')
    p.add_argument('--root-anon-mib', type=float, default=0.0,
                   help="2026-10-06: a worker's root instance whose anonymous RSS (RssAnon: its heap, about 250 MiB "
                        'after the launch) is above this many MiB is replaced at a state '
                        'boundary, after the new template was forked from it (its clones stay; tok_sampler.recycle_root). '
                        '0: never')
    p.add_argument('--trim-parked', type=int, default=1,
                   help='1 (default since 2026-10-06): every parked clone (a template, a whole-floor room entry) gives '
                        'its free heap pages back (malloc_trim; tok_sampler.trim_parked; the game from it is the same, '
                        'abplus_probe_trim.py); 0: as before')
    p.add_argument('--graph-entities', type=int, default=32,
                   help='entity tokens of the CUDA-graph actor (0: eager inference); a second graph takes all 64')
    p.add_argument('--overlap', action='store_true',
                   help='collect the next rollout while the learner updates (one update of policy lag; see above)')
    p.add_argument('--recompute-values', default='auto', choices=('auto', 'on', 'off'),
                   help="recompute the rollout's values with the learner's weights before the advantages (auto: on "
                        'with --overlap)')
    p.add_argument('--decoupled', type=int, default=0,
                   help='1: decoupled PPO objective (Hilton et al. 2021): the ratio is taken to the weights at the '
                        "update's start (log-probabilities recomputed then) and clipped there; the behaviour policy "
                        'enters as the importance weight pi_start / pi_behaviour. Synchronous PPO is unchanged by it '
                        '(the two policies are the same); with --overlap it keeps the clipping where synchronous PPO '
                        'has it. 0: the plain ratio to the behaviour policy')
    p.add_argument('--decoupled-max-weight', type=float, default=2.0,
                   help='upper bound of the decoupled importance weight')
    p.add_argument('--poll-want', type=int, default=0,
                   help='records the actor waits for before a call, up to --poll-wait s after the first (0: half the '
                        'workers, as before); 1: it answers whatever is ready')
    p.add_argument('--poll-wait', type=float, default=0.0005)
    p.add_argument('--actor-slots', type=int, default=2,
                   help='actor calls in flight at once (each its own CUDA graphs and stream): while one runs on the '
                        'GPU the next records are read, booked and sent; 1: one call at a time')
    p.add_argument('--sort-micro', type=int, default=1,
                   help="1: a minibatch's rows are ordered by entity count before they are cut into --micro parts "
                        '(fewer padded entity tokens; the same gradient up to summation order); 0: as drawn')
    p.add_argument('--packed', type=int, default=1,
                   help="1: the learner's Transformer layers run on the real tokens only (TokPolicy packed: padded door "
                        'and entity tokens skipped; the same values up to summation order); 0: on the padded layout')
    p.add_argument('--teach-merge', type=int, default=1,
                   help="1: a step's teacher records join the minibatch's last part (one forward and backward; the "
                        'gradient is the same sum of the PPO and teacher terms); 0: a pass of their own')
    p.add_argument('--teach-gpu', type=int, default=1,
                   help='1: the teacher records are also kept as bytes on the GPU and a step gathers its records there '
                        '(the same values as from the host copy); 0: from the host copy')
    p.add_argument('--amp', default='off', choices=('off', 'bf16'),
                   help="bf16: the learner's forward passes (PPO, teacher, the values recomputed at an update's start) "
                        'under bfloat16 autocast: matrix products, convolutions and attention in bfloat16; the '
                        'residual stream, layer norms, the input MLPs over raw float features (entity, player, door), '
                        'the heads, log-softmax and every loss stay float32 (TokPolicy). The actor (sampling, stored '
                        'log-probabilities) stays float32. off: float32 throughout (TF32 matrix products with --tf32)')
    p.add_argument('--learner-fast', type=int, default=0,
                   help="2026-10-10 (B18): 1: the learner's fast path: the same PPO update (losses, gradients, steps) "
                        'with about half the GPU and host time per minibatch step (attention per length bucket as '
                        'float32 matrix products, the input mlps on the real tokens, the CNNs on the distinct grids, '
                        'no host sync per step, the fused Adam; learn() in train_tok.py, EXPERIMENTS.md B18). Not '
                        'bit-identical: the same function in another summation order (abplus_bench_learner.py '
                        '--check, abplus_check_learner_fp64.py). 0: as before')
    p.add_argument('--learner-cpus', default='',
                   help='2026-10-10 (B18): CPUs for the learner thread alone, e.g. 56-59,168-171 (Linux numbering; '
                        'whole cores with their hyperthread siblings, best on the GPU\'s NUMA node): the trainer\'s '
                        'other threads, its workers and their games are kept off them. The fast learner is bound by '
                        'its host thread, which a busy sibling or a game on the same core slows by ~20%%. '
                        "'' (default): no pinning")
    p.add_argument('--fused-adam', type=int, default=0,
                   help='1: Adam as one fused CUDA kernel (the same update rule; float rounding may differ)')
    p.add_argument('--dump-rollouts', type=int, default=0,
                   help="save the first K rollouts handed to the learner (records, actions under way, actions, stored "
                        "log-probabilities and values, rewards) to <out>/rollouts/ (a measurement aid)")
    p.add_argument('--switch-interval', type=float, default=0.0005,
                   help="Python's thread switch interval (s): how long the actor may wait for the interpreter while "
                        'the learner thread runs Python (default of Python: 0.005)')
    args = p.parse_args()
    if args.poll_want <= 0:
        args.poll_want = max(1, args.workers // 2)
    if args.teacher_share is None:
        args.teacher_share = OVERLAP_TEACHER_SHARE if (args.overlap and args.teacher) else 0.0
    recompute = args.recompute_values == 'on' or (args.recompute_values == 'auto' and args.overlap)
    if args.switch_interval > 0:
        sys.setswitchinterval(args.switch_interval)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    (out / 'checkpoints').mkdir(exist_ok=True)
    run = args.mode == 'run'
    floor = args.mode == 'floor' or run
    if args.resume and not args.items:   # an items checkpoint keeps its heads
        args.items = bool(torch.load(args.resume, map_location='cpu').get('config', {}).get('items'))
    items = bool(args.items)
    if args.resume:
        args.charge = int(bool(torch.load(args.resume, map_location='cpu').get('config', {}).get('charge')))
    charge = bool(args.charge)
    if args.resume:   # 2026-10-08
        args.ent_ext = int(bool(torch.load(args.resume, map_location='cpu').get('config', {}).get('ent_ext')))
    ent_ext = bool(args.ent_ext)
    if args.resume:   # 2026-10-08 (characters)
        args.pchar = int(bool(torch.load(args.resume, map_location='cpu').get('config', {}).get('pchar')))
    elif args.pchar < 0:
        args.pchar = int(bool(args.characters))
    pchar = bool(args.pchar)
    rw = dict(room=0.0, explore=0.0, boss=0.0, exit=0.0, heal=0.0, resource=0.0)
    rw.update({k: float(v) for k, v in (kv.split('=') for kv in (FLOOR_REWARD if floor else ROOM_REWARD).split(','))})
    if args.reward:
        rw.update({k: float(v) for k, v in (kv.split('=') for kv in args.reward.split(','))})
    if floor:   # the root instances are launched with the normal group's spec; the floor reset does not use its rooms
        names, assign = ['run' if run else 'floor'], [0] * args.workers
        specs = [load_spec(argparse.Namespace(groups_file=args.groups_file, group='normal', tasks='', seconds=0.0))]
    else:
        names, assign = parse_groups(args.groups, args.workers)
        specs = [load_spec(argparse.Namespace(groups_file=args.groups_file, group=n, tasks='', seconds=0.0))
                 for n in names]
    lab_on = floor and args.lab_share > 0   # 2026-10-08 (tok_lab)
    if args.start_build_prob > 0 and not (floor and items):
        p.error('--start-build-prob needs --mode floor/run and --items')
    if args.stat_aug_prob > 0:
        if not floor:
            p.error('--stat-aug-prob needs --mode floor/run')
        from isaac_bridge.tok_floor import parse_stat_aug
        parse_stat_aug(args.stat_aug)   # (a bad spec fails here, not in the workers)
    char_ids = []   # 2026-10-08 (characters)
    if args.characters:
        if not floor:
            p.error('--characters needs --mode floor/run')
        from isaac_bridge.tok_floor import parse_characters
        try:
            char_ids = [int(c) for c in parse_characters(args.characters)[0]]
        except ValueError as exc:
            p.error(str(exc))
    lab_names, lab_panel, lab_specs, lab_owner = [], [], [], []
    if lab_on:
        from isaac_bridge.tok_lab import assign_panel, parse_panel, state_costs_from
        lab_names, lab_panel = parse_panel(args.lab_panel, args.lab_seed)
        lab_owner = assign_panel(lab_panel, lab_names, args.workers, state_costs=state_costs_from(
            args.lab_costs, len(lab_panel)) if args.lab_costs else None)
        lab_specs = [load_spec(argparse.Namespace(groups_file=args.groups_file, group=nm, tasks='', seconds=0.0))
                     for nm in lab_names]
    lo, hi = (int(v) for v in args.start_hp.split(':'))
    # the instances' stub list: stub_render_h.txt next to the library (B14; else _g), or --stub-list
    stub = args.stub_list or default_stub_list(default_preload())
    cfg = TokSamplerConfig(workers=args.workers, specs=specs, assign=assign, frames_per_decision=args.frames_per_decision,
                           episodes_per_state=args.episodes_per_state, episode_seconds=args.episode_seconds,
                           start_hp=(lo, hi), seed=args.seed, name=args.name, port=args.port,
                           teacher=args.teacher, teacher_share=args.teacher_share, teacher_queue=args.teacher_queue,
                           teacher_slots=args.teacher_slots, teacher_skip_known=bool(args.teacher_skip_known),
                           teacher_pair=bool(args.teacher_pair),
                           teacher_death_depths=tuple(int(v) for v in args.teacher_death_depths.split(',') if v.strip())
                           if floor else (), teacher_death_margin=args.teacher_death_margin, teacher_thread=args.teacher_thread,
                           mode=args.mode,
                           floor_seconds=args.floor_seconds, floor_stall_seconds=args.floor_stall_seconds,
                           archive_size=args.archive_size, archive_prob=args.archive_prob,
                           snapshot_prob=args.snapshot_prob, archive_boss=args.archive_boss, build_timeout=args.build_timeout,
                           root_anon_mib=args.root_anon_mib, reset_timeout=args.reset_timeout,
                           trim_parked=bool(args.trim_parked), items=items, run_seconds=args.run_seconds,
                           branch_share=args.branch_share if floor else 0.0, branch_seconds=args.branch_seconds,
                           branch_max=args.branch_max, branch_close=args.branch_close, branch_noise=args.branch_noise,
                           branch_queue=args.branch_queue, branch_kinds=args.branch_kinds,
                           branch_item_back=args.branch_item_back, branch_walk_seconds=args.branch_walk_seconds,
                           branch_weights=args.branch_weights, branch_uncert=args.branch_uncert,
                           branch_crn=bool(args.branch_crn), branch_twin=args.branch_twin,
                           branch_replicate_max=args.branch_replicate_max, branch_noise_prior=args.branch_noise_prior,
                           branch_reward=','.join(f'{k}={v}' for k, v in rw.items()), branch_gamma=args.gamma,
                           branch_archive=args.branch_archive, branch_left_back=args.branch_left_back,
                           lab_share=args.lab_share if lab_on else 0.0, lab_seconds=args.lab_seconds,
                           lab_specs=lab_specs, lab_panel=lab_panel, lab_owner=lab_owner,
                           start_build_prob=args.start_build_prob if floor else 0.0, start_build_max=args.start_build_max,
                           start_build_items=args.lab_items, archive_deep=args.archive_deep,
                           stat_aug_prob=args.stat_aug_prob if floor else 0.0, stat_aug=args.stat_aug,
                           characters=args.characters if floor else '',
                           bridge_lua=default_bridge_lua(), preload=default_preload(),
                           stub_list=stub if stub and Path(stub).is_file() else '')
    print('instances: stub list', cfg.stub_list or '(the launch mode\'s own)', '| environment',
          apply_instance_defaults(dict(os.environ)), '| teacher_pair', cfg.teacher_pair, flush=True)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    branching = cfg.branch_share > 0   # 2026-10-07 (tok_branch): branch episodes and choice records
    if args.choice_heads < 0:   # Phase B2: K choice-value heads with branches (a resumed checkpoint's own count wins)
        args.choice_heads = 4 if branching else 0
    if args.resume:
        args.choice_heads = int(torch.load(args.resume, map_location='cpu').get('config', {}).get('choice', 0)) or             args.choice_heads
    n_choice = int(args.choice_heads)
    learner_model = TokPolicy(args.width, args.layers, args.heads, items=items, charge=charge,
                              choice=n_choice, ent_ext=ent_ext, pchar=pchar).to(device)   # the weights PPO updates
    nh = len(learner_model.heads)
    fused = bool(args.fused_adam) or bool(args.learner_fast)   # (B18: --learner-fast 1 implies the fused Adam)
    # (Phase B2: the choice heads are not PPO's: main_parameters() is parameters() without them, in the same order)
    opt = torch.optim.Adam(learner_model.main_parameters(), lr=args.lr, eps=1e-5,
                           **(dict(fused=True) if fused else {}))
    copt = torch.optim.Adam(learner_model.choice.parameters(), lr=args.choice_lr) if n_choice else None
    update, decisions_total, seconds_total = 0, 0, 0.0
    if args.init:
        print('init', args.init, load_compatible(learner_model, torch.load(args.init, map_location=device)['model']))
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        missing, unexpected = learner_model.load_state_dict(ck['model'], strict=False)
        if unexpected or any(not k.startswith('choice.') for k in missing):
            raise SystemExit(f'--resume: weights do not match ({missing[:4]} {unexpected[:4]})')
        if copt is not None and ck.get('choice_optimizer'):
            copt.load_state_dict(ck['choice_optimizer'])
        osd = ck['optimizer']
        for g in osd['param_groups']:   # the checkpoint's groups carry its own implementation choice: this run's wins
            g['fused'], g['foreach'] = (True, None) if fused else (None, None)
        opt.load_state_dict(osd)        # (fused: Adam's step counts go to the device, else they stay / go to the CPU)
        if not fused:
            for st in opt.state.values():
                if torch.is_tensor(st.get('step')) and st['step'].is_cuda:
                    st['step'] = st['step'].cpu()
        update, decisions_total, seconds_total = ck['update'], ck['decisions'], ck.get('seconds', 0.0)
    amp = args.amp == 'bf16'
    if amp and not torch.cuda.is_bf16_supported():
        raise SystemExit('--amp bf16: this GPU has no bfloat16 support')

    def autocast():
        """The learner's forward passes (the losses are computed outside, in float32)."""
        return torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp)
    model = TokPolicy(args.width, args.layers, args.heads, items=items, charge=charge,
                      choice=n_choice, ent_ext=ent_ext, pchar=pchar).to(device).eval()   # the actor's copy
    model.load_state_dict(learner_model.state_dict())
    actor_params = [t for t in list(model.parameters()) + list(model.buffers())]
    learner_params = [t for t in list(learner_model.parameters()) + list(learner_model.buffers())]
    hours_per_decision = args.frames_per_decision / 30 / 3600
    config = dict(vars(args), groups_resolved=names, assign=assign,
                  parameters=sum(q.numel() for q in learner_model.parameters()), reward_terms=rw, lag=1,
                  record='tok_obs.ROW', sampler='tok_sampler (fork sampler, lean bridge)',
                  recompute_values_resolved=recompute)
    (out / 'config.json').write_text(json.dumps(config, indent=1, default=str))
    log_path = out / 'progress.csv'
    log_file = None

    def log_columns(columns):
        """Open progress.csv for appending with these columns. A file written by an older version with other columns
        (a resumed run) is moved aside to progress.csv.old<k> and copied into a new file whose header is its own
        columns followed by the new ones (empty in the old rows), so the file stays one table."""
        nonlocal log_file
        header = []
        if log_path.is_file() and log_path.stat().st_size > 0:
            with open(log_path, newline='') as f:
                header = next(csv.reader(f), [])
        if header and header != columns:
            merged = header + [c for c in columns if c not in header]
            k = 0
            while (out / f'progress.csv.old{k}').exists():
                k += 1
            old = out / f'progress.csv.old{k}'
            os.replace(log_path, old)
            with open(old, newline='') as f, open(log_path, 'w', newline='') as g:
                w = csv.DictWriter(g, fieldnames=merged, restval='')
                w.writeheader()
                for r in csv.DictReader(f):
                    w.writerow({c: r.get(c, '') for c in merged})
            columns = merged
        log_file = open(log_path, 'a', newline='')
        return columns
    episodes_path = open(out / 'episodes.jsonl', 'a')
    branch_path = open(out / 'branch_episodes.jsonl', 'a') if branching else None
    choice_path = open(out / 'choices.jsonl', 'a') if branching else None
    learner_cpus = set()   # 2026-10-10 (B18, --learner-cpus): before the workers start (they inherit the rest)
    for part_ in filter(None, args.learner_cpus.split(',')):
        lo_, _, hi_ = part_.partition('-')
        learner_cpus.update(range(int(lo_), int(hi_ or lo_) + 1))
    if learner_cpus:
        os.sched_setaffinity(0, os.sched_getaffinity(0) - learner_cpus)
    bench_job = None   # 2026-10-10 (B18, abplus_bench_learner.py): a saved rollout instead of the engine
    if os.environ.get('ISAAC_RL_LEARNER_BENCH') and Path(os.environ.get('ISAAC_RL_LEARNER_JOB') or '-').is_file():
        from abplus_bench_learner import JobSampler, load_job
        bench_job, bench_stats = load_job(os.environ['ISAAC_RL_LEARNER_JOB'], Rollout,
                                          args.rollout // args.workers * 4 + 64)
        sampler = JobSampler(bench_stats)
    else:
        sampler = TokSampler(cfg)
    n = args.workers
    lab = None
    if lab_on:   # 2026-10-08: the build lab's trainer side (tok_lab: LabBook, LabScheduler, LabModel)
        from isaac_bridge.tok_lab import (JI as LAB_JI, LAB_JOBS, LabBook, LabModel, LabScheduler, load_items)
        lab_items = load_items(args.lab_items or str(Path(__file__).resolve().parent.parent / 'abplus' / 'catalog' /
                                                     'lab_items.json'))
        mix = {k: float(v) for k, v in (kv.split('=') for kv in args.lab_mix.split(',') if kv.strip())}
        first_builds = json.loads(Path(args.lab_builds).read_text()) if args.lab_builds else []
        lab = dict(book=LabBook(len(lab_panel), args.lab_timeout),
                   sched=LabScheduler(lab_items, args.seed, mix, first_builds, args.lab_noise, args.lab_max_items),
                   model=LabModel(k=args.lab_model_k, device='cpu', seed=args.seed),
                   out=open(out / 'lab.jsonl', 'a'), hist={}, pool_rng=np.random.default_rng([args.seed, 0x900D]),
                   posted=0, preq=[], twin=[0, 0], loss=None)
        (out / 'lab_panel.json').write_text(json.dumps(dict(groups=lab_names, states=lab_panel, owner=lab_owner),
                                                       indent=0))

        def post_lab_jobs():
            """New jobs into free slots while fewer than --lab-inflight are open (the job id is written last)."""
            now_ = time.time()
            while len(lab['book'].open) < args.lab_inflight:
                row_ = lab['sched'].make_job()
                slot_ = int(row_[LAB_JI['job']]) % LAB_JOBS
                if int(sampler.lab_jobs[slot_, 0]) in lab['book'].open:   # (only with --lab-inflight >= LAB_JOBS)
                    break
                sampler.lab_jobs[slot_, 1:] = row_[1:]
                sampler.lab_jobs[slot_, 0] = row_[0]
                lab['book'].post(row_, now_)
                lab['posted'] += 1

        def lab_learn(results):
            """In the learner thread: the new build results (prequential prediction first, the twin check of
            replicates, then into the model and lab.jsonl), the model's steps and its candidates for the scheduler."""
            m_ = lab['model']
            out_ = dict(lab_builds=len(results), lab_complete=sum(bool(r_.get('complete')) for r_ in results))
            errs = []
            for r_ in results:
                pre = m_.prequential(r_)
                if pre is not None:
                    r_['pred'], r_['pred_sd'] = pre
                    errs.append(abs(pre[0] - r_['ret']))
                if r_['kind'] == 'replicate' and r_['flags'] in lab['hist'] and r_.get('per_state'):
                    o_ = lab['hist'][r_['flags']]
                    r_['of'] = dict(job=o_['job'], rep=o_['rep'], ret=o_.get('ret'))
                    if o_['rep'] == r_['rep']:   # a twin: per state with the same weights the same game
                        for s_, v_ in r_['per_state'].items():
                            w_ = o_.get('per_state', {}).get(s_)
                            if w_ is not None and v_[8] == v_[9] == w_[8] == w_[9]:
                                lab['twin'][0] += 1
                                lab['twin'][1] += int(v_[7] == w_[7])
                m_.add(r_)
                lab['hist'][r_['job']] = r_
                lab['out'].write(json.dumps(dict(r_, update=L['update'])) + '\n')
            if len(lab['hist']) > 6000:
                for j_ in sorted(lab['hist'])[:1000]:
                    del lab['hist'][j_]
            lab['out'].flush()
            if results:
                lab['loss'] = m_.train(args.lab_model_steps)
                sched_ = lab['sched']
                ids_, act_ = sched_.ids, sched_.actives
                pool, rng_ = [], lab['pool_rng']
                while len(pool) < args.lab_pool:
                    b_ = sorted(int(v) for v in rng_.choice(ids_, size=int(rng_.integers(2, args.lab_max_items + 1)),
                                                            replace=False))
                    if sum(i_ in act_ for i_ in b_) <= 1:
                        pool.append(b_)
                sched_.candidates = m_.rank(pool)[:64]
            out_['lab_preq_abs'] = float(np.mean(errs)) if errs else ''
            out_['lab_model_loss'] = lab['loss'] if lab['loss'] is not None else ''
            out_['lab_cand_sd'] = lab['sched'].candidates[0][0] if lab['sched'].candidates else ''
            out_['lab_twin_n'], out_['lab_twin_same'] = lab['twin']
            out_['lab_model_rows'] = len(m_.rows)
            return out_
    capacity = args.rollout // n * 4 + 64
    roll, spare = Rollout(n, capacity, nh), Rollout(n, capacity, nh)

    def teach_pending(meta):
        """The action under way of teacher records from their meta (int64 [k, 16], numpy or a tensor): columns 0-2,
        and with items the item / pill bits kept in column 15."""
        if nh == 3:
            return meta[:, :3]
        if isinstance(meta, np.ndarray):
            return np.concatenate([meta[:, :3], meta[:, 15:16] & 1, (meta[:, 15:16] >> 1) & 1], 1)
        return torch.cat([meta[:, :3], meta[:, 15:16] & 1, (meta[:, 15:16] >> 1) & 1], 1)
    # streams: the actor's calls go first on the GPU, the learner's kernels fill the rest. --actor-slots calls may be
    # in flight at once, each on its own stream (one GraphActor each)
    actors = [GraphActor(model, n, args.graph_entities, stream=torch.cuda.Stream(priority=-5), crn=bool(args.branch_crn))
              for _ in range(max(1, args.actor_slots))] if args.graph_entities > 0 else []
    free = list(actors)
    inflight = collections.deque()   # (slot, workers, records, actions under way, submitted at, learner busy then)
    actor_stream = torch.cuda.Stream(priority=-5)
    learner_stream = torch.cuda.Stream(priority=0)
    torch.cuda.synchronize()
    torch.cuda.set_stream(actor_stream)

    # ------------------------------------------------------------------ learner (its own thread and CUDA stream)
    teach_gpu = bool(args.teach_gpu)
    # 2026-10-06 (host memory): with --teach-gpu 1 the teacher's records are read from their bytes on the GPU, and the
    # host keeps only their entity and door counts (the minibatch planning reads them); the full host copy (--teach-
    # buffer x ROW, about 0.85 GB at 60,000) filled up over the first ~1,000 game hours. --teach-gpu 0: as before.
    host_teach = ROW if not teach_gpu else np.dtype([('n_ent', ROW.fields['n_ent'][0]),
                                                     ('n_doors', ROW.fields['n_doors'][0])])
    L = dict(update=update, decisions=decisions_total, seconds=seconds_total, writer=None, t_end=None,
             teach_rows=np.zeros((args.teach_buffer,), host_teach) if args.teacher else None,
             teach_meta=np.zeros((args.teach_buffer, 16), np.int64), teach_n=0, teach_at=0, teach_rate=None,
             dumped=0, teach_raw=None)
    if branching:   # 2026-10-07: the choice records' ring on the host (the choice-value head will read it)
        from isaac_bridge.tok_branch import CHOICE_META, decode_meta, parse_weights
        b_weights = parse_weights(args.branch_weights)
        L.update(choice_rows=np.zeros((args.choice_buffer,), ROW),
                 choice_meta=np.zeros((args.choice_buffer, CHOICE_META), np.float64),
                 choice_version=np.zeros(args.choice_buffer, np.int64), choice_n=0, choice_at=0)
        # Phase B2: per choice record its labels (tok_branch.take_labels: one D per pair, NaN where none), their
        # replicate codes, the collectible, the kind, the sign turning D into own - alternative, the held-out flag and
        # the K heads' bootstrap weights (Poisson(1), drawn at arrival with a generator of their own)
        from isaac_bridge.tok_branch import BRANCH_MAX, HI as C_HI, KIND_ITEM, KIND_ITEM_LEFT, take_labels
        PMAX = BRANCH_MAX // 2
        L.update(c_lab=np.full((args.choice_buffer, PMAX), np.nan, np.float32),
                 c_rep=np.full((args.choice_buffer, PMAX), -1, np.int8), c_item=np.zeros(args.choice_buffer, np.int64),
                 c_kind=np.zeros(args.choice_buffer, np.int8), c_sign=np.zeros(args.choice_buffer, np.float32),
                 c_hold=np.zeros(args.choice_buffer, bool),
                 c_boot=np.ones((args.choice_buffer, max(n_choice, 1)), np.float32), noise={})
        crng = np.random.default_rng([args.seed, 77])
    target_decisions = decisions_total + int(args.game_hours / hours_per_decision)
    start_decisions = decisions_total

    def save(path):
        torch.save(dict(model=learner_model.state_dict(), optimizer=opt.state_dict(), update=L['update'],
                        decisions=L['decisions'], seconds=L['seconds'], config=learner_model.config, args=vars(args),
                        **(dict(choice_optimizer=copt.state_dict()) if copt is not None else {})),
                   path)

    def learn(job):
        """One PPO update on a full rollout; signals `fresh` as soon as the new weights are final, then logs."""
        t0 = time.perf_counter()
        U = dict.fromkeys(('recompute', 'gae', 'batch', 'teach_new', 'ppo', 'teach_build', 'teach', 'opt', 'tail'),
                          0.0)   # seconds per part of the update (u_* columns)
        mark = [t0]
        cpu0 = cpu_ticks()

        def sec(name):
            if PROFILE:
                torch.cuda.current_stream().synchronize()
            now_ = time.perf_counter()
            U[name] += now_ - mark[0]
            mark[0] = now_
        rl = job['roll']
        lengths = rl.length
        # 2026-10-10 (B18, --learner-fast 1): the same update with about half the GPU and host work per step
        # (EXPERIMENTS.md B18). math: attention as float32 matrix products, all length buckets of a layer in one
        # autograd node (TokPolicy _MathAttention / _BucketAttention); buckets: attention per bucket of records with
        # similar token counts, each record's real tokens only (learner_plan / plan_tokens: the positions planned per
        # record on the host, expanded on the GPU); ekeep: the entity and door input mlps on the real tokens only, the
        # packed tokens built from them; dedup: the room / map / patch CNNs on a minibatch's distinct grids
        # (distinct_ids once per update); nhwc: channels-last convolutions; ln: the packed layer norms' weight
        # gradients as column sums; logp: the heads' log-softmax side by side; raw: a minibatch gathered as record
        # bytes. Always with it: the records staged in a pinned buffer, the teacher's meta read from a copy on the GPU
        # (no blocking copy per step), the fused Adam. Not bit-identical to 0 (another summation order):
        # abplus_bench_learner.py --check, abplus_check_learner_fp64.py.
        fast = bool(args.learner_fast)
        # (ISAAC_RL_LEARNER_PARTS: a subset of the parts, for the bench's attribution; default all)
        fparts = set(os.environ.get('ISAAC_RL_LEARNER_PARTS', 'math,ekeep,buckets,nhwc,ln,dedup,logp,raw').split(',')
                     if fast else ())
        learner_model.set_math_attention('math' in fparts, nhwc='nhwc' in fparts, ln='ln' in fparts)
        n_buckets = int(os.environ.get('ISAAC_RL_LEARNER_BUCKETS', '3'))   # (at most; a measurement aid)
        raw_mode = 'raw' in fparts and teach_gpu and bool(args.teach_merge)   # (B18: minibatches gathered as bytes)
        if L['dumped'] < args.dump_rollouts:   # the rollout and the learner's weights before this update
            L['dumped'] += 1
            m = int(lengths.max())
            (out / 'rollouts').mkdir(exist_ok=True)
            stem = out / 'rollouts' / f'rollout-{L["update"]:06d}'
            np.savez(str(stem) + '.npz', **{k: getattr(rl, k)[:, :m] for k in (
                'rows', 'pending', 'actions', 'logp', 'value', 'reward', 'terminal', 'version')},
                length=rl.length, open=rl.open, update=L['update'])
            torch.save(dict(model=learner_model.state_dict(), config=learner_model.config, update=L['update']),
                       str(stem) + '.pt')
            t0 = time.perf_counter()
            mark[0] = t0
        all_flat = (np.arange(n)[:, None] * capacity + np.arange(capacity)[None, :])[
            np.arange(capacity)[None, :] < lengths[:, None]]
        value_now = every = prox = None
        if (recompute or args.decoupled) and len(all_flat):
            # the values the advantages need (and with --decoupled the log-probabilities of the stored actions), from
            # the weights at the start of this update
            value_now = rl.value.copy()
            every = rows_to_device(rl.rows, all_flat, rl.pending.reshape(-1, nh)[all_flat], device)
            acts_all = torch.from_numpy(rl.actions.reshape(-1, nh)[all_flat]).to(device)
            vals, lps = [], []
            with torch.inference_mode():
                chunk = args.micro or args.minibatch
                for at in range(0, len(all_flat), chunk):
                    part = torch.arange(at, min(at + chunk, len(all_flat)), device=device)
                    with autocast():
                        lp_, _, v_, _ = learner_model.evaluate(take(every, part), acts_all[part],
                                                               packed=bool(args.packed))
                    vals.append(v_)
                    lps.append(lp_)
            if recompute:
                value_now.reshape(-1)[all_flat] = torch.cat(vals).cpu().numpy()
            else:
                value_now = None
            prox_all = torch.cat(lps)
            sec('recompute')
        adv, ret, mask = gae(rl, args.gamma, args.lam, value_now)
        if branching and not args.branch_ppo:   # 2026-10-07: the branch episodes' decisions left out of PPO
            mask &= rl.rows['branch'] == 0
        if lab_on and not args.lab_ppo:   # 2026-10-08: the lab episodes' decisions left out of PPO
            mask &= rl.rows['lab'] == 0
        sec('gae')
        flat = np.flatnonzero(mask.reshape(-1))
        count = len(flat)
        frac = min(1.0, (L['decisions'] - start_decisions) / max(1, target_decisions - start_decisions))
        lr = args.lr + (args.lr_final - args.lr) * frac
        for g in opt.param_groups:
            g['lr'] = lr
        if every is not None:   # the decisions with a reward, out of the records already on the device
            pos = torch.from_numpy(np.searchsorted(all_flat, flat)).to(device)
            batch = take(every, pos)
            raw_mode = False   # (B18: no record bytes here; the fields are gathered as before)
            if args.decoupled:
                prox = prox_all[pos].clone()
            del every, prox_all
        else:
            stage = None
            if fast:   # B18: gathered into a pinned buffer kept between updates, copied asynchronously
                if L.get('stage') is None or L['stage'].shape[0] < count:
                    L['stage'] = None
                    L['stage'] = torch.empty((int(count * 1.25) + 64, ROW_BYTES), dtype=torch.uint8).pin_memory()
                stage = L['stage']
            batch = rows_to_device(rl.rows, flat, rl.pending.reshape(-1, nh)[flat], device, stage=stage,
                                   with_raw=fast)
            if fast:   # (B18: a minibatch's records are gathered as bytes, the fields are views of them)
                batch, raw_all, pend_all = batch
        b_act =torch.from_numpy(rl.actions.reshape(-1, nh)[flat]).to(device)
        b_logp = torch.from_numpy(rl.logp.reshape(-1)[flat]).to(device)
        b_adv = torch.from_numpy(adv.reshape(-1)[flat]).to(device)
        b_ret = torch.from_numpy(ret.reshape(-1)[flat]).to(device)
        used_value = rl.value if value_now is None else value_now
        b_val = torch.from_numpy(used_value.reshape(-1)[flat]).to(device)
        stale = float((rl.version.reshape(-1)[flat] < L['update']).mean()) if count else 0.0
        sec('batch')
        acc = torch.zeros(5, device=device)        # pg, vf, ent, kl, clipped (sums over steps, on the GPU)
        tacc = torch.zeros(3, device=device)       # teach, danger, safe_mass
        steps = teach_steps = 0
        teach_new, teach_step, safe_new = 0, 0, 0.0
        teach_rows, teach_meta = L['teach_rows'], L['teach_meta']
        if teach_rows is not None:
            new_rows, new_meta = job['teach']
            teach_new = len(new_rows)
            if teach_new:   # the policy's mass on the safe moves of records it has not been trained on
                with torch.inference_mode():
                    meta = torch.from_numpy(new_meta.astype(np.int64)).to(device)
                    with autocast():
                        lg = learner_model(to_batch(new_rows, teach_pending(new_meta.astype(np.int64)), device),
                                         packed=bool(args.packed))[0]
                    mass = torch.logsumexp(torch.log_softmax(lg.float()[:, :9], -1)
                                           .masked_fill(meta[:, 3:12] > 0, -1e9), -1).exp()
                    has = meta[:, 14] > 0
                    safe_new = float((mass * has).sum() / has.sum().clamp(min=1))
            L['teach_rate'] = teach_new if L['teach_rate'] is None else 0.9 * L['teach_rate'] + 0.1 * teach_new
            planned = args.epochs * max(1, count // args.minibatch)
            teach_step = int(min(args.teach_batch, args.teach_reuse * L['teach_rate'] / planned))
            at_ = (L['teach_at'] + np.arange(teach_new)) % args.teach_buffer   # the ring positions, in order
            if teach_new:
                if teach_gpu:   # (field by field: the host keeps the counts only)
                    teach_rows['n_ent'][at_], teach_rows['n_doors'][at_] = new_rows['n_ent'], new_rows['n_doors']
                else:
                    teach_rows[at_] = new_rows
                teach_meta[at_] = new_meta
                if teach_gpu:   # the same records as bytes on the device (the teacher's minibatches gather there)
                    if L['teach_raw'] is None:
                        L['teach_raw'] = torch.zeros((args.teach_buffer, ROW_BYTES), dtype=torch.uint8, device=device)
                    L['teach_raw'][torch.from_numpy(at_).to(device)] = torch.from_numpy(
                        np.ascontiguousarray(new_rows).view(np.uint8).reshape(teach_new, ROW_BYTES)).to(device)
            L['teach_at'] = (L['teach_at'] + teach_new) % args.teach_buffer
            L['teach_n'] = min(L['teach_n'] + teach_new, args.teach_buffer)
            sec('teach_new')
        teach_n = L['teach_n']
        choice_new = 0
        if branching and job.get('choices') is not None:   # 2026-10-07: choice records into the ring and choices.jsonl
            c_rows, c_meta = job['choices']
            choice_new = len(c_rows)
            if choice_new:
                at_c = (L['choice_at'] + np.arange(choice_new)) % args.choice_buffer
                L['choice_rows'][at_c], L['choice_meta'][at_c] = c_rows, c_meta
                L['choice_version'][at_c] = L['update']   # the weights when the record arrived (meta: the actor's)
                L['choice_at'] = (L['choice_at'] + choice_new) % args.choice_buffer
                L['choice_n'] = min(L['choice_n'] + choice_new, args.choice_buffer)
                for m_ in c_meta:
                    choice_path.write(json.dumps(dict(decode_meta(m_, b_weights), update=L['update'])) + '\n')
                choice_path.flush()
                # Phase B2: the records' labels (see L's c_* arrays)
                for j_, m_ in zip(at_c, c_meta):
                    labs, sign_ = take_labels(m_, args.choice_label)
                    L['c_lab'][j_], L['c_rep'][j_] = np.nan, -1
                    for q_, (_, r_, d_) in enumerate(labs[:PMAX]):
                        L['c_lab'][j_, q_], L['c_rep'][j_, q_] = d_, r_
                    seed_ = int(m_[C_HI['seed']])
                    L['c_item'][j_] = min(max(int(m_[C_HI['key']]), 0), 1023)
                    L['c_kind'][j_], L['c_sign'][j_] = int(m_[C_HI['kind']]), sign_
                    # held out by the floor's seed: the points of one start state (16 episodes, the archive's room
                    # entries) are much alike; a split by point would test on situations seen in training
                    L['c_hold'][j_] = args.choice_holdout > 0 and \
                        ((seed_ * 2654435761) & 0xFFFFFFFF) % args.choice_holdout == 0
                    L['c_boot'][j_] = crng.poisson(1.0, L['c_boot'].shape[1])
            # the pooled noise variance of a label per kind: pairs 1 and the unconditional replicate pair of the same
            # point, (D1 - D2)^2 / 2; handed to the workers (aux) for their replicate-widening
            m_n = L['choice_n']
            lab_, rep_, kind_ = L['c_lab'][:m_n], L['c_rep'][:m_n], L['c_kind'][:m_n]
            for k_id, k_name in ((KIND_ITEM, 'item'), (KIND_ITEM_LEFT, 'item_left'), (2, 'door')):
                d1 = np.where(rep_ == 0, lab_, np.nan)
                d2 = np.where(rep_ == 1, lab_, np.nan)
                ok_ = (kind_ == k_id) & np.isfinite(d1).any(1) & np.isfinite(d2).any(1)
                if ok_.sum() >= 5:
                    v_ = float(np.mean((np.nanmax(d1[ok_], 1) - np.nanmax(d2[ok_], 1)) ** 2) / 2)
                    L['noise'][k_name] = (v_, int(ok_.sum()))
                    if sampler.aux is not None:
                        sampler.aux[n, k_id] = v_
        L['choice_new'] = choice_new
        corr_batch = None
        if branching and choice_new and args.choice_adv_coef > 0:
            # Phase B2, the exploit path: a new choice record whose own option was worse (or better) than the
            # alternative by more than --choice-adv-min and 2 noise s.e. corrects the advantage of its decision record
            # (the action decided there, with the actor's log-probability then): PPO's clipped term with that advantage
            sel_c = []
            for j_ in at_c:
                lab_j = L['c_lab'][j_][np.isfinite(L['c_lab'][j_])]
                if not len(lab_j) or L['c_kind'][j_] not in (KIND_ITEM, KIND_ITEM_LEFT):
                    continue
                kname = 'item' if L['c_kind'][j_] == KIND_ITEM else 'item_left'
                s2 = L['noise'].get(kname, (args.branch_noise_prior, 0))[0]
                a_own = float(L['c_sign'][j_] * lab_j.mean())
                if abs(a_own) >= args.choice_adv_min and abs(lab_j.mean()) > 2 * math.sqrt(s2 / len(lab_j)):
                    sel_c.append((j_, max(-args.choice_adv_max, min(args.choice_adv_max, a_own))))
            if sel_c:
                jj = np.array([j_ for j_, _ in sel_c])
                mm = L['choice_meta'][jj]
                c0 = C_HI['pend0']
                corr_batch = dict(
                    batch=to_batch(L['choice_rows'][jj], mm[:, c0:c0 + nh].astype(np.int64), device),
                    act=torch.from_numpy(mm[:, C_HI['act0']:C_HI['act0'] + nh].astype(np.int64)).to(device),
                    logp=torch.from_numpy(mm[:, C_HI['logp_b']].astype(np.float32)).to(device),
                    adv=torch.tensor([a_ for _, a_ in sel_c], dtype=torch.float32, device=device))
        L['corr_n'] = 0 if corr_batch is None else len(corr_batch['adv'])
        L['corr_abs'] = 0.0 if corr_batch is None else float(corr_batch['adv'].abs().mean())

        def train_choice():
            """Phase B2: the choice heads on the held choice records (detached player tokens of the learner's
            weights after this update's PPO steps; Huber on each pair's D, bootstrap weights per head; the held-out
            points never trained) and their held-out accuracy (sign of the records' mean D), calibration (slope of
            the mean D on the prediction, correlation) and the noise floor of that comparison."""
            m_ = L['choice_n']
            out_ = {}
            if copt is None or m_ == 0:
                return out_
            lab = L['c_lab'][:m_]
            kind = L['c_kind'][:m_]
            idx = np.flatnonzero(np.isfinite(lab).any(1) & ((kind == KIND_ITEM) | (kind == KIND_ITEM_LEFT)))
            out_['choice_labeled'] = len(idx)
            if len(idx) < 8:
                return out_
            meta_ = L['choice_meta'][idx]
            pend_ = meta_[:, C_HI['pend0']:C_HI['pend0'] + nh].astype(np.int64)
            feats = []
            with torch.no_grad():
                for at in range(0, len(idx), 512):
                    b_ = to_batch(L['choice_rows'][idx[at:at + 512]], pend_[at:at + 512], device)
                    with autocast():
                        feats.append(learner_model(b_, need_h=True)[3].float())
            H = torch.cat(feats)
            item_t = torch.from_numpy(L['c_item'][idx]).to(device)
            lab_i = lab[idx]
            train_m = ~L['c_hold'][idx]
            rr, qq = np.nonzero(np.isfinite(lab_i))
            keep_ = train_m[rr]
            s_r, s_q = rr[keep_], qq[keep_]
            noise_all = [v[0] for v in L['noise'].values()]
            delta = math.sqrt(float(np.mean(noise_all))) if noise_all else math.sqrt(args.branch_noise_prior)
            out_['choice_train_pairs'] = len(s_r)
            if len(s_r) >= 8:
                labels_t = torch.from_numpy(lab_i[s_r, s_q].astype(np.float32)).to(device)
                rows_t = torch.from_numpy(s_r).to(device)
                boot_t = torch.from_numpy(L['c_boot'][idx][s_r]).to(device)
                losses = []
                learner_model.choice.train()
                for _ in range(args.choice_steps):
                    pk = torch.from_numpy(crng.integers(0, len(s_r), min(args.choice_batch, len(s_r)))).to(device)
                    r_ = rows_t[pk]
                    pred = learner_model.choice.take_minus_skip(H[r_], item_t[r_])
                    err = torch.nn.functional.huber_loss(pred, labels_t[pk][:, None].expand_as(pred),
                                                         reduction='none', delta=delta)
                    w_ = boot_t[pk]
                    loss_ = (err * w_).sum() / w_.sum().clamp(min=1.0)
                    copt.zero_grad(set_to_none=True)
                    loss_.backward()
                    copt.step()
                    losses.append(loss_.detach())
                copt.zero_grad(set_to_none=True)   # (no stale gradients in PPO's clip_grad_norm_ over parameters())
                out_['choice_loss'] = float(torch.stack(losses).mean())
            with torch.no_grad():
                P = learner_model.choice.take_minus_skip(H, item_t).float()
                pm, ps = P.mean(1).cpu().numpy(), P.std(1).cpu().numpy()
            dbar = np.nanmean(lab_i, 1)
            npair = np.isfinite(lab_i).sum(1)
            s2 = np.array([L['noise'].get('item' if k == KIND_ITEM else 'item_left', (args.branch_noise_prior, 0))[0]
                           for k in kind[idx]])
            sig = np.abs(dbar) > 2 * np.sqrt(s2 / np.maximum(npair, 1))
            out_['choice_spread'] = float(ps.mean())
            for name_, sel_ in (('held', ~train_m), ('train', train_m)):
                nz = sel_ & (dbar != 0)
                out_[f'choice_{name_}_n'] = int(sel_.sum())
                if nz.sum() >= 3:
                    out_[f'choice_{name_}_acc'] = float((np.sign(pm[nz]) == np.sign(dbar[nz])).mean())
                    out_[f'choice_{name_}_mse'] = float(np.mean((pm[sel_] - dbar[sel_]) ** 2))
                    out_[f'choice_{name_}_floor'] = float(np.mean(s2[sel_] / np.maximum(npair[sel_], 1)))
                    if np.std(pm[sel_]) > 0 and np.std(dbar[sel_]) > 0:
                        out_[f'choice_{name_}_corr'] = float(np.corrcoef(pm[sel_], dbar[sel_])[0, 1])
                        out_[f'choice_{name_}_slope'] = float(np.polyfit(pm[sel_], dbar[sel_], 1)[0])
                ns = sel_ & sig
                out_[f'choice_{name_}_sig_n'] = int(ns.sum())
                if ns.sum() >= 3:
                    out_[f'choice_{name_}_acc_sig'] = float((np.sign(pm[ns]) == np.sign(dbar[ns])).mean())
            return out_

        def teacher_loss(t_logits, t_danger, meta, n_sel):
            """The teacher's losses on its records (weighted for a minibatch of n_sel decisions) and their stats:
            the danger head learns which held moves got the player hurt; the move head is pushed to put its mass on
            the moves that did not (where there was one)."""
            hurt_bits = meta[:, 3:12].float()
            danger_loss = torch.nn.functional.binary_cross_entropy_with_logits(t_danger.float(), hurt_bits)
            move_logp = torch.log_softmax(t_logits.float()[:, :9], -1)
            has_safe = meta[:, 14] > 0
            safe_mass = torch.logsumexp(move_logp.masked_fill(hurt_bits > 0, -1e9), -1)
            teach_loss = -(safe_mass * has_safe).sum() / has_safe.sum().clamp(min=1)
            share = args.teach_weight * teach_step / n_sel
            stats_ = torch.stack([teach_loss, danger_loss,
                                  (safe_mass.exp() * has_safe).sum() / has_safe.sum().clamp(min=1)]).detach()
            return share * (args.danger_coef * danger_loss + args.teach_coef * teach_loss), stats_
        ent_heads = torch.zeros(nh, device=device)
        # the records' entity and door counts on the host: the minibatches are planned here (orders, entity tokens,
        # packed token positions) and the learner thread never waits for the GPU inside an epoch
        ne_h = rl.rows['n_ent'].reshape(-1)[flat].astype(np.int64)
        nd_h = rl.rows['n_doors'].reshape(-1)[flat].astype(np.int64)
        width = batch['ent'].shape[1]
        # B18: the records' ids among the rollout's distinct room grids, level maps and player patches (the CNNs then
        # run on a minibatch's distinct ones: ~380 of 4,096 rooms and maps, ~2,100 patches in a run-mode rollout)
        dd = {k_: distinct_ids(batch[k_]).cpu().numpy() for k_ in DEDUP} if 'dedup' in fparts else {}
        dd_size = {k_: int(v_.max()) + 1 for k_, v_ in dd.items()}

        def dev(a):
            return torch.from_numpy(np.ascontiguousarray(a, np.int64)).pin_memory().to(device, non_blocking=True)

        def dev_many(arrays):
            """(B18) int64 arrays to the device in one pinned copy: views of it, in order."""
            on = torch.from_numpy(np.concatenate([np.asarray(a, np.int64) for a in arrays])).pin_memory()
            return list(on.to(device, non_blocking=True).split([len(a) for a in arrays]))
        tokens = parts = 0
        t_meta = torch.from_numpy(teach_meta[:max(teach_n, 1)]).to(device) if fast else None   # (B18)
        learner_model.train()
        corr_stats = torch.zeros(2, device=device)   # Phase B2: the correction's loss and clipped share (sums)
        for _ in range(args.epochs):
            perm = torch.randperm(count, device=device)
            perm_h = perm.cpu().numpy()
            first_step = True
            for at in range(0, count, args.minibatch):
                sel_h = perm_h[at:at + args.minibatch]
                if len(sel_h) < args.minibatch // 4:
                    continue
                sel = perm[at:at + args.minibatch]
                a_all = b_adv[sel]
                adv_std = a_all.std()
                a_all = (a_all - a_all.mean()) / (a_all.std() + 1e-8)
                opt.zero_grad(set_to_none=True)
                micro = args.micro or len(sel)
                if micro < len(sel) and args.sort_micro:
                    # the minibatch's rows ordered by their entity count before they are cut into parts: each part
                    # then pads to fewer entity tokens. The minibatch's gradient is the same sum over its rows
                    # (numpy's stable argsort: the order torch.argsort(stable=True) gives on the device)
                    order_h = np.argsort(ne_h[sel_h], kind='stable')
                    order = dev(order_h)
                    sel, a_all, sel_h = sel[order], a_all[order], sel_h[order_h]
                teaching = teach_step >= 8 and teach_n >= 256
                tb = None
                if teaching:
                    # the teacher's records of this step (drawn here, before the PPO parts: the same draws as when
                    # they were drawn after them, nothing else uses numpy's generator)
                    pick = np.random.randint(0, teach_n, teach_step)
                    if fast:   # B18: from the meta's copy on the device (no blocking copy)
                        pick_d = dev(pick)
                        meta = t_meta.index_select(0, pick_d)
                    else:
                        meta = torch.from_numpy(teach_meta[pick]).to(device)
                        pick_d = None
                    if raw_mode:   # (B18: merged as bytes below)
                        tb = pick_d
                    elif teach_gpu:
                        tb = decode_rows(L['teach_raw'].index_select(0, dev(pick) if pick_d is None else pick_d),
                                         pending=teach_pending(meta).clone(),
                                         entities=min(ENT_CAP, used_entities(teach_rows['n_ent'][pick])))
                    else:
                        tb = to_batch(teach_rows[pick], teach_pending(teach_meta[pick]), device)
                    sec('teach_build')
                last = (len(sel) - 1) // micro * micro
                for lo_ in range(0, len(sel), micro):   # the minibatch's means, a part at a time
                    part = sel[lo_:lo_ + micro]
                    part_h = sel_h[lo_:lo_ + micro]
                    w = len(part) / len(sel)
                    used = min(width, used_entities(ne_h[part_h]))
                    merged = tb is not None and args.teach_merge and lo_ == last
                    if raw_mode:
                        # B18: the part's records (and the teacher's) as bytes, the fields views of them (decode_rows)
                        rp, pp, e_ = raw_all.index_select(0, part), pend_all.index_select(0, part), used
                        if merged:
                            e_ = max(used, min(ENT_CAP, used_entities(teach_rows['n_ent'][pick])))
                            rp = torch.cat([rp, L['teach_raw'].index_select(0, pick_d)])
                            pp = torch.cat([pp, teach_pending(meta)])
                        mb = decode_rows(rp, pending=pp, entities=e_)
                    elif fast:   # (B18: the entity fields cut before the gather)
                        mb = {k: (v[:, :used] if k in ENTITY_FIELDS else v)[part] for k, v in batch.items()}
                    else:
                        mb = {k: v[part] for k, v in batch.items()}   # take(batch, part), entity tokens from the host
                        for k_e in ENTITY_FIELDS:
                            mb[k_e] = mb[k_e][:, :used]
                    tokens += used
                    parts += 1
                    ne_p, nd_p = ne_h[part_h], nd_h[part_h]
                    if merged:   # the teacher's records ride along in the last part: one forward and backward
                        if not raw_mode:
                            mb = cat_batch(mb, tb)
                        ne_p = np.concatenate([ne_p, teach_rows['n_ent'][pick]])
                        nd_p = np.concatenate([nd_p, teach_rows['n_doors'][pick]])
                    if args.packed and fast:   # B18: + the entity positions and the attention buckets, one copy
                        plan = learner_plan(ne_p, nd_p, mb['ent'].shape[1], n_buckets)
                        hn = ['rec', 'base']
                        for k_ in dd:   # the distinct grids of this part (the teacher's records each their own)
                            u_, i_ = first_of(dd[k_][part_h], dd_size[k_])
                            if merged:
                                t_ = len(mb['n_ent']) - len(part)
                                u_, i_ = (np.concatenate([u_, len(part) + np.arange(t_)]),
                                          np.concatenate([i_, len(u_) + np.arange(t_)]))
                            plan[k_ + '_u'], plan[k_ + '_i'] = u_, i_
                            hn += [k_ + '_u', k_ + '_i']
                        bk_ = plan['buckets'] if 'buckets' in fparts else []
                        got = dev_many([plan[k_].reshape(-1) for k_ in hn] + [c_ for c_, _ in bk_])
                        got[0] = got[0].view(-1, 6)
                        mb.update(zip(hn[2:], got[2:len(hn)]))
                        tok = plan_tokens(plan, got[0], got[1])
                        pnames = ['keep', 'starts']
                        if 'ekeep' in fparts:
                            pnames.append('ekeep')
                        if 'buckets' in fparts:
                            pnames.append('gpos')
                            if 'ekeep' in fparts:   # the packed tokens built from the real ones (forward's xperm)
                                pnames += ['dkeep', 'xperm']
                        mb.update((k_, tok[k_]) for k_ in pnames)
                        if bk_:
                            mb['buckets'] = [(c_, w_) for c_, (_, w_) in zip(got[len(hn):], bk_)]
                    elif args.packed:
                        mb['keep'], mb['starts'] = (dev(v) for v in packed_index(ne_p, nd_p, mb['ent'].shape[1]))
                    k_ = len(part)
                    with autocast():
                        logits_, value_, danger_ = learner_model(mb, packed=bool(args.packed))
                    logp, entropy = (learner_model.log_prob_fast if 'logp' in fparts else learner_model.log_prob)(
                        logits_[:k_], learner_model.gate(mb)[:k_], b_act[part])
                    value = value_[:k_].float()
                    a = a_all[lo_:lo_ + micro]
                    if prox is None:   # PPO's ratio to the behaviour policy (the one that sampled the action)
                        ratio = torch.exp(logp - b_logp[part])
                        pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                    else:   # decoupled: clipped around the weights at the update's start, importance-weighted
                        ratio = torch.exp(logp - prox[part])
                        weight = torch.exp(prox[part] - b_logp[part]).clamp(max=args.decoupled_max_weight)
                        pg = -(weight * torch.min(ratio * a, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a)
                               ).mean()
                    vf = 0.5 * ((value - b_ret[part]) ** 2).mean()
                    ent = entropy.sum(-1).mean()
                    loss = (pg + args.vf_coef * vf - args.ent_coef * ent) * w
                    if merged:
                        tloss, tstats = teacher_loss(logits_[k_:], danger_[k_:], meta, len(sel))
                        loss = loss + tloss
                    loss.backward()
                    with torch.no_grad():
                        acc += torch.stack([pg, vf, ent, (b_logp[part] - logp).mean(),
                                            ((ratio - 1).abs() > args.clip).float().mean()]) * w
                        ent_heads += entropy.mean(0) * w
                        if merged:
                            tacc += tstats
                            teach_steps += 1
                if corr_batch is not None and first_step:
                    # Phase B2: the choice records' correction, once per epoch (its records count as decisions of
                    # this minibatch, their advantage in the minibatch's units)
                    with autocast():
                        lg_c = learner_model(corr_batch['batch'], packed=bool(args.packed))[0]
                    lp_c, _ = learner_model.log_prob(lg_c, learner_model.gate(corr_batch['batch']), corr_batch['act'])
                    ratio_c = torch.exp(lp_c - corr_batch['logp'])
                    a_c = corr_batch['adv'] / (adv_std + 1e-8)
                    loss_c = -args.choice_adv_coef * torch.min(
                        ratio_c * a_c, torch.clamp(ratio_c, 1 - args.clip, 1 + args.clip) * a_c).sum() / len(sel)
                    loss_c.backward()
                    with torch.no_grad():
                        corr_stats += torch.stack([loss_c.detach(),
                                                   ((ratio_c - 1).abs() > args.clip).float().mean()])
                first_step = False
                steps += 1
                sec('ppo')
                if tb is not None and not args.teach_merge:
                    with autocast():
                        t_logits, _, t_danger = learner_model(tb, packed=bool(args.packed))
                    tloss, tstats = teacher_loss(t_logits, t_danger, meta, len(sel))
                    tloss.backward()
                    with torch.no_grad():
                        tacc += tstats
                    teach_steps += 1
                    sec('teach')
                if L.get('trace') is not None:   # B18 check: the loss sums after each step, the first step's gradient
                    L['trace'].append((acc.clone(), tacc.clone(), None if L['trace'] else
                                       [None if q.grad is None else q.grad.clone()
                                        for q in opt.param_groups[0]['params']]))
                torch.nn.utils.clip_grad_norm_(learner_model.parameters(), args.max_grad_norm)
                opt.step()
                sec('opt')
        learner_model.eval()
        choice_log = train_choice() if branching else {}   # Phase B2 (before the actor takes the new weights)
        torch.cuda.current_stream().synchronize()
        sec('tail')
        update_s = time.perf_counter() - t0
        cpu_end = cpu_ticks()
        job['cpu0'] = cpu0
        L['update'] += 1
        L['decisions'] += count
        L['last_update_s'], L['last_steps'], L['last_count'], L['last_U'] = update_s, steps, count, U   # (B18 bench)
        if not L.get('bench'):
            fresh.set()                           # the actor may take the new weights now
        # ---- logging (the actor is already running on)
        t_end = time.perf_counter()
        wall = t_end - (L['t_end'] if L['t_end'] is not None else job['t_start'])
        L['t_end'] = t_end
        L['seconds'] += wall
        upd = L['update']
        acc_c, tacc_c = acc.cpu().numpy(), tacc.cpu().numpy()
        var = float(b_ret.var())
        ev = 1.0 - float((b_ret - b_val).var()) / var if var > 0 else 0.0
        steps_ = max(steps, 1)
        tsteps = max(teach_steps, 1)
        prof = job['prof']
        calls = max(prof['calls'], 1)
        dst = job['worker']
        stats = sampler.stats
        finished = job['finished']
        row = {
            'update': upd, 'decisions': L['decisions'], 'game_hours': L['decisions'] * hours_per_decision,
            'x_real_time': count * args.frames_per_decision / 30 / max(wall, 1e-9),
            'collect_s': job['collect_s'], 'update_s': update_s, 'infer_ms': 1000 * prof['latency'] / calls,
            'rows_per_call': prof['rows'] / calls, 'eager_calls': prof['eager'], 'lr': lr,
            'pg': acc_c[0] / steps_, 'vf': acc_c[1] / steps_, 'entropy': acc_c[2] / steps_,
            'ent_move': float(ent_heads[0]) / steps_, 'ent_shoot': float(ent_heads[1]) / steps_,
            'ent_bomb': float(ent_heads[2]) / steps_, 'kl': acc_c[3] / steps_,
            **({'ent_item': float(ent_heads[3]) / steps_, 'ent_pill': float(ent_heads[4]) / steps_} if nh > 3 else {}),
            'clip_frac': acc_c[4] / steps_, 'explained_variance': ev,
            'reward_mean': float(rl.reward.reshape(-1)[flat].mean()), 'episodes': len(finished),
            'errors': float(stats[:, 5].sum()),
            'stalls': float(stats[:, ST['stalls']].sum()), 'build_retries': float(stats[:, ST['retries']].sum()),
            'port_moves': float(stats[:, ST['port_moves']].sum()), 'root_recycles': float(stats[:, ST['recycles']].sum()),
            'parked_trims': float(stats[:, ST['trims']].sum()),
            'teach_new': teach_new, 'teach_held': teach_n, 'teach_step': teach_step, 'safe_mass_new': safe_new,
            'teach_loss': tacc_c[0] / tsteps if teach_steps else 0.0,
            'danger_loss': tacc_c[1] / tsteps if teach_steps else 0.0,
            'safe_mass': tacc_c[2] / tsteps if teach_steps else 0.0,
            'searches': float(stats[:, 18].sum()), 'avoidable': float(stats[:, 19].sum()),
            'hurts': float(stats[:, 20].sum()), 'teach_s': float(stats[:, 17].sum()),
            'snaps': float(stats[:, 21].sum()), 'archive': float(stats[:, 22].sum()),
            'archive_starts': float(stats[:, 23].sum()),
            'wall_s': wall, 'overlap': int(args.overlap), 'stale_share': stale,
            'ent_tokens': tokens / max(parts, 1), 'ent_over32': float((ne_h > 32).mean()),
            'gpu_reserved_gb': torch.cuda.max_memory_reserved() / 2 ** 30,
            'wait_learner_s': job['wait_learner_s'],
            'infer_upd_ms': 1000 * prof['act_busy'] / max(prof['calls_busy'], 1), 'calls_upd': prof['calls_busy'],
            'p_poll': prof['poll'], 'p_book': prof['book'], 'p_act': prof['act'], 'p_reply': prof['reply'],
            'p_store': prof['store'], 'polls': prof['polls'],
            'idle_share': dst['wait_s'] / max(n * job['cycle_s'], 1e-9), 'w_step': dst['step_s'],
            'w_wait': dst['wait_s'], 'w_encode': dst['encode_s'], 'w_fork': dst['fork_s'], 'w_teach': dst['teach_s'],
            'w_overrun': dst.get('overrun_s', 0.0), 'w_state': dst.get('state_s', 0.0),
            'w_close': dst.get('close_s', 0.0), 'w_total': n * job['cycle_s'],
            'q_teach': float(stats[:, ST['queue']].sum()), 'teach_dropped': float(stats[:, ST['dropped']].sum()),
            'cpu_collect': job.get('cpu_collect', 0.0), 'cpu_update': cpu_share(job['cpu0'], cpu_end),
            'c_wait_learner': job.get('cycle_wait_s', 0.0), 'c_dec_min': job.get('dec_min', 0.0),
            'c_dec_max': job.get('dec_max', 0.0), 'dec_w': job.get('dec_w', ''),
            'c_trainer_cores': job.get('proc_cores', 0.0),
        }
        row.update({f'u_{k}': v for k, v in U.items()})
        row['u_log_prev'], row['u_ckpt_prev'] = L.get('log_s', 0.0), L.get('ckpt_s', 0.0)
        if run or items:   # 2026-10-06: run / item counters (absent in the other modes: their columns are unchanged)
            m_all = max(len(finished), 1)
            row.update({'items_taken': sum(e.get('items', 0) for e in finished),
                        'active_uses': sum(e.get('uses', 0) for e in finished),
                        'pills_used': sum(e.get('pills', 0) for e in finished),
                        'exits': sum(e.get('exits', 0) for e in finished),
                        'stage_max': max([e.get('stage', 0) for e in finished] or [0]),
                        'stage_mean': sum(e.get('stage', 0) for e in finished) / m_all})
            if cfg.start_build_prob > 0:   # 2026-10-08: start builds transplanted so far (worker totals)
                row['start_builds'] = float(stats[:, ST['start_builds']].sum())
        if cfg.stat_aug_prob > 0:   # 2026-10-08: stat augmentations applied so far (worker totals)
            row['stat_augs'] = float(stats[:, ST['stat_augs']].sum())
        if cfg.teacher_death_depths:   # 2026-10-09: fatal hurts searched / with a safe move / their seconds (totals)
            for k_d in ('death_searches', 'death_avoidable', 'death_s'):
                row[k_d] = float(stats[:, ST[k_d]].sum())
        if char_ids:   # 2026-10-08 (characters): floor starts built as non-Isaac (worker totals) and, per character,
            #            this update's finished episodes and their floor clears
            row['char_starts'] = float(stats[:, ST['char_starts']].sum())
            for c_id in char_ids:
                eps_c = [e for e in finished if e.get('char0') == c_id]
                row[f'char_{c_id}/episodes'] = len(eps_c)
                row[f'char_{c_id}/win'] = sum(e['done'] == 1 for e in eps_c) / max(len(eps_c), 1)
        if book.gains is not None:   # 2026-10-08: heal / resource terms on: per finished episode
            m_all = max(len(finished), 1)
            row['healed'] = sum(e.get('healed', 0.0) for e in finished) / m_all
            row['gathered'] = sum(e.get('gathered', 0.0) for e in finished) / m_all
            for k_st in range(1, 9):
                row[f'stage_{k_st}'] = sum(e.get('stage', 0) == k_st for e in finished)
        if branching:   # 2026-10-07: branch counters (worker totals so far) and this update's branch episodes
            for k_b in ('bp_item', 'bp_door', 'bp_item_left', 'bp_exit', 'bp_item_far', 'bp_dropped', 'b_points',
                        'b_branches', 'b_dec', 'b_frames', 'b_s', 'b_cpu', 'b_prep_s', 'b_invalid', 'b_walk_fail',
                        'b_skip_took', 'b_errors', 'choices', 'b_queue', 'b_replicates', 'b_ped_seen', 'b_game_cpu'):
                row[k_b] = float(stats[:, ST[k_b]].sum())
            row['actual_game_hours'] = float(stats[:, ST['frames']].sum()) / 30 / 3600
            row['branch_game_hours'] = float(stats[:, ST['b_frames']].sum()) / 30 / 3600
            row['branch_episodes'] = len(job.get('branch_finished', []))
            row['choices_new'] = L.get('choice_new', 0)
            row['choices_held'] = L.get('choice_n', 0)
            for k_b in ('b_shop_skip', 'b_unreach_skip', 'b_twins', 'b_twin_act_diff', 'b_pairs', 'b_noise_pairs',
                        'b_cond_pairs', 'b_archived'):
                row[k_b] = float(stats[:, ST[k_b]].sum())
            for k_n in ('item', 'item_left'):
                v_n = L['noise'].get(k_n)
                row[f'noise_var_{k_n}'] = v_n[0] if v_n else ''
                row[f'noise_n_{k_n}'] = v_n[1] if v_n else 0
            row['adv_corr_n'], row['adv_corr_abs'] = L.get('corr_n', 0), L.get('corr_abs', 0.0)
            row['snap_calls'], row['snap_miss'] = prof.get('snap_calls', 0), prof.get('snap_miss', 0)
            cs_ = corr_stats.cpu().numpy()
            row['adv_corr_loss'], row['adv_corr_clip'] = float(cs_[0]), float(cs_[1]) / max(args.epochs, 1)
            for k_c in ('choice_labeled', 'choice_train_pairs', 'choice_loss', 'choice_spread', 'choice_held_n',
                        'choice_held_acc', 'choice_held_acc_sig', 'choice_held_sig_n', 'choice_held_mse',
                        'choice_held_floor', 'choice_held_corr', 'choice_held_slope', 'choice_train_n',
                        'choice_train_acc', 'choice_train_acc_sig', 'choice_train_mse', 'choice_train_floor',
                        'choice_train_corr', 'choice_train_slope'):
                row[k_c] = choice_log.get(k_c, '')
        if lab is not None:   # 2026-10-08: the lab's counters (worker totals so far) and this update's builds
            for k_l in ('lab_jobs', 'lab_states', 'lab_dec', 'lab_s', 'lab_cpu', 'lab_game_cpu', 'lab_errors',
                        'lab_results', 'lab_panel', 'lab_panel_s', 'lab_panel_builds'):
                row[k_l] = float(stats[:, ST[k_l]].sum())
            row['actual_game_hours'] = float(stats[:, ST['frames']].sum()) / 30 / 3600
            row['lab_game_hours'] = float(stats[:, ST['lab_frames']].sum()) / 30 / 3600
            row['lab_episodes'] = len(job.get('lab_finished', []))
            row['lab_posted'], row['lab_open'] = lab['posted'], len(lab['book'].open)
            row.update(lab_learn(job.get('lab_results', [])))
        for gi, name in enumerate(names):
            eps = [e for e in finished if e['group'] == gi]
            m = max(len(eps), 1)
            row.update({f'{name}/episodes': len(eps),
                        f'{name}/win': sum(e['done'] == 1 for e in eps) / m,
                        f'{name}/death': sum(e['done'] == 2 for e in eps) / m,
                        f'{name}/timeout': sum(e['done'] == 3 for e in eps) / m,
                        f'{name}/hurt': sum(e['hurt'] for e in eps) / m,
                        f'{name}/nohit_win': sum(e['done'] == 1 and e['hurt'] == 0 for e in eps) / m,
                        f'{name}/return': sum(e['ret'] for e in eps) / m,
                        f'{name}/seconds': sum(e['decisions'] for e in eps) / m * args.frames_per_decision / 30,
                        f'{name}/rooms': sum(e['rooms'] for e in eps) / m,
                        f'{name}/boss': sum(e['boss'] for e in eps) / m})
        if L['writer'] is None:
            columns = log_columns(list(row))
            L['writer'] = csv.DictWriter(log_file, fieldnames=columns, restval='')
            if log_file.tell() == 0:
                L['writer'].writeheader()
        L['writer'].writerow(row)
        log_file.flush()
        for e in finished:
            episodes_path.write(json.dumps(dict(e, update=upd)) + '\n')
        episodes_path.flush()
        if branch_path is not None:
            for e in job.get('branch_finished', []):
                branch_path.write(json.dumps(dict(e, update=upd)) + '\n')
            branch_path.flush()
        if upd % 5 == 0 or upd == 1:
            wins = ' '.join(f"{nm} win {row[f'{nm}/win']:.2f} hurt {row[f'{nm}/hurt']:.2f}" for nm in names)
            print(f"u{upd} {row['game_hours']:.1f} gh {row['x_real_time']:.0f}x collect {job['collect_s']:.1f}s "
                  f"update {update_s:.1f}s idle {row['idle_share']:.2f} infer {row['infer_ms']:.2f}/"
                  f"{row['infer_upd_ms']:.2f}ms kl {row['kl']:.4f} ev {ev:.2f} ent {row['entropy']:.2f} | {wins}",
                  flush=True)
        t_ck = time.perf_counter()
        L['log_s'] = t_ck - t_end
        if upd % args.checkpoint_every == 0:
            save(out / 'checkpoints' / f'update-{upd:06d}.pt')
        L['ckpt_s'] = time.perf_counter() - t_ck

    jobs = queue.Queue()
    fresh = threading.Event()        # set by the learner when new weights are final
    failure = []

    def learner_loop():
        torch.cuda.set_stream(learner_stream)
        if learner_cpus:   # (B18: this thread, and the threads it starts, on --learner-cpus)
            os.sched_setaffinity(0, learner_cpus)
        while True:
            job = jobs.get()
            if job is None:
                return
            try:
                if os.environ.get('ISAAC_RL_LEARNER_BENCH') and not L.get('stop'):
                    # 2026-10-10 (B18): the learner bench on this job (abplus_bench_learner.py), then the run ends
                    from abplus_bench_learner import run_bench
                    if run_bench(dict(learn=learn, job=job, model=learner_model, opt=opt, L=L, args=args,
                                      stats=sampler.stats.copy())):
                        fresh.set()
                        continue
                learn(job)
            except BaseException as exc:   # handed to the main thread
                failure.append(exc)
                fresh.set()
                return

    learner = threading.Thread(target=learner_loop, name='learner', daemon=True)
    learner.start()

    # ------------------------------------------------------------------ actor (the main thread)
    pending = np.zeros((n, nh), np.int64)            # the action under way per worker
    book = Episodes(n, rw, floor, run=run, items=items)
    if cfg.stat_aug_prob > 0:   # 2026-10-08: the start stats of every episode into episodes.jsonl
        book.stats0 = {}
    if char_ids:   # 2026-10-08: the character of every episode into episodes.jsonl (char0)
        book.char0 = {}
    version = update                                # the actor's weights: updates done
    busy = False                                     # an update is running on the learner
    handed = decisions_total
    wcols = ('step_s', 'wait_s', 'encode_s', 'fork_s', 'teach_s', 'overrun_s', 'state_s', 'close_s')
    sync_wait = 0.0

    def take_weights():
        nonlocal version, busy
        fresh.clear()
        if failure:
            raise failure[0]
        with torch.no_grad():
            for a, b in zip(actor_params, learner_params):
                a.copy_(b)
        torch.cuda.current_stream().synchronize()
        version += 1
        sampler.control[3] = version   # 2026-10-07: the actor's weights, for the workers' choice records
        if snaps and version % args.branch_snap_every == 0:   # Phase B2: the older snapshot takes these weights
            e = min(snaps, key=lambda e_: e_['id'])
            with torch.no_grad():
                for a, b in zip(e['params'], actor_params):
                    a.copy_(b)
            torch.cuda.current_stream().synchronize()
            e['id'] = version
            aux[n, AUX_SNAP] = version
        busy = False

    def wait_learner():
        while not fresh.wait(0.5):
            if failure or not learner.is_alive():
                break
        take_weights()

    aux = sampler.aux   # Phase B2 (branches): the actor's value, log-probability and choice outputs per worker
    from isaac_bridge.tok_branch import AUX_REQ, AUX_SNAP, AUX_SNAP_ON, AUX_USED
    snaps = []   # Phase B2: the weight snapshots the branch records are answered with (tok_branch, aux block)
    if aux is not None and args.branch_snap_every > 0 and args.graph_entities > 0:
        for k_s in range(2):
            m_s = TokPolicy(args.width, args.layers, args.heads, items=items, charge=charge,
                            choice=n_choice, ent_ext=ent_ext, pchar=pchar).to(device).eval()
            m_s.load_state_dict(model.state_dict())
            e_s = dict(id=version if k_s == 0 else -1, model=m_s, actors=[],
                       params=list(m_s.parameters()) + list(m_s.buffers()))
            for _ in range(max(1, args.actor_slots)):   # (calls in flight per snapshot, as the actor's own)
                g_s = GraphActor(m_s, n, args.graph_entities, stream=torch.cuda.Stream(priority=-5),
                                 crn=bool(args.branch_crn))
                g_s.snap, g_s.in_flight = e_s, False
                e_s['actors'].append(g_s)
            snaps.append(e_s)
        aux[n, AUX_SNAP], aux[n, AUX_SNAP_ON] = version, 1
        torch.cuda.synchronize()

    def answer(prof, a_idx, sub, under_way, actions, logp, value, extra=None, ver=None):
        """The actions out to the workers, the decisions into the rollout. ver: the weights that answered (a
        snapshot's id; None: the actor's own)."""
        t2 = time.perf_counter()
        sampler.actions[a_idx, :nh] = actions
        if aux is not None:   # (written before the reply: the worker reads it after its answer)
            aux[a_idx, 0], aux[a_idx, 1] = value, logp
            if extra is not None:
                aux[a_idx, 2], aux[a_idx, 3] = extra[:, 0], extra[:, 1]
            else:
                aux[a_idx, 2:4] = 0.0
            aux[a_idx, AUX_USED] = version if ver is None else ver
        sampler.reply(a_idx)
        t3 = time.perf_counter()
        prof['reply'] += t3 - t2
        store(roll, a_idx, sub, under_way, actions, logp, value, version if ver is None else ver)
        pending[a_idx] = actions
        prof['store'] += time.perf_counter() - t3

    def finish(prof):
        """Wait for the oldest call in flight and answer it."""
        slot, a_idx, sub, under_way, t_sub, was_busy = inflight.popleft()
        t = time.perf_counter()
        actions, logp, value = slot.result()
        t2 = time.perf_counter()
        ver = None
        if getattr(slot, 'snap', None) is not None:   # a snapshot's actor (Phase B2)
            slot.in_flight = False
            ver = slot.snap['id']
        else:
            free.append(slot)
        prof['act'] += t2 - t
        prof['latency'] += t2 - t_sub
        if was_busy:
            prof['act_busy'] += t2 - t_sub
        answer(prof, a_idx, sub, under_way, actions, logp, value, slot.extra, ver)

    def submit_snapshots(prof, idx_b, sub_b, uw_b, t1):
        """Phase B2: branch records to the snapshots their workers ask for (the newest when that one is gone)."""
        req = aux[idx_b, AUX_REQ].astype(np.int64)
        for sid in np.unique(req):
            m_ = req == sid
            e = next((e_ for e_ in snaps if e_['id'] == sid), None)
            if e is None:
                e = max(snaps, key=lambda e_: e_['id'])
                prof['snap_miss'] += int(m_.sum())
            while all(g_.in_flight for g_ in e['actors']):
                finish(prof)
            g = next(g_ for g_ in e['actors'] if not g_.in_flight)
            g.submit(sub_b[m_], uw_b[m_])
            g.in_flight = True
            inflight.append((g, idx_b[m_], sub_b[m_], uw_b[m_], t1, busy))
            prof['snap_calls'] += 1

    if lab is not None:   # 2026-10-08: the first lab jobs, and the builds ordinary episodes end with
        book.seen_inv = []
        post_lab_jobs()
    t_run = time.perf_counter()
    t_cycle, w_mark = t_run, sampler.stats.copy()
    proc_mark = time.process_time()   # this process's CPU seconds (all threads)
    try:
        if bench_job is not None:   # B18: the saved job straight to the learner's bench, in this thread (no collection)
            from abplus_bench_learner import run_bench
            torch.cuda.set_stream(learner_stream)
            run_bench(dict(learn=learn, job=bench_job, model=learner_model, opt=opt, L=L, args=args,
                           stats=sampler.stats.copy()))
            torch.cuda.set_stream(actor_stream)
        while handed < target_decisions and sampler.live and not L.get('stop'):
            # ---- collect
            t0 = time.perf_counter()
            cpu_c0 = cpu_ticks()
            prof = dict(poll=0.0, book=0.0, act=0.0, reply=0.0, store=0.0, polls=0, calls=0, rows=0, eager=0,
                        act_busy=0.0, calls_busy=0, latency=0.0, snap_calls=0, snap_miss=0)
            while roll.usable() < args.rollout and sampler.live:
                if busy and fresh.is_set():
                    while inflight:
                        finish(prof)
                    take_weights()
                ta = time.perf_counter()
                if inflight:   # a call is running: take what is ready now, else finish the oldest call
                    if not free:
                        finish(prof)
                        continue
                    idx, slots = sampler.poll_ready(idle=0.0)
                    if not len(idx):
                        prof['poll'] += time.perf_counter() - ta
                        finish(prof)
                        continue
                else:
                    idx, slots = sampler.poll_ready(idle=0.05, want=args.poll_want, max_wait=args.poll_wait)
                tb = time.perf_counter()
                prof['poll'] += tb - ta
                prof['polls'] += 1
                if not len(idx):
                    continue
                rows = sampler.rows[slots]
                go = book.arrive(roll, rows, idx, pending)
                t1 = time.perf_counter()
                prof['book'] += t1 - tb
                if not go.all():   # records that end an episode only need their answer
                    sampler.reply(idx[~go])
                if not go.any():
                    prof['reply'] += time.perf_counter() - t1
                    continue
                t1 = time.perf_counter()
                a_idx = idx[go]
                sub = rows if go.all() else rows[go]
                under_way = pending[a_idx]
                prof['calls'] += 1
                prof['rows'] += len(a_idx)
                if busy:
                    prof['calls_busy'] += 1
                if snaps:   # Phase B2: branch records go to their point's weight snapshot
                    is_b = sub['branch'] != 0
                    if is_b.any():
                        submit_snapshots(prof, a_idx[is_b], sub[is_b], under_way[is_b], t1)
                        if is_b.all():
                            prof['act'] += time.perf_counter() - t1
                            continue
                        a_idx, sub, under_way = a_idx[~is_b], sub[~is_b], under_way[~is_b]
                if free and free[-1].submit(sub, under_way):
                    inflight.append((free.pop(), a_idx, sub, under_way, t1, busy))
                    prof['act'] += time.perf_counter() - t1
                    continue
                # no graph (--graph-entities 0) or more records than it holds: the eager path, at once
                while inflight:
                    finish(prof)
                prof['eager'] += 1
                with torch.inference_mode():
                    u_crn = None
                    if args.branch_crn and sub['crn'].any():   # Phase B2: the branch records' common random numbers
                        u_crn = torch.rand((len(sub), sum(model.heads)), device=device)
                        sel_c = np.flatnonzero(sub['crn'])
                        u_crn[torch.from_numpy(sel_c).to(device)] = torch.from_numpy(
                            crn_uniforms(sub['crn'][sel_c], sub['t'][sel_c], u_crn.shape[1])).to(device)
                    actions, logp, value = model.act(to_batch(sub, under_way, device), u=u_crn)
                actions, logp, value = actions.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
                t2 = time.perf_counter()
                prof['act'] += t2 - t1
                prof['latency'] += t2 - t1
                if busy:
                    prof['act_busy'] += t2 - t1
                answer(prof, a_idx, sub, under_way, actions, logp, value)
            while inflight:
                finish(prof)
            collect_s = time.perf_counter() - t0
            if not sampler.live:
                break
            # ---- hand the rollout to the learner
            wait_s = 0.0
            if busy:   # --overlap: the update before is still running; the workers wait (and may search)
                tw = time.perf_counter()
                sampler.control[1] = 1
                wait_learner()
                sampler.control[1] = 0
                wait_s = time.perf_counter() - tw
            now = time.perf_counter()
            wst = sampler.stats.copy()
            dw = (wst - w_mark).sum(0)
            dec_w = (wst - w_mark)[:, ST['decisions']]   # decisions per worker in this cycle
            lab_job = {}
            if lab is not None:   # 2026-10-08: the lab's results joined per job, the builds seen, new jobs posted
                lab['book'].add(sampler.lab_records(), time.time())
                lab_job = dict(lab_results=lab['book'].take(), lab_finished=book.lab_finished)
                book.lab_finished = []
                for held in book.seen_inv:
                    lab['sched'].note_seen(held)
                book.seen_inv = []
                post_lab_jobs()
            job = dict(roll=roll, finished=book.finished, collect_s=collect_s, prof=prof, wait_learner_s=wait_s,
                       **lab_job,
                       teach=sampler.teacher_records() if args.teacher else None, t_start=t_run,
                       choices=sampler.choice_records() if branching else None,
                       branch_finished=book.branch_finished,
                       cycle_s=now - t_cycle, worker={c: float(dw[ST[c]]) for c in wcols if c in ST},
                       cpu_collect=cpu_share(cpu_c0, cpu_ticks()), cycle_wait_s=sync_wait,
                       dec_min=float(dec_w.min()), dec_max=float(dec_w.max()),
                       dec_w=' '.join(str(int(v)) for v in dec_w))
            t_cycle, w_mark = now, wst
            proc_now = time.process_time()
            job['proc_cores'] = (proc_now - proc_mark) / max(job['cycle_s'], 1e-9)
            proc_mark = proc_now
            handed += roll.usable()
            spare.carry(roll)
            roll, spare = spare, roll
            book.finished = []
            book.branch_finished = []
            busy = True
            jobs.put(job)
            if not args.overlap:   # synchronous: the workers wait for the new weights (queued searches run)
                tw = time.perf_counter()
                sampler.control[1] = 1
                wait_learner()
                sampler.control[2] = time.monotonic_ns()
                sampler.control[1] = 0
                sync_wait = time.perf_counter() - tw
            if failure:
                raise failure[0]
    finally:
        jobs.put(None)
        learner.join(timeout=900)
        torch.cuda.synchronize()
        save(out / 'checkpoints' / 'last.pt')
        sampler.close()
        if log_file is not None:
            log_file.close()
        episodes_path.close()
        for f_ in (branch_path, choice_path) + ((lab['out'],) if lab is not None else ()):
            if f_ is not None:
                f_.close()
        if branching and L.get('choice_n'):   # the choice records' ring as it is at the end (rows + meta)
            np.savez(out / 'choices-last.npz', rows=L['choice_rows'][:L['choice_n']],
                     meta=L['choice_meta'][:L['choice_n']], version=L['choice_version'][:L['choice_n']])
        print('finished', L['update'], 'updates', round(L['decisions'] * hours_per_decision, 1), 'game hours',
              round((time.perf_counter() - t_run) / 3600, 2), 'h')
        if failure:
            raise failure[0]


if __name__ == '__main__':
    main()
