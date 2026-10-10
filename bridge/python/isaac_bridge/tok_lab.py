"""Build benchmark lab (Phase C, 2026-10-08; the user's 10-06 design): the strength of a build measured directly.

A build (held collectibles; their stats, hearts and familiars follow from them) is a transferable object in Isaac, unlike a
Go position: it can be given to the player in any state. The lab gives a build to the player in each state of a fixed
panel of parked room states and lets the current policy play there, with common random numbers across builds; the
results train a build model (items -> benchmark scores) whose disagreement picks the next builds, and the lab episodes
are ordinary on-policy episodes in which the policy practises with items.

Transplant (transplant_lua): Lua `EntityPlayer:AddCollectible(id, charge, true)` per collectible, in the order given,
  with an active item's MaxCharges as its charge. This is the engine's own pickup path minus the pedestal: a touched
  pedestal queues the item, the player holds it up and Entity_Player::FlushQueueItem (RVA 0x2A7130) then calls
  AddCollectible(id, the pedestal's charge, not touched before) (decompile); AddCollectible itself calls EvaluateItems,
  adds the item's hearts / consumables, spawns its familiars and pickups (calls.csv). The exactness check is
  abplus_probe_transplant.py (EXPERIMENTS.md, Phase C).
Panel (cfg.lab_panel): rooms of the catalog groups (room-mode resets of tok_sampler's room workers: a goto room with its
  own enemies, doors barred, Isaac with 6 half hearts, 1 bomb, no items), entered from held-out seeds. State k is built
  and parked by worker k mod W (build_panel_state, in the root between its floor builds); a lost panel (root relaunch)
  is rebuilt from the same seeds.
Job (the trainer's LabScheduler, shared array cfg.lab jobs): a build and a replicate index rep. Each worker runs its
  panel states for every job in id order: a lean fork of the parked state, the transplant, the in-place reseed
  state_reseed(state seed, rep) (the game's global MT: common random numbers from the first record on whatever the
  build drew at the transplant), then a lab episode with the actor's actions: records tagged ROW 'lab' (job id) and
  'lab_s' (panel state + 1), the policy's sampling uniforms common too (ROW 'crn' = state_crn(state, rep): GraphActor's
  crn_uniforms), ends: death, the room cleared, the room left, cfg.lab_seconds (the last three a truncation: done
  DONE_BRANCH, the trainer bootstraps), the floor's stall rule (done 3). Two runs of one job (same build, same rep) with
  the same weights play the same game (the twin check); another rep is the seed noise.
Result record (worker ring LAB_RING of RES_F float64): per (job, state) the outcome (RES), written by the worker; the
  trainer joins them per job (LabBook) into a build result: per state and aggregated (clear, death, timeout, hurt,
  damage share, seconds to the end, the undiscounted return in the trainer's reward units: the scalar strength).
"""
import json
from pathlib import Path

import numpy as np

LAB_MAX_ITEMS = 8       # collectibles in one build at most
LAB_JOBS = 64           # job slots of the shared schedule (job id j in slot j mod LAB_JOBS)
LAB_RING = 512          # result records per worker the trainer may fall behind by
JOB = ('job', 'rep', 'n', 'kind', 'flags') + tuple(f'item{j}' for j in range(LAB_MAX_ITEMS))
JOB_F = len(JOB)
JI = {k: i for i, k in enumerate(JOB)}
# build kinds (JOB 'kind')
K_BASE, K_SINGLE, K_MODEL, K_RANDOM, K_SEEN, K_LIST, K_REPLICATE = range(7)
KIND_NAMES = {K_BASE: 'base', K_SINGLE: 'single', K_MODEL: 'model', K_RANDOM: 'random', K_SEEN: 'seen', K_LIST: 'list',
              K_REPLICATE: 'replicate'}
# per (job, state) result
STATS_F = ('damage', 'fire_delay_max', 'shot_speed', 'range', 'speed', 'luck', 'can_fly', 'hearts', 'max_hearts',
           'soul', 'black', 'bone', 'bombs', 'coins', 'keys', 'weapons', 'active', 'size')
RES = ('job', 'state', 'rep', 'worker', 'ok', 'end', 'clear', 'death', 'timeout', 'left', 'hurt', 'damage_share',
       'decisions', 'frames', 'ret', 'v_end', 'boot', 'version0', 'version1', 'reseed', 'crn', 'wall_s', 'cpu_s',
       'trace', 'first', 'n_items', 'familiars', 'pickups') + tuple('st_' + k for k in STATS_F)
RES_F = len(RES)
RI = {k: i for i, k in enumerate(RES)}
# how a lab episode ended (RES 'end')
E_DEATH, E_CLEAR, E_LEFT, E_CAP, E_STALL, E_ERROR, E_STOP = range(1, 8)
END_NAMES = {E_DEATH: 'death', E_CLEAR: 'clear', E_LEFT: 'left', E_CAP: 'cap', E_STALL: 'stall', E_ERROR: 'error',
             E_STOP: 'stop'}

TRANSPLANT_LUA = """
local p = Isaac.GetPlayer(0)
local cfg = Isaac.GetItemConfig()
local done = 0
for _, id in ipairs({{{ids}}}) do
  local charge = 0
  pcall(function()
    local c = cfg:GetCollectible(id)
    if c ~= nil and c.Type == ItemType.ITEM_ACTIVE then charge = c.MaxCharges end
  end)
  p:AddCollectible(id, charge, true)
  done = done + 1
end
return string.format('%d|%.6f|%.6f|%.6f|%.6f|%.6f|%d|%d', done, p.Damage, p.MaxFireDelay, p.ShotSpeed, p.TearHeight,
  p.MoveSpeed, p:GetCollectibleCount(), p:GetHearts())
"""


def transplant_lua(items):
    """The Lua chunk that gives the player of the current state these collectibles (see the module docstring)."""
    return TRANSPLANT_LUA.format(ids=', '.join(str(int(i)) for i in items))


def mix64(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def state_reseed(seed, rep):
    """The global MT reseed of a panel state (its seed) for replicate rep: 1 .. 2^31 - 2, the same for every build."""
    return int(mix64(int(seed) * 1009 + int(rep) * 7919 + 0x1AB) % (2 ** 31 - 2)) + 1


def state_crn(state, rep):
    """The policy-noise key of a panel state's lab episodes for replicate rep (never 0; ROW 'crn')."""
    return int(mix64(0x1AB0000 + int(state) * 4096 + int(rep)) & 0x7FFFFFFFFFFFFFFF) or 1


def parse_panel(text, seed0):
    """'normal:24,boss:6,normal_big:6' -> ([group names], [(group index, seed)]): state k of group g has the seed
    seed0 + 1000 x g + j (j its number within the group). The groups are interleaved (normal, boss, big, normal, ...) so
    that states k mod W spread the kinds over the workers."""
    names, counts = [], []
    for part in (text or '').split(','):
        if part.strip():
            name, _, c = part.strip().partition(':')
            names.append(name)
            counts.append(int(c or 1))
    per = [[(g, int(seed0) + 1000 * g + j) for j in range(c)] for g, c in enumerate(counts)]
    out = []
    while any(per):
        for lst in per:
            if lst:
                out.append(lst.pop(0))
    return names, out


GROUP_COST = {'boss': 3.2, 'normal_big': 1.5, 'normal': 1.0}   # a lab episode's mean game time relative to a 1x1
#                                                                  normal room's (run A, C67 final: 51 / 24 / 16 s)


def assign_panel(panel, names, workers, costs=None, state_costs=None):
    """The worker of each panel state (a job is complete when its slowest worker is done): longest expected episode
    first, each to the least loaded worker (LPT), by the state's measured cost (state_costs: a list per state, e.g.
    state_costs_from(an earlier run's lab.jsonl)) or else its group's (GROUP_COST; 1 for another group)."""
    costs = dict(GROUP_COST, **(costs or {}))

    def cost(k):
        if state_costs is not None and k < len(state_costs) and state_costs[k] > 0:
            return float(state_costs[k])
        return costs.get(names[panel[k][0]], 1.0)
    load = [0.0] * int(workers)
    owner = [0] * len(panel)
    for k in sorted(range(len(panel)), key=lambda k: (-cost(k), k)):
        w = min(range(len(load)), key=lambda i: (load[i], i))
        owner[k] = w
        load[w] += cost(k)
    return owner


def state_costs_from(path, n_states):
    """Mean game seconds per panel state over the complete builds of a run's lab.jsonl (0 where none)."""
    tot, cnt = [0.0] * n_states, [0] * n_states
    for line in open(path):
        r = json.loads(line)
        for s, v in (r.get('per_state') or {}).items():
            s = int(s)
            if s < n_states:
                tot[s] += float(v[4]) / 30.0
                cnt[s] += 1
    return [t / c if c else 0.0 for t, c in zip(tot, cnt)]


def load_items(path):
    """The eligible collectibles of the lab (catalog/lab_items.json): {id: record}."""
    raw = json.loads(Path(path).read_text(encoding='utf8'))
    return {int(r['id']): r for r in raw['items']}


def job_items(row):
    n = int(row[JI['n']])
    return [int(v) for v in row[JI['item0']:JI['item0'] + n]]


def build_panel_state(inst, spec, seed):
    """A panel state in the worker's root instance (between its floor builds, the builder idle): a room-mode reset of
    `seed` in the task group of `spec` (catalog rooms, the room workers' reset: Isaac with cfg.start_hp half hearts and
    cfg.bombs bombs, no items, doors barred), forked and returned as a parked clone. The root's reset mode and tasks are
    restored afterwards (its next floor build restarts the run)."""
    from .abplus_tasks import TaskSampler
    b = inst.env.bridge
    old = (b.reset_mode, b.tasks)
    s = spec['spec']
    try:
        b.tasks = TaskSampler(s['weights'], s['normal'], s['boss'], (spec.get('target') or {}).get('arms'))
        b.reset_mode = None
        _, info = inst.reset(seed)
        clone = b.fork(tag='panel', alarm=0)
    finally:
        b.reset_mode, b.tasks = old
    return clone, dict(task=info.get('task'), variant=info.get('room_variant'))


def stats_of(o):
    """The STATS_F values of player 0 of a lean observation, the familiars and the pickups of the room."""
    p = o.players[0]
    st = [float(p[k]) for k in STATS_F]
    e = o.entities
    fam = int(np.sum(e['type'] == 3)) if len(e) else 0
    pick = int(np.sum((e['type'] == 5) & (e['variant'] != 100))) if len(e) else 0
    return st, fam, pick


# ----------------------------------------------------------------------------------------------------- trainer side

class LabBook:
    """The trainer's join of the workers' result records per job (a build result once every panel state has
    reported; a job older than `timeout` s with states missing is closed as incomplete)."""

    def __init__(self, n_states, timeout=600.0):
        self.n_states, self.timeout = int(n_states), float(timeout)
        self.open = {}      # job id -> dict(job row, results {state: RES vector}, posted at)
        self.done = []      # completed build results since the last take()

    def post(self, row, now):
        self.open[int(row[JI['job']])] = dict(row=np.array(row, np.int64), res={}, t=now)

    def add(self, recs, now):
        for r in recs:
            j = int(r[RI['job']])
            e = self.open.get(j)
            if e is None:
                continue
            e['res'][int(r[RI['state']])] = np.array(r, np.float64)
        for j in sorted(self.open):
            e = self.open[j]
            if len(e['res']) >= self.n_states or now - e['t'] > self.timeout:
                self.done.append(self.finish(e))
                del self.open[j]

    def finish(self, e):
        row = e['row']
        res = e['res']
        states = sorted(res)
        ok = [s for s in states if res[s][RI['ok']] > 0]
        out = dict(job=int(row[JI['job']]), rep=int(row[JI['rep']]), kind=KIND_NAMES.get(int(row[JI['kind']]), '?'),
                   flags=int(row[JI['flags']]), items=job_items(row), states=len(states), ok=len(ok),
                   complete=len(ok) == self.n_states)
        if ok:
            m = np.stack([res[s] for s in ok])

            def mean(k):
                return float(m[:, RI[k]].mean())
            out.update({k: mean(k) for k in ('clear', 'death', 'timeout', 'left', 'hurt', 'damage_share', 'ret')})
            out['seconds'] = float((m[:, RI['frames']] / 30.0).mean())
            out['game_s'] = float((m[:, RI['frames']] / 30.0).sum())
            out['wall_s'] = float(m[:, RI['wall_s']].sum())
            out['cpu_s'] = float(m[:, RI['cpu_s']].sum())
            out['versions'] = [int(m[:, RI['version0']].min()), int(m[:, RI['version1']].max())]
            out['stats'] = {k: float(np.median(m[:, RI['st_' + k]])) for k in STATS_F}
            out['familiars'] = float(np.median(m[:, RI['familiars']]))
            out['per_state'] = {int(s): [round(float(res[s][RI[k]]), 4) for k in
                                         ('clear', 'death', 'hurt', 'damage_share', 'frames', 'ret', 'end', 'trace',
                                          'version0', 'version1', 'first')] for s in ok}
        return out

    def take(self):
        out, self.done = self.done, []
        return out


class LabScheduler:
    """The trainer's choice of builds (one job per build and replicate). Mix per new job (cfg shares, the rest single
    items): the empty build, the next single item (every eligible collectible once in a shuffled order, then again),
    pairs / triples of the model's candidates (the highest ensemble disagreement of a random pool, LabModel) or random
    ones, builds seen in training episodes, the builds of a list file first (--lab-builds), and replicates of earlier
    jobs (noise: the same rep = the twin check, another rep = seed noise)."""

    def __init__(self, items, seed, shares, builds_first=(), noise=0.0, max_items=3):
        self.items = dict(items)
        self.ids = sorted(self.items)
        self.actives = {i for i, r in self.items.items() if r['kind'] == 'active'}
        self.rng = np.random.default_rng([int(seed), 0x1AB])
        self.shares = dict(shares)
        self.queue = [list(b) for b in builds_first]
        self.order, self.next = [], 0
        self.noise = float(noise)
        self.max_items = int(max_items)
        self.job = 0
        self.history = []        # (job id, items, rep) of posted jobs
        self.seen = {}           # tuple(items) -> count (training episodes' inventories)
        self.candidates = []     # (score, items) of the model (LabModel.rank), best first

    def valid(self, build):
        b = [int(i) for i in build if int(i) in self.items]
        if sum(i in self.actives for i in b) > 1:   # one active slot: keep the first active only
            first = next(i for i in b if i in self.actives)
            b = [i for i in b if i not in self.actives or i == first]
        return b[:LAB_MAX_ITEMS]

    def random_build(self, k):
        while True:
            b = sorted(int(v) for v in self.rng.choice(self.ids, size=k, replace=False))
            if sum(i in self.actives for i in b) <= 1:
                return b

    def next_single(self):
        if self.next >= len(self.order):
            self.order = [int(v) for v in self.rng.permutation(self.ids)]
            self.next = 0
        self.next += 1
        return [self.order[self.next - 1]]

    def note_seen(self, inv_ids):
        b = tuple(sorted(int(i) for i in inv_ids if int(i) in self.items))
        if b:
            self.seen[b] = self.seen.get(b, 0) + 1

    def draw(self):
        """(items, kind, rep, flags) of the next job."""
        if self.queue:
            return self.valid(self.queue.pop(0)), K_LIST, 0, 0
        u = self.rng.random()
        if self.noise > 0 and self.history and u < self.noise:   # a replicate of a recent job
            j, items, rep = self.history[int(self.rng.integers(max(0, len(self.history) - 64), len(self.history)))]
            twin = self.rng.random() < 0.5
            return list(items), K_REPLICATE, rep if twin else rep + 1 + int(self.rng.integers(1000)), j
        u = self.rng.random()
        acc = 0.0
        for kind, key in ((K_BASE, 'base'), (K_MODEL, 'model'), (K_RANDOM, 'random'), (K_SEEN, 'seen')):
            acc += self.shares.get(key, 0.0)
            if u < acc:
                if kind == K_BASE:
                    return [], K_BASE, 0, 0
                if kind == K_MODEL and self.candidates:
                    k = min(len(self.candidates), 8)
                    _, items = self.candidates.pop(int(self.rng.integers(k)))
                    return self.valid(items), K_MODEL, 0, 0
                if kind == K_SEEN and self.seen:
                    keys = list(self.seen)
                    return self.valid(list(keys[int(self.rng.integers(len(keys)))])), K_SEEN, 0, 0
                if kind in (K_MODEL, K_RANDOM):
                    return self.random_build(int(self.rng.integers(2, self.max_items + 1))), K_RANDOM, 0, 0
        return self.next_single(), K_SINGLE, 0, 0

    def make_job(self):
        items, kind, rep, flags = self.draw()
        self.job += 1
        row = np.zeros(JOB_F, np.int64)
        row[JI['job']], row[JI['rep']], row[JI['n']], row[JI['kind']], row[JI['flags']] = \
            self.job, rep, len(items), kind, flags
        row[JI['item0']:JI['item0'] + len(items)] = items
        if kind != K_REPLICATE:
            self.history.append((self.job, tuple(items), rep))
        return row


class LabModel:
    """The build model: K small MLPs over a build's collectibles (an embedding bag, a multi-hot) and its stats (measured
    after the transplant; for a candidate not yet measured, the base stats plus the sum of its items' measured single-item
    deltas) -> the benchmark targets TARGETS (standardised); each head trained on its own Poisson(1) bootstrap weights of
    the build results. Disagreement (the heads' standard deviation of the scalar 'ret', in its units) ranks
    candidates."""
    TARGETS = ('ret', 'clear', 'death', 'hurt', 'damage_share', 'seconds')

    def __init__(self, k=5, n_items=1024, emb=32, hidden=64, lr=1e-3, device='cpu', seed=0):
        import torch
        from torch import nn
        self.torch, self.device, self.k = torch, torch.device(device), int(k)
        self.nets = nn.ModuleList()
        with torch.random.fork_rng(devices=[]):   # its own initialisation; the trainer's generator is untouched
            torch.manual_seed(int(seed))
            for _ in range(self.k):
                net = nn.ModuleDict(dict(bag=nn.EmbeddingBag(n_items, emb, mode='sum', padding_idx=0),
                                         mlp=nn.Sequential(nn.Linear(emb + len(STATS_F), hidden), nn.GELU(),
                                                           nn.Linear(hidden, hidden), nn.GELU(),
                                                           nn.Linear(hidden, len(self.TARGETS)))))
                with torch.no_grad():
                    net['bag'].weight.mul_(0.05)
                self.nets.append(net)
        self.nets.to(self.device)
        self.opt = torch.optim.Adam(self.nets.parameters(), lr=lr)
        self.rows = []        # (items, stats, targets, bootstrap weights)
        self.rng = np.random.default_rng([int(seed), 0x30D])
        self.base_stats = None
        self.single_delta = {}   # item -> stats delta of its single-item build (medians of the panel)
        self.mu = np.zeros(len(self.TARGETS))
        self.sd = np.ones(len(self.TARGETS))
        self.smu = np.zeros(len(STATS_F))
        self.ssd = np.ones(len(STATS_F))

    def add(self, result):
        """A complete build result of LabBook."""
        if not result.get('complete') or 'stats' not in result:
            return
        st = np.array([result['stats'][k] for k in STATS_F], np.float64)
        items = list(result['items'])
        if not items and self.base_stats is None:
            self.base_stats = st
        if len(items) == 1 and self.base_stats is not None:
            self.single_delta[items[0]] = st - self.base_stats
        y = np.array([result[k] for k in self.TARGETS], np.float64)
        self.rows.append((items, st, y, self.rng.poisson(1.0, self.k).astype(np.float32)))

    def est_stats(self, items):
        base = self.base_stats if self.base_stats is not None else np.zeros(len(STATS_F))
        return base + sum((self.single_delta.get(i, 0.0) for i in items), np.zeros(len(STATS_F)))

    def _batch(self, builds, stats):
        torch = self.torch
        flat, offs = [], []
        for b in builds:
            offs.append(len(flat))
            flat.extend(int(i) for i in b)
        if not flat:   # EmbeddingBag needs one index: the padding id 0 (zero vector)
            flat = [0]
        x_items = torch.tensor(flat, dtype=torch.long, device=self.device)
        x_offs = torch.tensor(offs, dtype=torch.long, device=self.device)
        # (clipped: a stat that was constant in the held builds, e.g. luck, has a floor of 0.25 as its scale)
        s = np.clip((np.asarray(stats, np.float64) - self.smu) / self.ssd, -8.0, 8.0)
        return x_items, x_offs, torch.tensor(s, dtype=torch.float32, device=self.device)

    def _forward(self, net, xi, xo, xs):
        return net['mlp'](self.torch.cat([net['bag'](xi, xo), xs], -1))

    def train(self, steps=50, batch=128):
        """steps of Adam on the held build results (each head on its bootstrap weights). Returns the mean loss."""
        torch = self.torch
        n = len(self.rows)
        if n < 8:
            return None
        Y = np.stack([r[2] for r in self.rows])
        S = np.stack([r[1] for r in self.rows])
        self.mu, self.sd = Y.mean(0), Y.std(0) + 1e-6
        self.smu, self.ssd = S.mean(0), np.maximum(S.std(0), 0.25)
        W = np.stack([r[3] for r in self.rows])
        losses = []
        for _ in range(steps):
            pick = self.rng.integers(0, n, min(batch, n))
            xi, xo, xs = self._batch([self.rows[j][0] for j in pick], S[pick])
            y = torch.tensor((Y[pick] - self.mu) / self.sd, dtype=torch.float32, device=self.device)
            w = torch.tensor(W[pick], dtype=torch.float32, device=self.device)
            loss = 0.0
            for h, net in enumerate(self.nets):
                err = ((self._forward(net, xi, xo, xs) - y) ** 2).mean(-1)
                loss = loss + (err * w[:, h]).sum() / w[:, h].sum().clamp(min=1.0)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            losses.append(float(loss.detach()) / self.k)
        return float(np.mean(losses))

    def predict(self, builds, stats=None):
        """[K, n, targets] in the targets' units (stats None: est_stats of each build)."""
        torch = self.torch
        if stats is None:
            stats = [self.est_stats(b) for b in builds]
        with torch.no_grad():
            xi, xo, xs = self._batch(builds, stats)
            out = torch.stack([self._forward(net, xi, xo, xs) for net in self.nets]).cpu().numpy()
        return out * self.sd + self.mu

    def rank(self, pool):
        """[(disagreement of 'ret', build)] of the candidate builds, highest first."""
        if not pool or len(self.rows) < 8:
            return []
        p = self.predict(pool)[:, :, 0]
        sd = p.std(0)
        order = np.argsort(-sd)
        return [(float(sd[j]), list(pool[j])) for j in order]

    def prequential(self, result):
        """Before a new build result is added: the heads' mean prediction of its 'ret' and their spread (an
        out-of-sample check: the model has not seen this build's result). None before training."""
        if len(self.rows) < 8 or not result.get('complete') or 'stats' not in result:
            return None
        st = np.array([result['stats'][k] for k in STATS_F], np.float64)
        p = self.predict([list(result['items'])], [st])[:, 0, 0]
        return float(p.mean()), float(p.std())
