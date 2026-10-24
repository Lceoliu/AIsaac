"""Fork sampler (EXPERIMENTS.md B10): every episode is a clone of a parked start state.

The training loop so far (abplus_worker / abplus_vec) rebuilt a room for every episode with console commands (about
0.36 s of one core, more than a short episode itself costs), kept a second game instance per environment for that, ran
its reward, geometry and label code in Python at every step (7-18 ms a decision against 1.3 ms for the game and the
bridge) and stepped all environments in lockstep with the learner's inference. Here:

  worker process x N (this module's worker_main; no torch):
    root      one AB+ instance (FORK_ENV). It only ever resets: after a reset it is at an episode's start state.
    template  a clone of the root at that start state, parked. While it serves episodes the root already builds the
              next start state in a background thread.
    episode   a clone of the template (bridge.fork, ~5 ms), with the global MT reseeded, so the episodes of one
              template differ in the game's random draws. It is stepped until the episode ends and then closed.
              A template serves `episodes_per_state` episodes, more if the next state is not ready yet.
    Each decision the worker decodes the binary observation, writes one fixed-size record (FRAME: raw player, entity
    and terrain arrays, the step's damage counters and the episode's end) into its row of a shared array, and either
    draws a random action itself (policy 'random') or tells the server the row is ready and waits for its action.
  server (ForkSampler.serve, the process that owns the GPU): gathers the rows that are ready, runs one batched forward
    pass, writes the actions and wakes those workers. Workers that are still stepping are not waited for, so with more
    workers than cores the cores stay busy while a batch is in the network.

The record is a stand-in for the policy input that is still to be designed (step 3): nothing here computes rewards
shaping terms, geometry labels or masks. All of this is measured in abplus_bench_fork_sampler.py.
"""
import multiprocessing as mp
import os
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing import connection, shared_memory

import numpy as np

from .abplus import FORK_ENV
from .abplus_goexplore import GxConfig, Instance
from .abplus_lean import LeanDecoder, read_lean
from .env import BridgeError
from .transformer_obs import terrain_channels

PLAYER_F = 24
ENTITY_CAP = 96
ENTITY_F = 20
GRID = (7, 16, 28)
FRAME = np.dtype([
    ('player', np.float32, (PLAYER_F,)),
    ('entities', np.float32, (ENTITY_CAP, ENTITY_F)),
    ('grid', np.uint8, GRID),
    ('n_entities', np.int32),
    ('t', np.int32),             # decisions since the episode's start
    ('episode', np.int64),       # the worker's episode counter
    ('hurt', np.float32),        # half hearts lost during the step that led here
    ('damage', np.float32),      # share of the room's enemy HP removed during that step
    ('done', np.uint8),          # 0 running, 1 win (room clear), 2 death, 3 time limit
    ('first', np.uint8),         # the episode's first record
], align=True)
STATS = ('decisions', 'frames', 'episodes', 'forks', 'states', 'errors', 'step_s', 'wait_s', 'fork_s', 'encode_s',
         'wins', 'deaths', 'timeouts', 'state_wait_s', 'extra_episodes', 'ready', 'swap_s', 'close_s', 'loop_s')
ACTION_F = 4    # move (9), shoot (5), bomb, item


@dataclass
class SamplerConfig:
    workers: int = 16
    policy: str = 'random'            # 'random': the worker draws; 'server': ForkSampler.serve answers
    frames_per_decision: int = 4
    episodes_per_state: int = 4
    seed: int = 1000
    name: str = 'fs'
    port: int = 28100
    nice: int = 10                    # workers and games; the server stays at 0 and gets the CPU when it wants it
    min_batch: int = 0                # serve(): wait for this many ready rows (0: a third of the workers) ...
    max_wait_ms: float = 1.0          # ... but no longer than this after the first one
    bridge_lua: str = ''
    preload: str = ''
    clone_alarm: int = 600            # real seconds an episode clone may live
    lean: bool = True                 # episode clones in the bridge's lean mode (abplus_lean.py)
    lag: int = 0                      # 1: a decision's action is applied one decision later (see worker_main)
    stub_list: str = ''               # ABP_STUB_LIST for the instances ('' = the exact mode's own list)
    start_hp: int = 6
    bombs: int = 1


def encode_frame(obs, previous, row, terrain_cache):
    """One observation as a FRAME record. previous: the combat block before the step (None at the episode's start)."""
    p = obs['players'][0]
    px, py = p['pos']
    vx, vy = p.get('vel') or (0.0, 0.0)
    row['player'] = (px, py, vx, vy, p['size'], p['hearts'], p['max_hearts'], p['soul'], p['black'], p['bone'],
                     p['eternal'], p['bombs'], p['keys'], p['coins'], p['damage'], p['fire_delay_max'], p['shot_speed'],
                     p['range'], p['speed'], p['luck'], p['can_fly'], p['invulnerable'], p['active'], p['active_ready'])
    ents = obs['entities']
    n = min(len(ents), ENTITY_CAP)
    if n:
        rows = []
        for e in ents[:n]:
            ex, ey = e['pos']
            evx, evy = e.get('vel') or (0.0, 0.0)
            rows.append((e['type'], e['variant'], e['subtype'], ex - px, ey - py, evx, evy, e['size'], e['coll'],
                         e['gcoll'], e['cdmg'], e['aframe'], e['age'], 1.0 if e.get('projectile') else 0.0,
                         1.0 if e.get('enemy') else 0.0, 1.0 if e.get('boss') else 0.0, e.get('height', 0.0),
                         e.get('fall', 0.0), e.get('boss_hp', 0.0), 1.0 if e.get('blocking') else 0.0))
        row['entities'][0, :n] = rows
    row['n_entities'] = n
    version = (obs['terrain'].get('version'), tuple((d['slot'], d['open']) for d in obs['doors']))
    if terrain_cache.get('version') != version:
        terrain_cache['version'] = version
        terrain_cache['grid'] = terrain_channels(obs, GRID[1:]).astype(np.uint8)
    row['grid'] = terrain_cache['grid']
    combat = obs['combat']
    if previous is None:
        row['hurt'] = row['damage'] = 0.0
    else:
        row['hurt'] = combat['player_damage'] - previous['player_damage']
        row['damage'] = combat['enemy_damage_fraction'] - previous['enemy_damage_fraction']
    return combat


PLAYER_COLUMNS = ('x', 'y', 'vx', 'vy', 'size', 'hearts', 'max_hearts', 'soul', 'black', 'bone', 'eternal', 'bombs',
                  'keys', 'coins', 'damage', 'fire_delay_max', 'shot_speed', 'range', 'speed', 'luck', 'can_fly',
                  'invulnerable', 'active', 'active_ready')


def encode_lean(o, previous, row, terrain_cache):
    """A lean observation (abplus_lean.LeanObs) as a FRAME record, the same columns as encode_frame. previous: (damage
    taken, monsters' HP) before the step, None at the episode's start; returns this observation's pair."""
    p = o.players[0]
    px, py = p['x'], p['y']
    row['player'][0] = [p[name] for name in PLAYER_COLUMNS]
    e = o.entities
    n = min(len(e), ENTITY_CAP)
    if n:
        e = e[:n]
        out = row['entities'][0]
        kind, flags = e['kind'], e['flags']
        out[:n, 0], out[:n, 1], out[:n, 2] = e['type'], e['variant'], e['subtype']
        out[:n, 3], out[:n, 4], out[:n, 5], out[:n, 6] = e['x'] - px, e['y'] - py, e['vx'], e['vy']
        out[:n, 7], out[:n, 8], out[:n, 9], out[:n, 10] = e['size'], e['coll'], e['gcoll'], e['cdmg']
        out[:n, 11], out[:n, 12] = e['aframe'], e['age']
        out[:n, 13], out[:n, 14], out[:n, 15] = kind == 2, flags & 1, (flags >> 2) & 1
        out[:n, 16], out[:n, 17], out[:n, 18], out[:n, 19] = e['height'], e['fall'], e['hp'], (flags >> 3) & 1
    row['n_entities'] = n
    version = (o.terrain_version, o.doors['open'].tobytes())
    if terrain_cache.get('version') != version:
        terrain_cache['version'] = version
        doors = [dict(open=bool(d['open']), pos=(float(d['x']), float(d['y']))) for d in o.doors]
        terrain_cache['grid'] = terrain_channels(dict(terrain=o.terrain, doors=doors), GRID[1:]).astype(np.uint8)
    row['grid'] = terrain_cache['grid']
    if previous is None:
        row['hurt'] = row['damage'] = 0.0
    else:
        row['hurt'] = o.damage_taken - previous[0]
        row['damage'] = max(0.0, previous[1] - o.monsters_hp)
    return o.damage_taken, o.monsters_hp


def lean_step(bridge, decoder, action, repeat):
    bridge._send({"cmd": "step", "repeat": repeat, "move": int(action[0]), "shoot": int(action[1]),
                  "bomb": int(action[2]), "item": int(action[3])})
    return read_lean(bridge, decoder)


def raw_step(bridge, action, repeat):
    bridge._send({"cmd": "step", "repeat": repeat, "move": int(action[0]), "shoot": int(action[1]),
                  "bomb": int(action[2]), "item": int(action[3])})
    line = bridge._read_line()
    if not line.startswith(b"B "):
        raise BridgeError(f"unexpected reply {line[:120]!r}")
    return bridge.decoder.decode(bridge._read_exact(int(line.split(b" ")[1])))


def attach(names, workers):
    """The shared arrays (frames, actions, stats, control) of a sampler from their shared-memory names. Two frame rows
    per worker: with lag 1 a worker writes its next record while the server may still read the one before."""
    blocks = [shared_memory.SharedMemory(name=n) for n in names]
    frames = np.ndarray((2 * workers,), FRAME, buffer=blocks[0].buf)
    actions = np.ndarray((workers, ACTION_F), np.int32, buffer=blocks[1].buf)
    stats = np.ndarray((workers, len(STATS)), np.float64, buffer=blocks[2].buf)
    control = np.ndarray((4,), np.int64, buffer=blocks[3].buf)   # [0] stop
    return blocks, frames, actions, stats, control


def worker_main(index, cfg, spec, names, conn):
    blocks, frames, actions, stats, control = attach(names, cfg.workers)
    if cfg.nice > 0:
        os.nice(cfg.nice)
    st = {k: i for i, k in enumerate(STATS)}
    mine = stats[index]
    rows = (frames[2 * index:2 * index + 1], frames[2 * index + 1:2 * index + 2])
    row = rows[0]
    os.environ.update(FORK_ENV)
    gx = GxConfig(bridge_lua=cfg.bridge_lua, preload=cfg.preload, al_stopped=True, nice=cfg.nice,
                  frames_per_decision=cfg.frames_per_decision, start_hp=cfg.start_hp, bombs=cfg.bombs,
                  stub_list=cfg.stub_list)
    inst = None
    template = episode = None
    rng = np.random.default_rng([cfg.seed, index])
    limit = int(round(float(spec['seconds']) * 30 / cfg.frames_per_decision))
    state_n = 0

    def next_seed():
        nonlocal state_n
        seed = cfg.seed + index + cfg.workers * state_n
        state_n += 1
        return seed

    ready = threading.Event()
    failure = []

    def build(seed):
        try:
            inst.reset(seed)
        except Exception as exc:   # the main loop relaunches the instance
            failure.append(exc)
        ready.set()

    def start_build():
        ready.clear()
        threading.Thread(target=build, args=(next_seed(),), daemon=True).start()

    try:
        inst = Instance(f'{cfg.name}{index}', cfg.port + index, gx, spec)
        start_build()
        episodes_left = 0
        episode_n = 0
        parent = mp.parent_process()
        t_loop = time.perf_counter()
        while not control[0] and (parent is None or parent.is_alive()):
            mine[st['loop_s']] += time.perf_counter() - t_loop
            t_loop = time.perf_counter()
            if template is None or (episodes_left <= 0 and ready.is_set()):
                t = time.perf_counter()
                ready.wait()
                mine[st['state_wait_s']] += time.perf_counter() - t
                if failure:
                    failure.clear()
                    mine[st['errors']] += 1
                    inst.relaunch()
                    start_build()
                    continue
                t = time.perf_counter()
                fresh = inst.env.bridge.fork(tag='template', alarm=0)
                if template is not None:
                    template.close()
                template = fresh
                mine[st['states']] += 1
                episodes_left = cfg.episodes_per_state
                start_build()
                mine[st['swap_s']] += time.perf_counter() - t
            elif episodes_left <= 0:
                mine[st['extra_episodes']] += 1
            t = time.perf_counter()
            try:
                episode = template.fork(alarm=cfg.clone_alarm, reseed=int(rng.integers(1, 2 ** 31 - 1)),
                                        lean=True if cfg.lean else None)
                if cfg.lean:   # the clone's first observation, in the lean layout (it carries the terrain)
                    decoder = LeanDecoder()
                    episode._send({"cmd": "obs"})
                    obs = read_lean(episode, decoder)
                else:
                    obs = episode.last_obs
            except (BridgeError, OSError):
                mine[st['errors']] += 1
                template = None
                continue
            mine[st['fork_s']] += time.perf_counter() - t
            mine[st['forks']] += 1
            episodes_left -= 1
            episode_n += 1
            previous, cache, t_ep = None, {}, 0
            # lag 1 (policy 'server'): the action answered for decision t is applied at decision t + 1, so the game
            # steps while the server works on the record it just produced, and waits only if the answer to the
            # record before is still missing. The first decision of an episode is a no-op. A reaction time of one
            # decision (133 ms at 4 frames) for the policy; without lag the game idles through every inference.
            lagged = cfg.lag == 1 and cfg.policy == 'server'
            waiting, action = False, (0, 0, 0, 0)
            try:
                while True:
                    t = time.perf_counter()
                    if lagged:
                        row = rows[t_ep & 1]
                    if cfg.lean:
                        previous = encode_lean(obs, previous, row, cache)
                        dead, clear = obs.dead, obs.clear
                    else:
                        previous = encode_frame(obs, previous, row, cache)
                        dead, clear = obs['players'][0]['dead'], obs['room']['clear']
                    done = 2 if dead else 1 if clear else 3 if t_ep >= limit else 0
                    row['t'], row['episode'], row['done'], row['first'] = t_ep, episode_n, done, t_ep == 0
                    t1 = time.perf_counter()
                    mine[st['encode_s']] += t1 - t
                    if lagged:
                        if waiting:   # the answer to the record before this one
                            conn.recv_bytes()
                            action = tuple(int(v) for v in actions[index])
                        mine[st['ready']] += 1
                        conn.send_bytes(b'1' if t_ep & 1 else b'0')
                        waiting = True
                        if done or control[0]:
                            conn.recv_bytes()   # nothing may be outstanding when the next episode starts
                            waiting = False
                        mine[st['wait_s']] += time.perf_counter() - t1
                    elif cfg.policy == 'server':
                        mine[st['ready']] += 1
                        conn.send_bytes(b'0')
                        conn.recv_bytes()
                        action = actions[index]
                        mine[st['wait_s']] += time.perf_counter() - t1
                    else:
                        action = (int(rng.integers(9)), int(rng.integers(5)), 0, 0)
                    if done or control[0]:
                        break
                    t = time.perf_counter()
                    if cfg.lean:
                        obs = lean_step(episode, decoder, action, cfg.frames_per_decision)
                    else:
                        obs = raw_step(episode, action, cfg.frames_per_decision)
                    mine[st['step_s']] += time.perf_counter() - t
                    t_ep += 1
                    mine[st['decisions']] += 1
                    mine[st['frames']] += cfg.frames_per_decision
                mine[st['episodes']] += 1
                mine[st[{1: 'wins', 2: 'deaths', 3: 'timeouts'}.get(done, 'timeouts')]] += done != 0
            except (BridgeError, OSError, RuntimeError, ValueError):
                mine[st['errors']] += 1
            finally:
                t = time.perf_counter()
                try:
                    episode.close()
                except OSError:
                    pass
                episode = None
                mine[st['close_s']] += time.perf_counter() - t
    except Exception:
        traceback.print_exc()
        mine[st['errors']] += 1000
    finally:
        for c in (episode, template):
            try:
                if c is not None:
                    c.close()
            except OSError:
                pass
        if inst is not None:
            inst.close()
        if cfg.policy == 'server':
            try:
                conn.send_bytes(b'x')
            except OSError:
                pass
        for b in blocks:
            b.close()


class ForkSampler:
    """The workers and their shared arrays. serve(policy) answers ready rows in batches (policy 'server')."""

    def __init__(self, cfg, spec):
        self.cfg, self.spec = cfg, spec
        n = cfg.workers
        sizes = (FRAME.itemsize * 2 * n, 4 * ACTION_F * n, 8 * len(STATS) * n, 8 * 4)
        self.blocks = [shared_memory.SharedMemory(create=True, size=s) for s in sizes]
        names = [b.name for b in self.blocks]
        # the arrays live in the attached blocks' mappings: the blocks must stay referenced as long as the arrays
        self.attached, self.frames, self.actions, self.stats, self.control = attach(names, n)
        self.stats[:] = 0
        self.control[:] = 0
        ctx = mp.get_context('spawn')
        self.conns, self.procs = [], []
        for i in range(n):
            ours, theirs = ctx.Pipe()
            proc = ctx.Process(target=worker_main, args=(i, cfg, spec, names, theirs), daemon=True)
            proc.start()
            theirs.close()
            self.conns.append(ours)
            self.procs.append(proc)
        self.index = {c: i for i, c in enumerate(self.conns)}

    def totals(self):
        return dict(zip(STATS, self.stats.sum(axis=0).tolist()))

    def serve(self, policy, seconds, on_batch=None, conns=None, stop=None):
        """Answer ready rows until `seconds` have passed. policy(records, workers) -> int array (len(workers),
        ACTION_F). Returns (batches, rows answered). conns: only these workers' connections (several threads can each
        serve a share of the workers); stop: a threading.Event that ends the loop."""
        end = time.perf_counter() + seconds
        live = list(self.conns if conns is None else conns)
        batches = rows = 0
        want = self.cfg.min_batch or max(1, len(live) // 3)
        max_wait = self.cfg.max_wait_ms / 1000
        while live and time.perf_counter() < end and not (stop is not None and stop.is_set()):
            idx, slots, waiting, first = [], [], set(live), None
            # the first ready row starts the clock: more rows are taken until `want` are there or max_wait has passed
            while len(idx) < want:
                left = 0.2 if first is None else max_wait - (time.perf_counter() - first)
                if left <= 0:
                    break
                ready = connection.wait(list(waiting), timeout=left)
                if not ready:
                    break
                for c in ready:
                    waiting.discard(c)
                    try:
                        msg = c.recv_bytes()
                        if msg in (b'0', b'1'):   # which of the worker's two rows holds the record
                            idx.append(self.index[c])
                            slots.append(2 * self.index[c] + (msg == b'1'))
                        else:
                            live.remove(c)
                    except (EOFError, OSError):
                        live.remove(c)
                if first is None and idx:
                    first = time.perf_counter()
            if not idx:
                continue
            batch = self.frames[slots]
            self.actions[idx] = policy(batch, idx)
            if on_batch is not None:
                on_batch(batch, idx)
            for i in idx:
                self.conns[i].send_bytes(b'a')
            batches += 1
            rows += len(idx)
        return batches, rows

    def close(self, timeout=60.0):
        self.control[0] = 1
        end = time.time() + timeout
        if self.cfg.policy == 'server':   # workers waiting for an action must be released to see the stop flag
            while time.time() < end and any(p.is_alive() for p in self.procs):
                for c in connection.wait(self.conns, timeout=0.2):
                    try:
                        if c.recv_bytes() in (b'0', b'1'):
                            c.send_bytes(b'a')
                    except (EOFError, OSError):
                        self.conns.remove(c)
        for p in self.procs:
            p.join(max(0.1, end - time.time()))
            if p.is_alive():
                p.terminate()
        self.frames = self.actions = self.stats = self.control = None
        for b in self.attached:
            b.close()
        for b in self.blocks:
            b.close()
            b.unlink()
