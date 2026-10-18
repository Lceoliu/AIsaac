"""CPU smoke test of B40 (2026-10-18): the counterfactual teacher's wall trigger and its bomb_at branch
(isaac_bridge/tok_cf.py 'wall', BombIntent; train_tok --cf 1 --cf-wall 1).

No game, no GPU needed. Plain script:
  cd rl/bridge/python && python tests/test_cf_wall_smoke.py
Checks:
  (a) detection on synthetic ROW maps: a 1x1 room with doors on two sides and unknown cells on the other two (the door
      sides excluded, the candidate with more shown neighbours first, its bomb spot, p_vis); a shown neighbour without
      a door is no candidate; not before the clear, not without a bomb, not with wall off; boss rooms and cells next to
      a boss room excluded; a cleared room without candidates is marked seen; TRIGGERS order;
  (b) once per room: detect does not consume the trigger (a record whose point went elsewhere tries again), the
      caller's consume_wall does (seen for the room, the room key for the episode: a revisit does not fire again,
      another room does);
  (c) geometry: a 2x2 room's walls and bomb spots, a door on one of them; an L room's inner-corner wall; a narrow room's
      long sides have no slots;
  (d) p_vis: win_values against rl/macro's win_probabilities (when importable) on random offset groups;
  (e) BombIntent on a simulated walk with the one-decision action lag: moves toward the spot, then exactly one action
      with move 0 and bomb 1 (the policy's other heads), then the policy's whole action; stuck near the wall: the bomb
      after BOMB_STUCK decisions; stuck far: no bomb; no bomb held: no bomb; the room changed: no bomb; opened only after
      the bomb (the neighbour cell shown / a door on the bombed wall); reset;
  (f) cf_search with stand-in branches: the bomb_at intent first, kept for cf_wall_distill records, its summary taken
      before the replicate, 'distill' = through the bomb decision (capped), records_of with it; intents() for a wall
      target;
  (g) the deferral (--cf-defer-target, A30's A2): a door target / bomb_at intent plays the policy's whole action until
      the room is clear, then walks (its cap counted from the clear; the bomb_at's single bomb after the walk); the room
      changing first: never starts; intents() defers door and wall targets only; cf_search's 'defer' and the distill
      count from the intent's start.
"""
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_memory_smoke as M   # noqa: E402,F401  (its path set-up: the package, the gymnasium stand-in)

import numpy as np   # noqa: E402

from isaac_bridge import tok_cf as CF   # noqa: E402
from isaac_bridge import tok_teacher2 as T2   # noqa: E402
from isaac_bridge.tok_branch import parse_reward   # noqa: E402
from isaac_bridge.tok_obs import EV_CLEARED, EV_NEW_ROOM, ROW   # noqa: E402
from isaac_bridge.tok_pickup import SECTOR_MOVE, move_toward   # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


def room(cells, inside, pos, shown=(), bombs=1, clear=1.0, rtype=1, stage=1):
    """A record of a room covering minimap cells `cells` (the player's room), the inner area `inside` = (r0, r1, c0,
    c1) tile rows / columns of grid channel 0 (inclusive), the player at inner-area tiles pos, other shown cells."""
    r = np.zeros(1, ROW)
    e = r[0]
    for y, x in cells:
        e['map'][0, y, x] = e['map'][1, y, x] = e['map'][2, y, x] = e['map'][3, y, x] = 1
    for y, x in shown:
        e['map'][0, y, x] = 1
    r0, r1, c0, c1 = inside
    e['grid'][0, r0:r1 + 1, c0:c1 + 1] = 1
    W, H = c1 - c0 + 1, r1 - r0 + 1
    e['player'][0], e['player'][1] = pos[0] / W, pos[1] / H
    e['player'][23] = clear
    r['bombs'] = bombs
    r['room_type'] = rtype
    r['stage'] = stage
    return r


def put_door_at(r, j, X, Y):
    """A door at inner-area tiles (X, Y) (doors columns 0, 1: relative to the player / 200 px)."""
    e = r[0]
    W, H = CF.inner_area(e)
    px, py = CF.player_tiles(e, (W, H))
    e['doors'][j, 0], e['doors'][j, 1] = (X - px) * 40 / 200, (Y - py) * 40 / 200
    e['doors'][j, 2] = 1.0
    r['n_doors'] = max(int(r['n_doors'][0]), j + 1)


ONE = (1, 7, 1, 13)   # a 1x1 room's inner area: grid rows 1..7, columns 1..13 (13 x 7 tiles)


def base_1x1():
    """1x1 room at (6, 6); left (6, 5) and down (7, 6) shown with doors; (5, 5) shown (diagonal); up / right unknown."""
    r = room([(6, 6)], ONE, (6.5, 3.5), shown=[(6, 5), (7, 6), (5, 5)])
    put_door_at(r, 0, -0.5, 3.5)    # left door
    put_door_at(r, 1, 6.5, 7.5)     # down door
    return r


def test_detect():
    r = base_1x1()
    c = CF.wall_candidates(r[0])
    check('(a) candidates: up and right (the door sides left / down excluded)',
          [CF.SIDES[w['side']] for w in c] == ['up', 'right'], str([(CF.SIDES[w['side']], w['p']) for w in c]))
    check('(a) up first: 2 shown neighbours (-3) beat 1 (-6)', c[0]['p'] > c[1]['p'] > 0, f"{c[0]['p']:.3f} {c[1]['p']:.3f}")
    seen = set()
    f = CF.detect(r, seen, None, wall=True, walled=set())
    check('(a) detect: the wall trigger with its target', [CF.TRIGGERS[t] for t, _ in f] == ['wall'] and
          f[0][1][0] == 'wall', str(f))
    info = f[0][1][1]
    check('(a) bomb spot: the up wall\'s middle one tile in, the neighbour cell, the side',
          abs(info['x'] - 6.5) < 1e-9 and abs(info['y'] - 0.5) < 1e-9 and info['cell'] == 5 * 13 + 6 and
          info['side'] == 0 and info['n'] == 2 and info['room'] == (1, 6 * 13 + 6), str(info))
    check('(a) detect does not mark the wall seen', 'wall' not in seen)
    check('(a) wall off: never', CF.detect(r, set(), None) == [] and CF.detect(r, set(), None, wall=False) == [])
    r2 = base_1x1()
    r2['map'][0, 0, 5, 6] = 1   # the up neighbour shown (no door): not a candidate
    check('(a) a shown neighbour without a door is no candidate',
          [CF.SIDES[w['side']] for w in CF.wall_candidates(r2[0])] == ['right'])
    r3 = base_1x1()
    r3['player'][0, 23] = 0.0
    check('(a) not before the room is clear', CF.detect(r3, set(), None, wall=True, walled=set()) == [])
    r4 = base_1x1()
    r4['bombs'] = 0
    check('(a) not without a bomb', CF.detect(r4, set(), None, wall=True, walled=set()) == [])
    r5 = base_1x1()
    r5['map'][0, 4, 6, 6] = 1   # the room itself a boss room
    check('(a) a boss room: none', CF.wall_candidates(r5[0]) == [])
    r5b = base_1x1()
    r5b['room_type'] = 7
    check('(a) a secret room: none', CF.wall_candidates(r5b[0]) == [])
    r6 = base_1x1()
    r6['map'][0, 0, 4, 6] = r6['map'][0, 4, 4, 6] = 1   # a boss room above the up neighbour
    check('(a) a cell next to a boss room: no candidate',
          [CF.SIDES[w['side']] for w in CF.wall_candidates(r6[0])] == ['right'])
    r7 = base_1x1()
    r7['map'][0, 0, 5, 6] = r7['map'][0, 0, 6, 7] = 1
    seen7 = set()
    check('(a) a cleared room without candidates: none, marked seen',
          CF.detect(r7, seen7, None, wall=True, walled=set()) == [] and 'wall' in seen7)
    r8 = base_1x1()
    r8['events'] = EV_CLEARED
    put_door_at(r8, 2, 13.5, 3.5)   # a third door (right): the cleared trigger needs >= 2 doors
    f8 = CF.detect(r8, set(), None, wall=True, walled=set())
    check('(a) the clear record: cleared and wall both found, cleared first',
          [CF.TRIGGERS[t] for t, _ in f8] == ['cleared', 'wall'], str([CF.TRIGGERS[t] for t, _ in f8]))
    check('(a) TRIGGERS: wall before random, random last', CF.TI['wall'] == CF.TI['random'] - 1 and
          CF.TI['random'] == len(CF.TRIGGERS) - 1)


def test_once():
    r = base_1x1()
    seen, walled = set(), set()
    f1 = CF.detect(r, seen, None, wall=True, walled=walled)
    f2 = CF.detect(r, seen, None, wall=True, walled=walled)
    check('(b) not consumed by detect: the next record finds it again', f1 == f2 and len(f1) == 1)
    CF.consume_wall(seen, walled, f1[0][1])
    check('(b) consumed: not again in this room', CF.detect(r, seen, None, wall=True, walled=walled) == [])
    check('(b) a revisit of the room (seen anew): not again in this episode',
          CF.detect(r, set(), None, wall=True, walled=walled) == [])
    other = room([(6, 7)], ONE, (6.5, 3.5), shown=[(6, 6)])
    f3 = CF.detect(other, set(), None, wall=True, walled=walled)
    check('(b) another room fires', [CF.TRIGGERS[t] for t, _ in f3] == ['wall'] and
          f3[0][1][1]['room'] == (1, 6 * 13 + 7), str(f3))
    nxt = base_1x1()
    nxt['stage'] = 2
    check('(b) the same cell on the next floor fires', len(CF.detect(nxt, set(), None, wall=True, walled=walled)) == 1)


def test_geometry():
    big = room([(6, 6), (6, 7), (7, 6), (7, 7)], (1, 14, 1, 26), (13.0, 7.0))
    walls = CF.room_walls(big[0])
    by = {(w['cell'], CF.SIDES[w['side']]): w for w in walls}
    check('(c) 2x2: eight perimeter walls', len(walls) == 8)
    w = by[((6, 7), 'up')]
    check('(c) 2x2: the up wall of the right cell, its bomb spot', w['wall'] == (19.5, 0.0) and w['target'] == (19.5, 0.5))
    w = by[((7, 6), 'left')]
    check('(c) 2x2: the left wall of the lower cell', w['wall'] == (0.0, 10.5) and w['target'] == (0.5, 10.5))
    w = by[((7, 7), 'down')]
    check('(c) 2x2: the down wall of the lower right cell', w['wall'] == (19.5, 14.0) and w['target'] == (19.5, 13.5))
    put_door_at(big, 0, 6.5, -0.5)   # the top door of the left cell
    walls = CF.room_walls(big[0])
    doors = [(w['cell'], CF.SIDES[w['side']]) for w in walls if w['door']]
    check('(c) 2x2: a door on the up wall of the left cell only', doors == [((6, 6), 'up')], str(doors))
    ell = room([(6, 7), (7, 6), (7, 7)], (1, 14, 1, 26), (19.5, 10.5))   # shape 9: the top left cell missing
    by = {(w['cell'], CF.SIDES[w['side']]): w for w in CF.room_walls(ell[0])}
    w = by.get(((7, 6), 'up'))
    check('(c) L: the inner-corner wall above the lower left cell', w is not None and w['wall'] == (6.5, 7.0) and
          w['target'] == (6.5, 7.5) and w['nb'] == (6, 6), str(w))
    w = by.get(((6, 7), 'left'))
    check('(c) L: the inner-corner wall left of the upper right cell', w is not None and w['wall'] == (13.0, 3.5) and
          w['target'] == (13.5, 3.5))
    narrow = room([(6, 6)], (4, 6, 1, 13), (6.5, 1.5))   # IH: 13 x 3 tiles
    sl = {CF.SIDES[w['side']]: w['slots'] for w in CF.room_walls(narrow[0])}
    check('(c) narrow horizontal: no slots up / down, slots left / right',
          sl == dict(up=False, down=False, left=True, right=True), str(sl))
    check('(c) narrow: only the short sides are candidates',
          sorted(CF.SIDES[w['side']] for w in CF.wall_candidates(narrow[0])) == ['left', 'right'])
    vert = room([(6, 6), (7, 6)], (1, 14, 6, 8), (1.5, 7.0))   # IIV: 3 x 14 tiles
    sl = {(w['cell'], CF.SIDES[w['side']]): w['slots'] for w in CF.room_walls(vert[0])}
    check('(c) narrow vertical: left / right no slots', not any(v for (c_, s_), v in sl.items() if s_ in ('left', 'right'))
          and sl[((6, 6), 'up')] and sl[((7, 6), 'down')], str(sl))


def test_scores():
    rng = np.random.default_rng(7)
    sys.path.insert(0, str(HERE.parents[2] / 'macro'))
    try:
        from isaac_macro.secret import win_probabilities
    except Exception as exc:   # (rl/macro not importable here: the port is checked against a Monte Carlo instead)
        win_probabilities = None
        print(f'     (rl/macro not importable: {type(exc).__name__}; Monte Carlo check)')
    for trial in range(6):
        groups = {o: int(rng.integers(0, 5)) for o in (0, -3, -6)}
        groups = {o: c for o, c in groups.items() if c}
        if not groups:
            continue
        v = CF.win_values(groups)
        if win_probabilities is not None:
            offs, k = {}, 0
            for o, c in groups.items():
                for _ in range(c):
                    offs[k] = o
                    k += 1
            ref = win_probabilities(offs)
            ok = all(abs(ref[i] - v[o]) < 1e-12 for i, o in offs.items())
        else:
            cells = [o for o, c in groups.items() for _ in range(c)]
            n = 20000
            sc = rng.integers(10, 15, (n, len(cells))) + np.array(cells)
            mx = sc.max(1, keepdims=True)
            win = (sc == mx) / (sc == mx).sum(1, keepdims=True)
            est = win.mean(0)
            ok = all(abs(est[i] - v[o]) < 0.02 for i, o in enumerate(cells))
        total = sum(v[o] * c for o, c in groups.items())
        check(f'(d) win_values {groups}: = the reference, sums to 1', ok and abs(total - 1) < 1e-9, f'{v}')
    shown = np.zeros((13, 13), bool)
    shown[6, 6] = shown[6, 5] = shown[5, 5] = True
    p = CF.secret_scores(shown, np.zeros((13, 13), bool))
    check('(d) secret_scores: a distribution over the unknown cells next to the map', abs(p.sum() - 1) < 1e-9 and
          p[5, 6] > p[7, 6] > 0 and p[6, 6] == 0 and p[0, 0] == 0)


def move_vec(code):
    """The unit vector of a move code (inverse of SECTOR_MOVE; 0: none)."""
    if code == 0:
        return 0.0, 0.0
    k = SECTOR_MOVE.index(code)
    return math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)


def walk(intent, r, steps, a, speed=0.5, block=None):
    """Drive intent.compose over a simulated walk: record j shows the player after the step of the act decided at
    j - 2 (the one-decision lag, as play_branch: act_j is applied in the step after record j + 1). block(x, y) -> the
    position kept (walls / rocks). Returns the acts."""
    W, H = CF.inner_area(r[0])
    x, y = CF.player_tiles(r[0], (W, H))
    k = int(r['n_doors'][0])
    doors = [(x + 5 * float(r['doors'][0, i, 0]), y + 5 * float(r['doors'][0, i, 1])) for i in range(k)]
    acts, under = [], (0, 0, 0, 0, 0)
    for j in range(steps):
        r['player'][0, 0], r['player'][0, 1] = x / W, y / H
        for i, (X, Y) in enumerate(doors):   # (the doors stay where they are: relative to the moved player)
            r['doors'][0, i, 0], r['doors'][0, i, 1] = (X - x) / 5, (Y - y) / 5
        act = intent.compose(j, a, r)
        acts.append(act)
        dx, dy = move_vec(under[0])
        nx, ny = x + speed * dx, y + speed * dy
        nx, ny = min(max(nx, 0.3), W - 0.3), min(max(ny, 0.3), H - 0.3)
        if block is not None:
            nx, ny = block(nx, ny, x, y)
        x, y = nx, ny
        if under[2]:
            r['bombs'] = max(int(r['bombs'][0]) - 1, 0)
        under = act
    return acts


def test_intent():
    r = base_1x1()
    r['player'][0, 0], r['player'][0, 1] = 6.5 / 13, 5.5 / 7
    info = CF.detect(r, set(), None, wall=True, walled=set())[0][1]
    it = CF.BombIntent(info, h=150)
    a = (3, 2, 0, 1, 0)
    acts = walk(it, r, 40, a)
    bombs = [j for j, x in enumerate(acts) if x[2] == 1]
    up = move_toward(0.0, -1.0)
    check('(e) exactly one bomb', len(bombs) == 1 and it.bombed == bombs[0], str(bombs))
    jb = bombs[0]
    check('(e) before it: moves up toward the spot (the other heads the policy\'s)',
          all(x == (up,) + a[1:] for x in acts[:jb]) and jb >= 5, f'jb {jb} {acts[:3]}')
    check('(e) the bomb action: move 0, the policy\'s shoot / item / pill, bomb 1', acts[jb] == (0, 2, 1, 1, 0))
    check('(e) after it: the policy\'s whole action', all(x == a for x in acts[jb + 1:]))
    check('(e) bombs counted (the bomb step applied)', it.bombs0 == 1 and it.bombs_min == 0, f'{it.bombs0} {it.bombs_min}')
    check('(e) not opened while the map is unchanged', not it.summary()['opened'])
    r['map'][0, 0, 5, 6] = 1   # the neighbour cell shows on the minimap
    it.compose(41, a, r)
    check('(e) opened once the neighbour cell shows after the bomb', it.summary()['opened'])
    it.reset()
    check('(e) reset', it.bombed == -1 and not it.opened and not it.done and it.map3 is None)
    # a door appearing on the bombed wall = opened
    r = base_1x1()
    r['player'][0, 0], r['player'][0, 1] = 6.5 / 13, 1.0 / 7
    it = CF.BombIntent(info, h=150)
    walk(it, r, 6, a)
    put_door_at(r, 2, 6.5, -0.5)
    it.compose(7, a, r)
    check('(e) opened by a door on the bombed wall', it.bombed >= 0 and it.summary()['opened'])
    # stuck near the wall (a rock): the bomb after BOMB_STUCK decisions
    r = base_1x1()
    r['player'][0, 0], r['player'][0, 1] = 6.5 / 13, 5.5 / 7
    it = CF.BombIntent(info, h=150)
    acts = walk(it, r, 40, a, block=lambda nx, ny, x, y: (nx, max(ny, 1.8)))
    bombs = [j for j, x in enumerate(acts) if x[2] == 1]
    check('(e) stuck 1.3 tiles away: one bomb after the stuck count', len(bombs) == 1 and bombs[0] >= CF.BOMB_STUCK,
          str(bombs))
    r = base_1x1()
    r['player'][0, 0], r['player'][0, 1] = 6.5 / 13, 5.5 / 7
    it = CF.BombIntent(info, h=30)
    acts = walk(it, r, 40, a, block=lambda nx, ny, x, y: (nx, max(ny, 3.5)))
    check('(e) stuck 3 tiles away: no bomb, the policy after the cap', not any(x[2] for x in acts) and
          acts[-1] == a and it.bombed == -1)
    r = base_1x1()
    r['player'][0, 0], r['player'][0, 1] = 6.5 / 13, 5.5 / 7
    it = CF.BombIntent(info, h=150)
    acts = []
    for j in range(40):   # bombs gone before arrival (no step simulation needed: the player is already close)
        if j == 0:
            r['player'][0, 1] = 0.6 / 7
            r['bombs'] = 0
        acts.append(it.compose(j, a, r))
    check('(e) no bomb held at the spot: no bomb, the policy', not any(x[2] for x in acts) and acts[0] == a)
    r = base_1x1()
    r['player'][0, 0], r['player'][0, 1] = 6.5 / 13, 5.5 / 7
    it = CF.BombIntent(info, h=150)
    it.compose(0, a, r)
    r2 = r.copy()
    r2['events'] = EV_NEW_ROOM
    check('(e) the room changed: no bomb, the policy', it.compose(1, a, r2) == a and it.done and it.bombed == -1)


def test_search():
    pt = T2.V2Point(target=('wall', dict(x=6.5, y=0.5, side=0, cell=71, p=0.4, n=2, room=(1, 84))))
    cfg = type('C', (), dict(cf_branches=4, teacher_v2_sticky=4.0, cf_sticky_eps=0.25))()
    its = CF.intents(pt, cfg, np.random.default_rng(0), 30)
    check('(f) intents: bomb_at + 2 sticky for K = 4', [i.kind for i in its] == ['bomb_at', 'sticky', 'sticky'] and
          its[0].code == T2.INTENT_BOMB)
    calls = []

    class FakeS:
        def close(self):
            pass

    def fake_restore(pt_, d_, ctx_):
        return FakeS(), 10.0

    def fake_branch(S, ctx_, pt_, d_, hp0, intent, policy, reseed=None, keep=True, trace=None):
        intent.reset()   # (play_branch resets the intent)
        calls.append((intent.kind, reseed, keep))
        n_rows = keep if isinstance(keep, int) and not isinstance(keep, bool) else 0
        if intent.kind == 'bomb_at' and reseed is None:
            intent.bombed, intent.opened = 12, True
        base = {'bomb_at': 3.0, 'sticky': 0.5}[intent.kind]
        rows = [np.zeros(1, ROW)[0] for _ in range(n_rows)]
        return dict(score=base - (0.1 if reseed is not None else 0.0), n=90, end=T2.END_CAP, rows=rows,
                    pend=[(0, 0, 0, 0, 0)] * n_rows, act=[(1, 0, 0, 0, 0)] * n_rows, ret=np.zeros(n_rows),
                    value=[0.0] * n_rows, logp=[0.0] * n_rows)
    rw = parse_reward('damage=1,hurt=0.75,room=1,explore=0.5,boss=2,exit=3,death=2,timeout=1,time=0.002')
    cfg = type('C', (), dict(cf_branches=4, teacher_v2_sticky=4.0, cf_sticky_eps=0.25, cf_target_seconds=20.0,
                             frames_per_decision=4, cf_margin=0.3, cf_distill_steps=8, cf_wall_distill=64))()
    log = T2.V2Log()
    for t in range(60):
        rr = np.zeros(1, ROW)
        log.add(rr, 0, t, rw=rw)
        log.value.append(0.0)
        log.vg.append(0.0)
    ctx = type('X', (), dict(cfg=cfg, cap=20, gamma=0.99, rw=rw, run=True, exit_end=False, boot_vg=True, gains=False,
                             stats=dict(replicates=0)))()
    pt = T2.V2Point(at=20, offset=5, log=log, target=pt.target, bank=None, kind=T2.KIND_CF, trig=CF.TI['wall'])
    res = CF.cf_search(pt, ctx, None, np.random.default_rng(0), branch=fake_branch, restore_fn=fake_restore)
    check('(f) branches: bomb_at kept for cf_wall_distill records, sticky for cf_distill_steps, the replicate',
          [c[0] for c in calls] == ['bomb_at', 'sticky', 'sticky', 'bomb_at'] and calls[0][2] == 64 and
          calls[1][2] == 8 and calls[3][1] is not None, str(calls))
    w = res.get('wall')
    check('(f) the wall summary from the first run (not the replicate)', w is not None and w['bombed'] == 12 and
          w['opened'] and abs(w['gain'] - (3.0 - res['taken'])) < 1e-12 and w['p'] == 0.4 and w['side'] == 0, str(w))
    check('(f) improving, best = bomb_at, distill through the bomb decision', res['improving'] and w['best'] and
          res['best']['intent'].kind == 'bomb_at' and res['distill'] == 13, f"distill {res.get('distill')}")
    rows_, meta_ = T2.records_of(res, ctx, pt, 0, 2.0, steps=res['distill'])
    check('(f) records_of: 13 records, intent INTENT_BOMB, trig wall + 1', len(rows_) == 13 and
          np.all(meta_[:, T2.VI['intent']] == T2.INTENT_BOMB) and np.all(meta_[:, T2.VI['trig']] == CF.TI['wall'] + 1))
    calls.clear()
    cfg.cf_wall_distill = 10
    res = CF.cf_search(pt, ctx, None, np.random.default_rng(0), branch=fake_branch, restore_fn=fake_restore)
    check('(f) distill capped at cf_wall_distill', res['distill'] == 10 and calls[0][2] == 10)
    calls.clear()
    pt2 = T2.V2Point(at=20, offset=5, log=log, target=('ent', 'coin'), bank=None)

    def fake_branch2(S, ctx_, pt_, d_, hp0, intent, policy, reseed=None, keep=True, trace=None):
        calls.append((intent.kind, reseed, keep))
        return dict(score={'target': 3.0, 'sticky': 0.5}[intent.kind], n=30, end=T2.END_CAP, rows=[], pend=[], act=[],
                    ret=np.zeros(0), value=[], logp=[])
    res = CF.cf_search(pt2, ctx, None, np.random.default_rng(0), branch=fake_branch2, restore_fn=fake_restore)
    check('(f) other triggers unchanged: keep cf_distill_steps, distill cf_distill_steps, no wall summary',
          calls[0][2] == 8 and res['distill'] == 8 and 'wall' not in res)


def test_defer():
    a = (3, 2, 0, 1, 0)
    # a door target in a room still being fought: the policy's whole action until the clear, then the walk
    r = room([(6, 6)], ONE, (6.5, 5.5), shown=[(5, 6)], clear=0.0)
    put_door_at(r, 0, 6.5, -0.5)   # the up door
    it = CF.TargetIntent(('door', 0), h=6, defer=True)
    acts = [it.compose(j, a, r) for j in range(5)]
    check('(g) deferred door target: the policy\'s whole action while the room is not clear', acts == [a] * 5 and
          it.keep_from() is None)
    r['events'] = EV_CLEARED
    r['player'][0, 23] = 1.0
    up = move_toward(0.0, -1.0)
    out = it.compose(5, a, r)
    check('(g) deferred door target: the walk from the clear record on', out == (up,) + a[1:] and it.keep_from() == 5,
          f'{out} {it.keep_from()}')
    r['events'] = 0
    acts = [it.compose(j, a, r) for j in range(6, 13)]
    check('(g) deferred door target: the cap h counts from the clear (5 more moves, then the policy)',
          acts[:5] == [(up,) + a[1:]] * 5 and acts[5:] == [a] * 2, str(acts))
    it.reset()
    check('(g) deferred: reset waits again', it.keep_from() is None)
    nd = CF.TargetIntent(('door', 0), h=6, defer=False)
    r['player'][0, 23] = 0.0
    check('(g) not deferred: the walk at once (as B34)', nd.compose(0, a, r) == (up,) + a[1:] and nd.keep_from() == 0)
    # the room changes before its clear: the intent never starts
    r = room([(6, 6)], ONE, (6.5, 5.5), clear=0.0)
    put_door_at(r, 0, 6.5, -0.5)
    it = CF.TargetIntent(('door', 0), h=6, defer=True)
    it.compose(0, a, r)
    r2 = room([(5, 6)], ONE, (6.5, 5.5), clear=1.0)
    r2['events'] = EV_NEW_ROOM
    check('(g) the room changed while waiting: never starts', it.compose(1, a, r2) == a and it.gone and
          it.compose(2, a, r2) == a and it.keep_from() is None)
    # a deferred bomb_at (a stand-in uncleared record first): the policy, then moves, exactly one bomb, the policy
    r = base_1x1()
    r['player'][0, 0], r['player'][0, 1] = 6.5 / 13, 5.5 / 7
    info = CF.detect(r, set(), None, wall=True, walled=set())[0][1]
    it = CF.BombIntent(info, h=150, defer=True)
    r['player'][0, 23] = 0.0
    pre = [it.compose(j, a, r) for j in range(4)]
    r['player'][0, 23] = 1.0
    W, H = 13.0, 7.0
    x, y, under, acts = 6.5, 5.5, (0, 0, 0, 0, 0), []
    for j in range(4, 44):
        r['player'][0, 0], r['player'][0, 1] = x / W, y / H
        act = it.compose(j, a, r)
        acts.append(act)
        dx, dy = move_vec(under[0])
        x, y = min(max(x + 0.5 * dx, 0.3), W - 0.3), min(max(y + 0.5 * dy, 0.3), H - 0.3)
        under = act
    bombs = [j for j, x_ in enumerate(acts) if x_[2] == 1]
    check('(g) deferred bomb_at: the policy while uncleared', pre == [a] * 4)
    check('(g) deferred bomb_at: moves, then exactly one bomb, then the policy', len(bombs) == 1 and
          all(x_ == (up,) + a[1:] for x_ in acts[:bombs[0]]) and acts[bombs[0]] == (0, 2, 1, 1, 0) and
          all(x_ == a for x_ in acts[bombs[0] + 1:]) and it.j0 == 4 and it.bombed == 4 + bombs[0], str(bombs))
    # intents() with cfg.cf_defer_target: door and wall targets deferred, pickups not
    cfg = type('C', (), dict(cf_branches=2, teacher_v2_sticky=4.0, cf_sticky_eps=0.25, cf_defer_target=True))()
    rng = np.random.default_rng(0)
    d_ = CF.intents(T2.V2Point(target=('door', 1)), cfg, rng, 30)[0]
    w_ = CF.intents(T2.V2Point(target=info), cfg, rng, 30)[0]
    e_ = CF.intents(T2.V2Point(target=('ent', 'coin')), cfg, rng, 30)[0]
    check('(g) intents: door and wall deferred, pickups not', d_.defer and w_.defer and not e_.defer and
          e_.keep_from() == 0)
    cfg.cf_defer_target = False
    check('(g) intents: off = nothing deferred', not CF.intents(T2.V2Point(target=('door', 1)), cfg, rng, 30)[0].defer)
    # cf_search: the deferred branch's start and the distill count from it
    calls = []

    class FakeS:
        def close(self):
            pass

    def fake_branch(S, ctx_, pt_, d_, hp0, intent, policy, reseed=None, keep=True, trace=None):
        intent.reset()
        calls.append((intent.kind, reseed, keep))
        out = dict(score=0.5, n=90, end=T2.END_CAP, rows=[], pend=[], act=[], ret=np.zeros(0), value=[], logp=[])
        if intent.kind == 'bomb_at' and reseed is None:
            intent.j0, intent.bombed = 30, 41
            out.update(score=3.0, rows=[np.zeros(1, ROW)[0]] * 20, pend=[(0,) * 5] * 20, act=[(1, 0, 0, 0, 0)] * 20,
                       ret=np.zeros(20), value=[0.0] * 20, logp=[0.0] * 20)
            out['from'] = 30
        elif intent.kind == 'bomb_at':
            out['score'] = 2.9
        return out
    rw = parse_reward('damage=1,hurt=0.75,room=1,explore=0.5,boss=2,exit=3,death=2,timeout=1,time=0.002')
    cfg = type('C', (), dict(cf_branches=3, teacher_v2_sticky=4.0, cf_sticky_eps=0.25, cf_target_seconds=20.0,
                             frames_per_decision=4, cf_margin=0.3, cf_distill_steps=8, cf_wall_distill=64,
                             cf_defer_target=True))()
    log = T2.V2Log()
    for t in range(60):
        log.add(np.zeros(1, ROW), 0, t, rw=rw)
        log.value.append(0.0)
        log.vg.append(0.0)
    ctx = type('X', (), dict(cfg=cfg, cap=20, gamma=0.99, rw=rw, run=True, exit_end=False, boot_vg=True, gains=False,
                             stats=dict(replicates=0)))()
    pt = T2.V2Point(at=20, offset=5, log=log, target=info, bank=None, kind=T2.KIND_CF, trig=CF.TI['wall'])
    res = CF.cf_search(pt, ctx, None, rng, branch=fake_branch, restore_fn=lambda *x: (FakeS(), 10.0))
    check('(g) cf_search: defer info (started at 30) and distill = bomb - j0 + 1 = 12',
          res.get('defer') == dict(started=True, wait=30) and res['distill'] == 12 and res['wall']['j0'] == 30,
          f"{res.get('defer')} {res['distill']}")


if __name__ == '__main__':
    t0 = time.perf_counter()
    for fn in (test_detect, test_once, test_geometry, test_scores, test_intent, test_search, test_defer):
        fn()
    print(f'{len(FAIL)} failed' + (': ' + ', '.join(FAIL) if FAIL else '') + f'  ({time.perf_counter() - t0:.1f} s)')
    sys.exit(1 if FAIL else 0)
