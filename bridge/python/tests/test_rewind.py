"""The reward ledger that follows the game's Glowing Hour Glass rewinds (2026-11-20, reward-rewind-design.md v1; the
user / the coordinator): the trainer's book (train_tok Episodes.set_rewind / _rewind / _snap), the workers' encoder
(tok_obs.encode_row: ROW 'rewind', the restored state, --stall-progress first, EV_ACTIVE), the workers' scorers
(tok_branch.RewindLedger), the lean decoder's count block and the configuration checks.

Book cases follow the design's section 7: a clear -> rewind -> clear loop pays nothing net (plan I: the discounted
return of a loop is its time terms only; plan P leaves the discount's gap), a hurt rewound is refunded, a new item /
boss rewound is cancelled and paid again (once) in the new timeline, an archive start rewound before any room change
goes back to the episode's start, the rewind record's own jumps (hearts, coins, items) pay nothing, two rewinds go back
to the same checkpoint, and without rewinds every reward is the old one bit for bit."""
import struct
import sys
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_tok as T   # noqa: E402
from isaac_bridge import abplus_lean as AL   # noqa: E402
from isaac_bridge import tok_obs as TO   # noqa: E402
from isaac_bridge.tok_branch import RewindLedger   # noqa: E402
from isaac_bridge.tok_obs import EV_BOSS, EV_CLEARED, EV_EXIT, EV_NEW_ROOM, ROW   # noqa: E402

G = 0.995
RW = dict(damage=1.0, hurt=0.75, room=1.0, explore=0.5, boss=2.0, exit=3.0, death=2.0, timeout=1.0, time=0.002,
          heal=0.5, resource=0.05, clear=1.0)
CK = TO.REWIND_CHECKPOINT


def _row(first=0, events=0, hurt=0.0, damage=0.0, done=0, rewind=0, units=6, coins=0, active=0, inv=(), stage=1,
         boss=False, t=0):
    r = np.zeros(1, ROW)
    r['t'] = t
    r['first'], r['done'], r['seed'], r['group'] = first, done, 7, 0
    r['events'], r['hurt'], r['damage'], r['rewind'] = events, hurt, damage, rewind
    r['player'][0, 7] = units / 12
    r['player'][0, 10] = coins / 50
    r['pitem'][0, 0] = active
    for k, v in enumerate(inv):
        r['inv'][0, k] = (v, 1)
    r['pinv'][0, 6] = len(inv) / 16
    r['stage'] = stage
    return r


class Book:
    """One worker's records through Episodes.arrive; rewards[i]: the reward of the decision the (i + 1)-th record
    closed."""

    def __init__(self, rewind=None, cap=512, gamma_g=None, **terms):
        self.rw = dict(RW, **terms)
        self.roll = T.Rollout(1, cap, close=True, value_g=gamma_g is not None)
        self.book = T.Episodes(1, self.rw, True, run=True, items=True)
        if rewind is not None:
            self.book.set_rewind(*rewind, G, cap, gamma_g=gamma_g)
        self.pending = np.zeros((1, 3), np.int64)
        self.rewards = []

    def rec(self, **kw):
        k_before = int(self.roll.open[0])
        go = self.book.arrive(self.roll, _row(**kw), np.array([0]), self.pending)
        if k_before >= 0:
            self.rewards.append(float(self.roll.reward[0, k_before]))
        if go[0]:
            k = int(self.roll.length[0])
            self.roll.open[0], self.roll.length[0] = k, k + 1
        return self


def loop(b, n, hurt_in_b=0.0, heart=False):
    """The policy's loop of seed 476341 (reward-rewind-design.md V3, replay page 2), n times: from room A into B (a
    checkpoint record: B new the first time), monsters hurt, B cleared, (a heart taken), the hour glass used (the rewind
    record: back in A, the hearts as at the checkpoint)."""
    for k in range(n):
        b.rec(events=EV_NEW_ROOM if k == 0 else 0, rewind=CK, units=4 if heart else 6)
        b.rec(damage=0.6, hurt=hurt_in_b, units=(4 if heart else 6) - (2 if hurt_in_b else 0))
        b.rec(damage=0.4, events=EV_CLEARED, units=(4 if heart else 6) - (2 if hurt_in_b else 0))
        if heart:
            b.rec(units=6)
        b.rec(rewind=1, units=4 if heart else 6)   # (the rewind: no event; the hearts back -- not a heal)
        b.rec()                                    # (walking in A)


def disc_net(rewards, start, end):
    """sum gamma^(j - start) r_j over rewards[start:end] (each loop's records, discounted to its first)."""
    return sum(G ** (j - start) * rewards[j] for j in range(start, end))


@pytest.mark.parametrize('interest,rtime', [(1, 0), (1, 1), (0, 0), (0, 1)])
def test_a_loop_pays_nothing_net(interest, rtime):
    b = Book(rewind=(interest, rtime, 4.0))
    b.rec(first=1)
    loop(b, 6)
    r = b.rewards
    per = 5   # records per loop
    for k in range(1, 6):   # (the first loop's explore is real: the loops after it)
        s = k * per
        seg = r[s:s + per]
        # the loop: entering B (time), damage 0.6, damage 0.4 + clear, the rewind, walking
        assert seg[1] == pytest.approx(0.6 - 0.002) and seg[2] == pytest.approx(1.4 - 0.002)
        net = disc_net(r, s, s + per - 1)   # (up to the rewind record; the walk after it is A's)
        # the cancellable parts of the two decisions after the checkpoint (the entry into B), 2 and 1 decisions
        # before the rewind; the rewind record pays its time term and the rollback
        x = [0.6 - 0.002 * rtime, 1.4 - 0.002 * rtime]
        c = [G ** -2, G ** -1] if interest else [1.0, 1.0]
        want = -0.002 + G * (0.6 - 0.002) + G ** 2 * (1.4 - 0.002) + G ** 3 * (-0.002 - c[0] * x[0] - c[1] * x[1])
        assert net == pytest.approx(want, abs=1e-6), (k, net, want)
        if interest:   # (plan I: the stretch's rewards and their rollback cancel in the discounted return exactly)
            kept = -0.002 * (1 + G + G ** 2 + G ** 3) if not rtime else -0.002 * (1 + G ** 3)
            assert net == pytest.approx(kept, abs=1e-6)   # (Rollout stores float32)
    fin = b.rec(done=2).book.finished[-1]
    assert fin['rooms'] == 0 and fin['rewinds'] == 6   # (the real timeline: the last clear rewound too)
    assert sum(fin['r'].values()) == pytest.approx(fin['ret'], abs=1e-3)   # (the shares still add up to the return)


def test_plan_i_no_time_matches_the_offline_numbers():
    """(R6 of the design note) plan I, time kept: a loop's discounted net is exactly its time terms; plan P keeps
    a positive residue (the reward before the rollback, the rollback discounted further)."""
    nets = {}
    for interest in (1, 0):
        b = Book(rewind=(interest, 0, 4.0))
        b.rec(first=1)
        loop(b, 3)
        nets[interest] = disc_net(b.rewards, 5, 9)
    assert nets[1] == pytest.approx(-0.002 * (1 + G + G ** 2 + G ** 3), abs=1e-6)
    assert nets[0] > 0.004


def test_hurt_rewound_is_refunded_then_counted_again():
    b = Book(rewind=(1, 0, 4.0))
    b.rec(first=1)
    b.rec(rewind=CK)                       # into B
    b.rec(hurt=2.0, units=4)               # -1.5
    b.rec(rewind=1, units=6)               # back: the hurt refunded (grown by the interest), the hearts not a heal
    r = b.rewards
    assert r[1] == pytest.approx(-1.5 - 0.002)
    assert r[2] == pytest.approx(-0.002 + 1.5 * G ** -1, abs=1e-6)
    b.rec(rewind=CK)
    b.rec(hurt=2.0, units=4)               # hurt again in the new timeline: charged
    assert b.rewards[-1] == pytest.approx(-1.5 - 0.002)
    fin = b.rec(done=2, units=4).book.finished[-1]
    assert fin['hurt'] == pytest.approx(2.0) and fin['healed'] == pytest.approx(0.0)


def test_a_new_item_and_a_boss_rewound_then_paid_once_again():
    b = Book(rewind=(1, 0, 4.0), item=1.0)
    b.rec(first=1, inv=(1,))
    b.rec(rewind=CK, inv=(1,))
    b.rec(inv=(7, 1))                                   # item 7: +1
    b.rec(events=EV_CLEARED | EV_BOSS, inv=(7, 1))      # the boss: +2 +1
    assert b.rewards[1] == pytest.approx(1.0 - 0.002) and b.rewards[2] == pytest.approx(3.0 - 0.002)
    b.rec(rewind=1, inv=(1,))                           # back: both cancelled; 7 no longer in the episode's set
    assert b.book.seen_c[0] == {1} and b.book.boss[0] == 0 and b.book.rooms[0] == 0
    b.rec(rewind=CK, inv=(1,))
    b.rec(inv=(7, 1))                                   # item 7 again: paid again, once
    b.rec(inv=(7, 1))
    assert b.rewards[-2] == pytest.approx(1.0 - 0.002) and b.rewards[-1] == pytest.approx(-0.002)
    fin = b.rec(done=2, inv=(7, 1)).book.finished[-1]
    assert fin['items_new'] == 1 and fin['boss'] == 0


def test_the_rewind_records_own_jumps_pay_nothing():
    """hearts, coins and a never-held active item jumping back at the rewind: none of heal / resource / item."""
    b = Book(rewind=(1, 0, 4.0), item=1.0)
    b.rec(first=1, units=4, coins=0, active=105)
    b.rec(rewind=CK, units=4, coins=0, active=105)
    b.rec(units=4, coins=0, active=105)
    b.rec(rewind=1, units=6, coins=9, active=422)   # (as if the store held more: none of it is this timeline's)
    x = b.rewards[1] + 0.002   # (the one decision rolled back: its time kept)
    assert b.rewards[-1] == pytest.approx(-0.002 - x * G ** -1, abs=1e-6)
    assert b.book.healed[0] == 0 and b.book.gathered[0] == 0 and 422 not in b.book.seen_c[0]


def test_an_archive_start_rewound_before_any_room_change():
    """(the coordinator, V4 2) an episode started from a parked state: its game's store is older than the episode --
    the rollback goes back to the episode's first record, the once-per-episode sets to theirs."""
    b = Book(rewind=(1, 0, 4.0), item=1.0)
    b.rec(first=1, inv=(1, 2), units=4)            # (an archive start: holds 1, 2)
    b.rec(inv=(9, 1, 2), units=4)                  # 9: +1
    b.rec(events=EV_CLEARED, inv=(9, 1, 2), units=6)   # +1 room, +1 heal
    b.rec(rewind=1, inv=(1,), units=2)             # the store was before the episode: back to its start
    assert b.book.seen_c[0] == {1, 2} and b.book.rooms[0] == 0 and b.book.healed[0] == 0
    want = -0.002 - sum(G ** -(2 - j) * (b.rewards[j] + 0.002) for j in range(2))
    assert b.rewards[-1] == pytest.approx(want, abs=1e-6)


def test_two_rewinds_go_back_to_the_same_checkpoint():
    b = Book(rewind=(1, 0, 4.0))
    b.rec(first=1)
    b.rec(rewind=CK)
    b.rec(events=EV_CLEARED)
    b.rec(rewind=1)                 # cancels the clear
    b.rec(damage=0.5)
    b.rec(rewind=1)                 # cancels only the damage after the first rewind (the same store)
    assert b.rewards[-1] == pytest.approx(-0.002 - 0.5 * G ** -1, abs=1e-6)


def test_without_rewinds_every_reward_is_the_old_one_bit_for_bit():
    """random record streams (test_reward_terms._stream: events, hurts, damage, items, stages) with checkpoint bits and
    no rewind: the same rewards and episode lines as the book without the ledger."""
    import test_reward_terms as R
    rw = dict(R.OLD, item=1.0, trinket=0.3, depth=0.25, nohit=0.5, heal=0.5, resource=0.05)
    for seed in (1, 2, 3):
        stream = R._stream(seed)
        rng = np.random.default_rng(seed)
        for rows, _ in stream:
            rows['rewind'] = np.where(rng.random(len(rows)) < 0.2, CK, 0)
        out = []
        for on in (False, True):
            W = 6
            roll = T.Rollout(W, 2048, close=True)
            book = T.Episodes(W, rw, True, run=True, items=True)
            if on:
                book.set_rewind(1, 0, 4.0, G, 4096)
            pending = np.zeros((W, 3), np.int64)
            for rows, ws in stream:
                go = book.arrive(roll, rows, ws.astype(np.int64), pending)
                for w in ws[go]:
                    k = int(roll.length[w])
                    roll.open[w], roll.length[w] = k, k + 1
            fin = [{k: v for k, v in f.items() if k not in ('rewinds', 'active_end')} for f in book.finished]   # (the ledger's own fields)
            out.append((roll.reward.copy(), fin))
        assert np.array_equal(out[0][0], out[1][0]) and out[0][1] == out[1][1], seed


def test_the_workers_scorers_ledger():
    led = RewindLedger((1, 1, 0, 4.0), G, RW)
    r = [led.step(x, b) for x, b in ((0.5, CK), (1.0, 0), (0.3, 0), (0.0, 1))]
    assert r[:3] == [0.5, 1.0, 0.3]
    assert r[3] == pytest.approx(-0.002 - (1.002 * G ** -2 + 0.302 * G ** -1), abs=1e-12)
    off = RewindLedger((0, 1, 0, 4.0), G, RW)
    assert [off.step(x, b) for x, b in ((0.5, CK), (1.0, 1))] == [0.5, 1.0]


# ---- the workers' encoder (tok_obs.encode_row) on synthetic lean payloads ----

import test_obs_v2_smoke as V   # noqa: E402


def payload(k, room_idx, clear, damage_taken=0.0, monsters_hp=30.0, hearts=6, rewinds=None, terrain=False, stage=1):
    p, d, e, room = V.world(0)
    p['hearts'], p['soul'], p['bone'], p['black'], p['eternal'] = hearts, 0, 0, 0, 0
    room = room[:4] + (room_idx, stage) + room[6:7] + (100 + k,) + room[8:]
    flags = (1 if clear else 0) + (2 if terrain else 0) + (128 if rewinds is not None else 0)
    out = struct.pack('<IIIBBBB', AL.MAGIC, 100 + 4 * k, 200 + 4 * k, flags, len(p), len(d), 0)
    out += struct.pack('<iiiiiiiiffff', *room)
    out += struct.pack('<dddd', damage_taken, monsters_hp, 30.0, 1.0)
    out += p.tobytes() + d.tobytes() + struct.pack('<H', len(e))
    rec = np.ascontiguousarray(e).view(np.uint8).reshape(len(e), AL.ENTITY.itemsize)
    out += rec[:, :148].tobytes()
    if terrain:
        tj, gj = V.terrain_json()
        out += struct.pack('<II', 7, len(tj)) + tj + struct.pack('<I', len(gj)) + gj
    if rewinds is not None:
        out += struct.pack('<i', rewinds)
    return out


class Worker:
    def __init__(self, stall=0, first=False, rewind=True):
        self.dec = AL.LeanDecoder()
        self.st = TO.EpisodeState(10 ** 6, floor=True, stall=stall, run=True, stall_first=first, rewind=rewind)
        self.t = 0
        self.out = []

    def rec(self, room_idx, clear, **kw):
        o = self.dec.decode(payload(self.t, room_idx, clear, terrain=self.t == 0 or kw.pop('terrain', False), **kw))
        row = np.zeros(1, ROW)
        done = TO.encode_row(o, self.st, row, self.t)
        self.out.append((int(row['events'][0]), int(row['rewind'][0]), float(row['hurt'][0]),
                         float(row['damage'][0]), done))
        self.t += 1
        return self.out[-1]


def test_the_count_block_decodes():
    o = AL.LeanDecoder().decode(payload(0, 84, True, rewinds=7, terrain=True))
    assert o.rewinds == 7
    assert AL.LeanDecoder().decode(payload(0, 84, True, terrain=True)).rewinds is None


def test_worker_rewind_field_restore_and_checkpoints():
    w = Worker()
    A, B = 84, 85
    assert w.rec(A, True, rewinds=3) == (0, 0, 0.0, 0.0, 0)                      # first: count known, no rewind
    ev, rb, *_ = w.rec(B, False, rewinds=None, monsters_hp=30.0)                 # into B: new room, a checkpoint
    assert ev & EV_NEW_ROOM and rb == CK
    w.rec(B, False, monsters_hp=12.0)                                            # monsters hurt
    ev, rb, *_ = w.rec(B, True, monsters_hp=0.0)                                 # cleared
    assert ev & EV_CLEARED and rb == 0
    ev, rb, hu, dm, _ = w.rec(A, True, rewinds=4, damage_taken=0.0)              # the rewind: back in A
    assert (ev, rb, hu, dm) == (0, 1, 0.0, 0.0)
    assert B in w.st.visited   # (the checkpoint was taken after B's entry: its explore stays paid, not again)
    ev, rb, *_ = w.rec(B, False, monsters_hp=30.0)                               # into B again: a checkpoint, not new
    assert not ev & EV_NEW_ROOM and rb == CK
    assert w.rec(B, True, rewinds=0, monsters_hp=0.0)[1] == 0                    # (a count that went down: no rewind)


def test_stall_first_ends_a_rewind_loop():
    """--stall-progress first: the loop's repeated clears are no progress -- the episode stalls STALL records after
    the first loop's last first-time event; with 'events' it never does."""
    STALL = 12

    def run(first):
        w = Worker(stall=STALL, first=first)
        A, B = 84, 85
        w.rec(A, True, rewinds=0)
        n = 1
        for k in range(30):
            w.rec(B, False, monsters_hp=30.0)
            w.rec(B, False, monsters_hp=10.0)
            w.rec(B, True, monsters_hp=0.0)
            if w.rec(A, True, rewinds=n)[4]:
                break
            n += 1
            if w.out[-1][4]:
                break
        return [o[4] for o in w.out]

    dones = run(True)
    assert 3 in dones and dones.index(3) < 4 + 4 + STALL + 1, dones
    assert 3 not in run(False)


def test_ev_active_only_on_real_uses():
    st = TO.EpisodeState(100, floor=True, run=True, items=True)
    row = np.zeros(1, ROW)

    def step(active, charge, maxc):
        row['events'] = 0
        st.max_charge = maxc
        TO._items_step(st, row, 1, active, charge, 0)
        return int(row['events'][0]) & TO.EV_ACTIVE

    assert step(105, 6, 6) == 0
    assert step(105, 0, 6) == TO.EV_ACTIVE      # used: the charge went down
    assert step(0, 0, 0) == 0                   # a pedestal swap's pickup animation: the slot empty a moment
    assert step(422, 2, 2) == 0                 # the other item in hand
    assert step(127, 0, 0) == 0
    assert step(0, 0, 0) == TO.EV_ACTIVE        # a single-use item (no charges) gone: used


def test_rewind_refused_where_it_cannot_follow(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv('ISAAC_RL_VISIBLE_V3', raising=False)
    cases = ((['--rewind', '1'], 'floor or run only'),
             (['--mode', 'run', '--rewind', '1', '--block-memory', '1'], 'not with --block-memory'))
    for i, (extra, msg) in enumerate(cases):
        with pytest.raises(SystemExit):
            T.main(['--groups-file', str(tmp_path / 'x.json'), '--out', str(tmp_path / f'o{i}')] + extra)
        assert msg in capsys.readouterr().err, extra


def test_row_layout_unchanged_but_the_padding_byte():
    assert ROW.itemsize == 151880 and ROW.fields['rewind'][1] == 20793
    assert ROW.fields['visible_schema'][1] == 20796 and TO.ROWC.fields['rewind'][1] == 20793


# ---- real sequences (tests/fixtures/rewind_loops.npz, made from replay pages 1 and 2; see its maker's docstring) ----

def _fixture_rows(name):
    z = np.load(HERE / 'fixtures' / 'rewind_loops.npz')
    n = len(z[f'{name}_events'])
    rows = np.zeros(n, ROW)
    for k in ('events', 'hurt', 'damage', 'stage', 'done', 'first', 'rewind'):
        rows[k] = z[f'{name}_{k}']
    rows['player'][:, 7:11] = z[f'{name}_player']
    rows['pitem'][:, :3] = z[f'{name}_pitem']
    rows['pinv'][:, 6] = z[f'{name}_pinv6']
    rows['inv'] = z[f'{name}_inv']
    rows['first'][0] = 1
    rows['done'] = 0
    return rows


def _feed(rows, rewind):
    b = Book(rewind=rewind, cap=len(rows) + 8, item=1.0)
    for r in rows:
        k_before = int(b.roll.open[0])
        go = b.book.arrive(b.roll, r[None], np.array([0]), b.pending)
        if k_before >= 0:
            b.rewards.append(float(b.roll.reward[0, k_before]))
        if go[0]:
            k = int(b.roll.length[0])
            b.roll.open[0], b.roll.length[0] = k, k + 1
    return np.array(b.rewards)


def _loops(rows):
    """(c, k) per rewind: c the last checkpoint record before it (or the first record), k the rewind record."""
    rb = rows['rewind'].astype(np.int64)
    out, c = [], 0
    for t in range(1, len(rows)):
        if rb[t] & 127:
            out.append((c, t))
        elif rb[t] & CK:
            c = t
    return out


@pytest.mark.parametrize('name', ['pol', 'hurt', 'heart'])
def test_real_loops(name):
    """Replay page 2 (the policy's own loop, u2200, seed 476341: swap D6 for the hour glass, use it at once) and page 1
    (the scripted 2x2-room loops, each loop first hurt / a heart eaten). Without the ledger every loop pays (page 2:
    about +2.7 a loop); with it (plan I, time kept) each loop's discounted return from its checkpoint is its time terms
    only."""
    rows = _fixture_rows(name)
    old = _feed(rows, None)
    new = _feed(rows, (1, 0, 4.0))
    loops = _loops(rows)
    assert len(loops) >= 5
    # (rewards[i] closes at record i + 1)
    paid = [old[c:k].sum() for c, k in loops]
    assert np.mean(paid) > (2.0 if name == 'pol' else 0.4), paid
    for c, k in loops:
        net = sum(G ** (j - c - 1) * new[j - 1] for j in range(c + 1, k + 1))
        want = -0.002 * sum(G ** m for m in range(k - c))
        assert net == pytest.approx(want, abs=2e-5), (name, c, k, net, want)


# ---- the user's two guarantees (2026-11-20, via the coordinator) ----
# (The reward of a rewind lands, as every reward, on the decision its record closes: with the one-decision action lag
#  the decision after the one that pressed the item button.)

@pytest.mark.parametrize('interest', [1, 0])
def test_user_req_u1_hurt_then_rewind_refunds_the_hurt(interest):
    """U1: into B, hurt, nothing else, the hour glass: the rewind's reward at least the hurt's penalty (plan I: a
    little more, the interest; plan P: equal, less the record's time term), the episode's hurt back to the checkpoint."""
    b = Book(rewind=(interest, 0, 4.0))
    b.rec(first=1)
    b.rec(rewind=CK)
    b.rec(hurt=2.0, units=4)                 # -1.5 (and its time)
    hurt_before = float(b.book.hurt[0])
    b.rec(rewind=1, units=6)
    refund = b.rewards[-1] + 0.002           # (the rewind record's own time term kept)
    assert hurt_before == 2.0 and float(b.book.hurt[0]) == 0.0
    if interest:
        assert refund == pytest.approx(1.5 * G ** -1, abs=1e-6) and refund > 1.5
    else:
        assert refund == pytest.approx(1.5, abs=1e-6)


@pytest.mark.parametrize('interest', [1, 0])
def test_user_req_u1_with_gains_in_the_room(interest):
    """U1, a room where something was gained first: into B, 0.5 of its monsters' HP removed (damage +0.5), hurt
    (-1.5), the hour glass: the rollback = the hurt refunded - the room's gains (each by its interest), here
    1.5 / G - 0.5 / G^2 (plan I), 1.5 - 0.5 (plan P); with more gained than lost it is negative."""
    for dmg, hurt in ((0.5, 2.0), (1.0, 0.5)):
        b = Book(rewind=(interest, 0, 4.0))
        b.rec(first=1)
        b.rec(rewind=CK)
        b.rec(damage=dmg)                        # +dmg
        b.rec(hurt=hurt, units=6 - hurt)         # -0.75 hurt
        b.rec(rewind=1, units=6)
        back = b.rewards[-1] + 0.002
        cd, ch = (G ** -2, G ** -1) if interest else (1.0, 1.0)
        assert back == pytest.approx(0.75 * hurt * ch - dmg * cd, abs=1e-6), (dmg, hurt, back)
        assert float(b.book.hurt[0]) == 0.0 and float(b.book.damage[0]) == 0.0
    # (the second case: 1.0 gained, 0.375 lost -- the rollback is negative: what the user is asked to confirm)


def test_user_req_u2_nohit_back_after_a_rewound_hurt(monkeypatch):
    """U2 (worker): into B, hurt there (B loses its no-hit standing), the hour glass, into B again, cleared without a
    hurt: EV_NOHIT on that clear -- the rewind restored B's standing with the hearts."""
    monkeypatch.setenv('ISAAC_RL_NOHIT', '1')
    w = Worker()
    A, B = 84, 85
    w.rec(A, True, rewinds=0)
    w.rec(B, False, monsters_hp=30.0)
    w.rec(B, False, monsters_hp=20.0, damage_taken=2.0)          # hurt in B
    w.rec(A, True, rewinds=1, damage_taken=2.0)                  # rewound: the game's hearts back
    w.rec(B, False, monsters_hp=30.0, damage_taken=2.0)
    w.rec(B, False, monsters_hp=10.0, damage_taken=2.0)
    ev = w.rec(B, True, monsters_hp=0.0, damage_taken=2.0)[0]
    assert ev & TO.EV_CLEARED and ev & TO.EV_NOHIT


def test_user_req_u2_variants(monkeypatch):
    """U2 (a): B cleared without a hurt (EV_NOHIT), rewound, cleared again: EV_NOHIT again (the book cancelled the
    first one), once -- a third clear in the same timeline gives none. (b) the hurt in an earlier visit of B, before
    the checkpoint (out and in again): the rewind does not restore the standing (the game's hearts are not restored
    either)."""
    monkeypatch.setenv('ISAAC_RL_NOHIT', '1')
    A, B = 84, 85
    w = Worker()
    w.rec(A, True, rewinds=0)
    w.rec(B, False, monsters_hp=30.0)
    w.rec(B, False, monsters_hp=10.0)
    assert w.rec(B, True, monsters_hp=0.0)[0] & TO.EV_NOHIT        # the first no-hit clear
    w.rec(A, True, rewinds=1)
    w.rec(B, False, monsters_hp=30.0)
    w.rec(B, False, monsters_hp=10.0)
    assert w.rec(B, True, monsters_hp=0.0)[0] & TO.EV_NOHIT        # again, in the new timeline
    w.rec(B, False, monsters_hp=5.0)
    assert not w.rec(B, True, monsters_hp=0.0)[0] & TO.EV_NOHIT    # (once per room and timeline)
    # (b)
    w = Worker()
    w.rec(A, True, rewinds=0)
    w.rec(B, False, monsters_hp=30.0)
    w.rec(B, False, monsters_hp=20.0, damage_taken=2.0)            # hurt in B, first visit
    w.rec(A, True, damage_taken=2.0)                               # out (a checkpoint)
    w.rec(B, False, monsters_hp=20.0, damage_taken=2.0)            # in again (a checkpoint: B already hurt in)
    w.rec(A, True, rewinds=1, damage_taken=2.0)                    # rewound to that entry
    w.rec(B, False, monsters_hp=20.0, damage_taken=2.0)
    assert not w.rec(B, True, monsters_hp=0.0, damage_taken=2.0)[0] & TO.EV_NOHIT


def test_user_req_u2_book_cancels_and_pays_nohit_once_per_timeline():
    b = Book(rewind=(1, 0, 4.0), nohit=0.5)
    b.rec(first=1)
    b.rec(rewind=CK)
    b.rec(events=EV_CLEARED | TO.EV_NOHIT)    # +1 room +0.5 nohit
    assert b.rewards[-1] == pytest.approx(1.5 - 0.002) and int(b.book.counts[0, 2]) == 1
    b.rec(rewind=1)
    assert int(b.book.counts[0, 2]) == 0
    b.rec(rewind=CK)
    b.rec(events=EV_CLEARED | TO.EV_NOHIT)
    assert b.rewards[-1] == pytest.approx(1.5 - 0.002)
    fin = b.rec(done=2).book.finished[-1]
    assert fin['nohit_rooms'] == 1 and fin['rooms'] == 1


# ---- Codex round 20 ----

def _gap(n, cmax):
    """checkpoint, a room cleared (+1), n - 2 decisions of time, the rewind: the n decisions' discounted return from
    the checkpoint (plan I, time kept)."""
    b = Book(rewind=(1, 0, cmax), cap=n + 16)
    b.rec(first=1)
    b.rec(rewind=CK)
    s = len(b.rewards)
    b.rec(events=EV_CLEARED)
    for _ in range(n - 2):
        b.rec()
    b.rec(rewind=1)
    return sum(G ** j * x for j, x in enumerate(b.rewards[s:]))


def test_cmax_residue_is_the_known_one():
    """(Codex round 20, 1; the coordinator: accepted, written down) with rewind_cmax 4 a rollback older than ~277
    decisions (gamma^-277 = 4) is capped: the discounted return of the stretch is its time terms plus a bounded
    residue -- 600 decisions after a +1 the stretch still nets +0.421 (Codex's +0.4220). Pinned: a change here is a
    change of the reward."""
    def time_part(n):
        return -0.002 * sum(G ** j for j in range(n))
    assert _gap(277, 4.0) == pytest.approx(time_part(277), abs=1e-5)          # (not capped yet)
    for n, net in ((300, -0.2047), (600, 0.4211), (900, 0.5602)):
        assert _gap(n, 4.0) == pytest.approx(net, abs=2e-4), n
        assert _gap(n, 1e9) == pytest.approx(time_part(n), abs=1e-5)         # (uncapped: the time terms only)


@pytest.mark.parametrize('done,boot,penalty', [(2, False, -2.0), (3, False, -1.0), (3, True, 0.0), (4, True, 0.0)])
def test_an_end_on_the_rewind_record_keeps_its_own_reward(done, boot, penalty, monkeypatch):
    """(Codex round 20, 3) hurt (-1.5), then a record that both rewinds and ends: death (-2), a timeout not
    bootstrapped (-1), a truncation (the cap with --cap-truncate: no penalty, bootstrapped) and a branch's end
    (DONE_BRANCH: none) -- the rollback and the end's own reward both paid."""
    if done == 3 and boot:
        monkeypatch.setattr(T, 'CAP_T', 5)
        monkeypatch.setattr(T, 'TRUNC', 1)
    b = Book(rewind=(1, 0, 4.0))
    b.rec(first=1)
    b.rec(rewind=CK)
    b.rec(hurt=2.0, units=4)
    k = int(b.roll.open[0])
    b.rec(rewind=1, done=done, units=6, t=10)
    want = -0.002 + 1.5 * G ** -1 + penalty
    assert b.rewards[-1] == pytest.approx(want, abs=1e-6)
    assert bool(b.roll.terminal[0, k]) == (done in (2, 3) and not boot)
    if done != 4:
        fin = b.book.finished[-1]
        assert sum(fin['r'].values()) == pytest.approx(fin['ret'], abs=1e-3)


def test_a_rewind_with_no_open_decision_is_counted_lost():
    """(Codex round 20, 2) the decisions after a clear dropped at the worker's capacity (store: open -1): the rewind's
    record closes nothing -- the rollback is lost, counted (rewind_lost), the state goes back all the same."""
    b = Book(rewind=(1, 0, 4.0))
    b.rec(first=1)
    b.rec(rewind=CK)
    b.rec(events=EV_CLEARED)
    assert b.book.rooms[0] == 1
    b.roll.open[0] = -1                  # (as store leaves it when the record's decision was dropped)
    go = b.book.arrive(b.roll, _row(rewind=1), np.array([0]), b.pending)
    assert go[0] and b.book.rewind_lost == 1 and b.book.rooms[0] == 0 and b.book.rewinds[0] == 1


def test_the_global_heads_rollback():
    """(Codex round 20, 4a) value_global's returns use gamma_global 0.999: its reward at a rewind is its own rollback
    (Rollout.reward_gd), so from the checkpoint its discounted return of the stretch is the time terms too -- a hurt
    undone 200 decisions later nets about +1.85 under 0.999 with the main head's rollback alone (Codex)."""
    GG = 0.999
    n = 200
    b = Book(rewind=(1, 0, 4.0), gamma_g=GG, cap=n + 16)
    b.rec(first=1)
    b.rec(rewind=CK)
    s = len(b.rewards)
    b.rec(hurt=2.0, units=4)
    for _ in range(n - 2):
        b.rec(units=4)
    b.rec(rewind=1, units=6)
    rm = np.array(b.rewards[s:])
    rg = rm + b.roll.reward_gd[0, s:s + n]
    tg = -0.002 * sum(GG ** j for j in range(n))
    assert sum(GG ** j * x for j, x in enumerate(rg)) == pytest.approx(tg, abs=1e-5)
    assert sum(G ** j * x for j, x in enumerate(rm)) == pytest.approx(-0.002 * sum(G ** j for j in range(n)), abs=1e-5)
    assert sum(GG ** j * x for j, x in enumerate(rm)) - tg > 1.5   # (the main head's rollback under 0.999)
    assert T.reward_g(b.roll) is not None
    assert np.count_nonzero(b.roll.reward_gd) == 1


def test_a_rewind_past_the_fork_point_drops_the_score():
    """(Codex round 20, 4b) a branch / continuation whose first rewind comes before its own first checkpoint goes
    back past its start: flagged (the scorers drop the point, cf_rewind_cross); after a checkpoint it is its own."""
    led = RewindLedger((1, 1, 0, 4.0), G, RW)
    led.step(1.0, 0)
    led.step(-0.002, 1)
    assert led.crossed
    led = RewindLedger((1, 1, 0, 4.0), G, RW)
    led.step(1.0, CK)
    led.step(0.5, 0)
    led.step(-0.002, 1)
    assert not led.crossed
    from isaac_bridge.tok_teacher2 import taken_score
    log = types.SimpleNamespace(damage=[0.0, 1.0, 0.0, 0.0], hurt=[0.0] * 4, done=[0] * 4, events=[0] * 4,
                                rewind=[0, 0, 1, 0], value=[0.0] * 4)
    fl = {}
    taken_score(log, 0, 3, G, RW, True, rewind=(1, 1, 0, 4.0), flags=fl)
    assert fl.get('rewind_cross')


def test_rewind_off_the_encoder_is_b53f14c():
    """(Codex round 20, 5) --rewind 0: the count is not looked at -- a rising count with damage_taken up still gives
    the hurt, the room change its events, ROW rewind 0 (b53f14c's encoding)."""
    w = Worker(rewind=False)
    A, B = 84, 85
    w.rec(A, True, rewinds=0)
    ev, rb, *_ = w.rec(B, False, monsters_hp=30.0)
    assert ev & EV_NEW_ROOM and rb == 0
    ev, rb, hu, dm, _ = w.rec(A, True, rewinds=1, damage_taken=1.0)
    assert (rb, hu) == (0, 1.0)


def test_a_missing_count_is_a_protocol_error():
    """(Codex round 20, 6) --rewind 1 and no count at an episode's first record: tok_floor.RewindFatal -- a
    VisibleV3Fatal (no BridgeError / RuntimeError: the workers' retry paths do not take it), raised where the main
    episode's first record is encoded; the sampler reads a worker's fatal exit with --rewind 1 in any visible mode."""
    import inspect
    from isaac_bridge import tok_floor, tok_sampler
    from isaac_bridge.visible_producer import V3_FATAL_EXIT, VisibleV3Fatal
    assert issubclass(tok_floor.RewindFatal, VisibleV3Fatal) and not issubclass(tok_floor.RewindFatal, RuntimeError)
    src = inspect.getsource(tok_floor)
    assert "raise RewindFatal('rewind ledger on (--rewind 1)" in src
    # (Codex round 23: the worker's fatal by the exception's always_fatal -- RewindFatal, FastRowFatal, ChildFatal)
    assert tok_floor.RewindFatal.always_fatal is True
    assert "if fatal_active() or getattr(exc, 'always_fatal', False):" in src
    s = tok_sampler.TokSampler.__new__(tok_sampler.TokSampler)
    s.fatal, s._fatal_reason = None, {}
    s.cfg = types.SimpleNamespace(rewind=1, fatal_dir='')
    s.procs = [types.SimpleNamespace(exitcode=V3_FATAL_EXIT, pid=None)]
    s._note_exit(0)
    assert s.fatal is not None and ('exit ' + str(V3_FATAL_EXIT)) in s.fatal


def test_two_rewinds_shares_still_add_up():
    """(Codex round 20, later) consecutive rewinds: the share 'rewind' accumulates both remainders -- the shares add
    up to the return."""
    b = Book(rewind=(1, 0, 4.0))
    b.rec(first=1)
    b.rec(rewind=CK)
    b.rec(events=EV_CLEARED)
    b.rec(rewind=1)
    b.rec(hurt=2.0, units=4)
    b.rec(rewind=1, units=6)
    fin = b.rec(done=2).book.finished[-1]
    assert sum(fin['r'].values()) == pytest.approx(fin['ret'], abs=1e-4)


# ---- Codex round 21 (its two CPU repros as regressions; the integration checks through the real paths) ----

def _cf_setup(cross_on_replicate):
    """cf_search with stand-ins (test_cf_smoke's): the target branch beats the taken continuation; only the best
    branch's reseed replicate goes back past the point (rewind_cross) when cross_on_replicate."""
    import test_cf_smoke as CS
    from isaac_bridge import tok_cf as CF
    from isaac_bridge import tok_teacher2 as T2
    from isaac_bridge.tok_branch import parse_reward
    rw = parse_reward(CS.FLOOR_REWARD)
    log = CS.make_log(60, np.random.default_rng(3), rw)

    class S_:
        def close(self):
            pass

    def fake_restore(pt, d_, ctx):
        return S_(), 10.0

    def fake_branch(S, ctx, pt, d_, hp0, intent, policy, reseed=None, keep=True, trace=None):
        base = {'target': 9.0, 'sticky': 0.5}[intent.kind]
        return dict(score=base - (0.05 if reseed is not None else 0.0), n=30, end=T2.END_CAP, rows=[], pend=[],
                    act=[], ret=np.zeros(0), value=[], logp=[],
                    rewind_cross=bool(cross_on_replicate and reseed is not None))
    cfg = type('C', (), dict(cf_branches=4, teacher_v2_sticky=4.0, cf_sticky_eps=0.25, cf_target_seconds=20.0,
                             frames_per_decision=4, cf_margin=0.3, cf_distill_steps=8))()
    ctx = type('X', (), dict(cfg=cfg, cap=20, gamma=0.99, rw=rw, run=True, exit_end=False, boot_vg=True, gains=True,
                             stats=dict(replicates=0, rewind_cross=0), rewind=(1, 1, 0, 4.0)))()
    pt = T2.V2Point(at=20, offset=5, log=log, target=('ent', 'coin'), bank=None)
    return CF.cf_search(pt, ctx, None, np.random.default_rng(0), branch=fake_branch, restore_fn=fake_restore), ctx


def test_cf_replicate_past_the_point_drops_the_point():
    """(Codex round 21, 1) the first branches within the point, only the best one's replicate rewinds past it: the
    point is dropped (None, cf_rewind_cross) -- before, cf_search returned improving=True and it went to the
    teaching ring. Without the crossing the same point improves."""
    res, ctx = _cf_setup(False)
    assert res is not None and res['improving'] and ctx.stats['rewind_cross'] == 0
    res, ctx = _cf_setup(True)
    assert res is None and ctx.stats['rewind_cross'] == 1 and ctx.stats['replicates'] == 1


def test_teacher_replicate_past_the_point_drops_the_point(monkeypatch):
    """(Codex round 21, 1) the same in teacher v2's search_point."""
    from isaac_bridge import tok_teacher2 as T2
    import test_cf_smoke as CS
    from isaac_bridge.tok_branch import parse_reward
    rw = parse_reward(CS.FLOOR_REWARD)
    log = CS.make_log(60, np.random.default_rng(3), rw)

    class S_:
        def close(self):
            pass
    monkeypatch.setattr(T2, 'restore', lambda pt, d, ctx: (S_(), 10.0))
    monkeypatch.setattr(T2, 'candidates', lambda cfg, rng: [types.SimpleNamespace(kind='a'), types.SimpleNamespace(kind='b')])

    def pb(S, ctx, pt, d, hp0, intent, policy, reseed=None, keep=True, trace=None):
        return dict(score=9.0 if intent.kind == 'a' else 0.0, rewind_cross=reseed is not None)
    monkeypatch.setattr(T2, 'play_branch', pb)
    cfg = types.SimpleNamespace(teacher_v2_margin=0.1, teacher_v2_depths=(0,), teacher_depths=(), teacher_death_depths=())
    ctx = types.SimpleNamespace(cfg=cfg, cap=20, gamma=0.99, rw=rw, run=True, rewind=(1, 1, 0, 4.0),
                                stats=dict(replicates=0, rewind_cross=0))
    pt = T2.V2Point(at=21, offset=5, log=log, target=None, bank=None)
    pt.kind = T2.KIND_RANDOM
    assert T2.search_point(pt, ctx, None, np.random.default_rng(0)) is None
    assert ctx.stats['rewind_cross'] == 1


@pytest.mark.parametrize('done,end', [(2, -2.0), (3, -1.0), (1, 3.0)])
def test_the_scorers_keep_an_end_on_the_rewind_record(done, end):
    """(Codex round 21, 2) a continuation with its own checkpoint, a hurt, then a record that rewinds and ends (death,
    a timeout, the floor's end): its score keeps the end's reward -- before, the rewind's record paid only its time
    term and the rollback (Codex: -0.00597 for -1.98602 with a death)."""
    from isaac_bridge.tok_teacher2 import taken_score
    log = types.SimpleNamespace(damage=[0.0] * 4, hurt=[0.0, 0.0, 2.0, 0.0], done=[0, 0, 0, done], events=[0] * 4,
                                rewind=[0, CK, 0, 1], value=[0.0] * 4)
    fl = {}
    got, m = taken_score(log, 0, 10, G, RW, True, rewind=(1, 1, 0, 4.0), flags=fl)
    want = -0.002 + G * (-1.5 - 0.002) + G ** 2 * (-0.002 + 1.5 * G ** -1 + end)
    assert not fl.get('rewind_cross') and m == 3
    assert got == pytest.approx(want, abs=1e-12)
    led = RewindLedger((1, 1, 0, 4.0), G, RW)
    led.step(-0.002, CK)
    led.step(-1.502, 0)
    assert led.step(-0.002 + end, 1, end=end) == pytest.approx(-0.002 + 1.5 * G ** -1 + end, abs=1e-12)


def test_a_lost_rollback_through_store_at_capacity_two():
    """(Codex round 21, the integration) the real store with capacity 2: first record, the checkpoint, a clear
    (+0.998 stored), its decision dropped at the cap; the rewind's record closes nothing -- rewind_lost 1, the state
    back, the lost amount its own share (rewind_lost) so the shares add up to ret; in the next rollout (carried), a
    second rewind charges only its time term (the dropped one is not taken again)."""
    nh = 3

    def answer(roll, rows, go):
        a = np.flatnonzero(go)
        if len(a):
            T.store(roll, a, rows[a], np.zeros((len(a), nh), np.int64), np.zeros((len(a), nh), np.int64),
                    np.zeros(len(a), np.float32), np.zeros(len(a), np.float32), 0)

    rw = dict(RW)
    roll = T.Rollout(1, 2, close=True)
    book = T.Episodes(1, rw, True, run=True, items=True)
    book.set_rewind(1, 0, 4.0, G, 64)
    pend = np.zeros((1, nh), np.int64)
    paid = []

    def rec(roll, **kw):
        k = int(roll.open[0])
        rows = _row(**kw)
        go = book.arrive(roll, rows, np.array([0]), pend)
        if k >= 0:
            paid.append(float(roll.reward[0, k]))
        answer(roll, rows, go)
    rec(roll, first=1)
    rec(roll, rewind=CK)
    rec(roll, events=EV_CLEARED)
    assert getattr(roll, 'dropped', 0) == 1 and roll.open[0] == -1 and paid[-1] == pytest.approx(0.998)
    rec(roll, rewind=1)
    assert book.rewind_lost == 1 and book.rooms[0] == 0 and book.rewinds[0] == 1 and int(book.led_n[0]) == 0
    roll2 = T.Rollout(1, 8, close=True)
    roll2.carry(roll)
    rec(roll2, rewind=CK)
    rec(roll2, rewind=1)
    assert paid[-1] == pytest.approx(-0.002, abs=1e-7)
    rec(roll2, done=2)
    fin = book.finished[-1]
    assert fin['ret'] == pytest.approx(sum(paid), abs=1e-5)
    assert sum(fin['r'].values()) == pytest.approx(fin['ret'], abs=1e-3)
    # (the share 'rewind_lost': the rollback that was never paid -- what ret still holds of the rewound clear)
    assert fin['r'].get('rewind_lost', 0.0) == pytest.approx(1.0 * G ** -1, abs=1e-3)


def test_the_global_targets_take_the_heads_own_rollback():
    """(Codex round 21, the integration) the learner's path (train_tok.global_targets, both its value sources): with
    values 0 and lam 1 the target of the first decision after the checkpoint is the stretch's gamma_global return --
    its time terms only; gae without reward_gd (the main head's rollback alone) is off by more than 1.5."""
    import inspect
    GG = 0.999
    n = 200
    b = Book(rewind=(1, 0, 4.0), gamma_g=GG, cap=n + 16)
    b.rec(first=1)
    b.rec(rewind=CK)
    s = len(b.rewards)
    b.rec(hurt=2.0, units=4)
    for _ in range(n - 2):
        b.rec(units=4)
    b.rec(rewind=1, units=6)
    zero = np.zeros_like(b.roll.value)
    _, ret_g, _ = T.global_targets(b.roll, GG, 1.0, zero)
    tg = -0.002 * sum(GG ** j for j in range(n))
    assert ret_g[0, s] == pytest.approx(tg, abs=1e-5)
    _, ret_plain, _ = T.gae(b.roll, GG, 1.0, zero)
    assert ret_plain[0, s] - tg > 1.5
    src = inspect.getsource(T)
    assert src.count('global_targets(rl, args.gamma_global, args.lam, vg_now)') == 2


@pytest.mark.parametrize('done,trunc,penalty,terminal', [(1, 0, 3.0, True), (3, 2, 0.0, False), (3, 1, 0.0, False)])
def test_more_ends_on_the_rewind_record(done, trunc, penalty, terminal, monkeypatch):
    """(Codex round 21) the floor's end (+exit), the stall's end truncated (--stall-truncate: t < the cap) and the
    cap's: the rollback and the end's own reward; a truncated end is answered (its record evaluated: the decision
    before bootstraps from it, store marks it boot)."""
    monkeypatch.setattr(T, 'CAP_T', 50)
    monkeypatch.setattr(T, 'TRUNC', trunc)
    b = Book(rewind=(1, 0, 4.0))
    b.rec(first=1)
    b.rec(rewind=CK)
    b.rec(hurt=2.0, units=4)
    k = int(b.roll.open[0])
    go = b.book.arrive(b.roll, _row(rewind=1, done=done, units=6, t=60 if trunc == 1 else 10), np.array([0]), b.pending)
    assert float(b.roll.reward[0, k]) == pytest.approx(-0.002 + 1.5 * G ** -1 + penalty, abs=1e-6)
    assert bool(b.roll.terminal[0, k]) == terminal
    assert bool(go[0]) == (not terminal)
    assert T.is_trunc(_row(done=done, t=60 if trunc == 1 else 10))[0] == (not terminal)
