"""A8 Go-Explore (user request 2026-09-30): action codes against the bridge's play decoding, cells, the exploration
policy's aim, the archive (new / better / terminal candidates, the replayed prefix, seen counts, failed returns and
retirement, selection), AbplusTransformerEnv.play's frame budget and bookkeeping, and the driver's task flow down to the
written room. No game instance needed."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaac_bridge.abplus import action_code, joint_code
from isaac_bridge.abplus_goexplore import (WIN, Archive, Explorer, ExploreEnv, GxConfig, aim, better, describe,
                                           pack, room_context, unpack)


def lua_decode(code):
    """abp_bridge.lua play_next: move = code % 9, shoot = (code // 9) % 5, bomb = (code // 45) % 2, item = (code // 90) % 2."""
    return code % 9, (code // 9) % 5, (code // 45) % 2, (code // 90) % 2


def obs(x=100.0, y=200.0, hearts=6, bombs=1, blocking_hp=20.0, count=2, dealt=0.0, clear=False, dead=False,
        entities=(), frames=0):
    return dict(logic_frames=frames,
                players=[dict(pos=[x, y], hearts=hearts, soul=0, eternal=0, bone=0, bombs=bombs, dead=dead,
                              active_ready=False)],
                combat=dict(blocking_hp=blocking_hp, blocking_count=count, monster_damage=dealt),
                room=dict(top_left=[60.0, 140.0], bottom_right=[580.0, 420.0], clear=clear, room_idx=5),
                entities=list(entities))


class CodesTest(unittest.TestCase):
    def test_bridge_decodes_what_the_client_encodes(self):
        for move in range(9):
            for shoot in range(5):
                for bomb in (0, 1):
                    for item in (0, 1):
                        self.assertEqual(lua_decode(action_code(move, shoot, bomb, item)), (move, shoot, bomb, item))

    def test_joint_and_storage_codes(self):
        for joint in range(45):
            for bomb in (0, 1):
                self.assertEqual(unpack(pack(joint, bomb)), (joint, bomb, 0))
                move, shoot = divmod(joint, 5)
                self.assertEqual(lua_decode(joint_code(joint, bomb)), (move, shoot, bomb, 0))
        self.assertLess(pack(44, 1, 1), 256)   # one byte per action


class CellTest(unittest.TestCase):
    def setUp(self):
        self.cfg = GxConfig(cell_px=80, hp_buckets=10)
        self.ctx = room_context(obs())

    def test_key(self):
        key, data = describe(obs(x=220, y=200), self.ctx, self.cfg)
        self.assertEqual(key, (2, 0, 6, 1, 2, 10))       # (220 - 60) // 80 = 2, (200 - 140) // 80 = 0
        self.assertEqual(data['progress'], 0.0)
        self.assertEqual(describe(obs(blocking_hp=18.0), self.ctx, self.cfg)[0][5], 9)   # (0.8, 0.9] of the start
        self.assertEqual(describe(obs(blocking_hp=2.0), self.ctx, self.cfg)[0][5], 1)
        self.assertEqual(describe(obs(blocking_hp=0.0, count=0), self.ctx, self.cfg)[0][5], 0)
        self.assertEqual(describe(obs(blocking_hp=60.0, count=5), self.ctx, self.cfg)[0][5], 20)   # spawns: capped
        self.assertEqual(describe(obs(bombs=7), self.ctx, self.cfg)[0][3], 2)
        self.assertAlmostEqual(describe(obs(blocking_hp=5.0), self.ctx, self.cfg)[1]['progress'], 0.75)

    def test_win_is_one_cell_per_health(self):
        a = describe(obs(x=100, clear=True, blocking_hp=0, count=0, hearts=4), self.ctx, self.cfg)[0]
        b = describe(obs(x=400, clear=True, blocking_hp=0, count=0, hearts=4), self.ctx, self.cfg)[0]
        self.assertEqual(a, (WIN, 4))
        self.assertEqual(a, b)

    def test_aim(self):
        e = lambda x, y, blocking=True: dict(pos=[x, y], blocking=blocking)
        self.assertEqual(aim(obs(entities=[e(300, 210)])), 2)
        self.assertEqual(aim(obs(entities=[e(0, 190)])), 4)
        self.assertEqual(aim(obs(entities=[e(110, 400)])), 3)
        self.assertEqual(aim(obs(entities=[e(90, 20)])), 1)
        self.assertEqual(aim(obs(entities=[e(300, 200), e(100, 230)])), 3)    # the nearest
        self.assertIsNone(aim(obs(entities=[e(300, 200, False)])))

    def test_explorer_is_sticky_and_masks_the_bomb(self):
        cfg = GxConfig(repeat_prob=1.0, bomb_prob=1.0)
        ex = Explorer(cfg, np.random.default_rng(1))
        first = ex.act(obs(bombs=0))
        self.assertEqual(first[1], 0)
        for _ in range(20):
            a = ex.act(obs(bombs=1))
            self.assertEqual(a[0], first[0])
            self.assertEqual(a[1], 1)


def root(cfg, key=(0, 0, 6, 1, 2, 10)):
    return dict(key=key, data=dict(health=6, bombs=1, alive=2, blocking_hp=20.0, progress=0.0, dealt=0.0, hurt=0.0,
                                   pos=[0, 0]),
                digest='d-root', hp0=20.0, health0=6, room=dict(name='r'))


def cand(key, n, frames, dealt=0.0, health=6, progress=0.0, outcome='running', digest=None, hurt=0.0):
    return dict(key=key, n=n, frames=frames, outcome=outcome, digest=digest or f'd{key}{frames}',
                data=dict(health=health, bombs=1, alive=2, blocking_hp=20.0, progress=progress, dealt=dealt,
                          hurt=hurt, pos=[0, 0]))


def result(cands, explore, seen=None, ends=None):
    return dict(cands=cands, explore=bytes(explore), seen=seen or {}, ends=ends or {})


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.cfg = GxConfig(max_fails=2)
        self.a = Archive(7, root(self.cfg), self.cfg)
        self.root_key = self.a.cells[0].key

    def test_new_better_terminal(self):
        a = self.a
        new, imp = a.merge(self.root_key, b'', result([cand((1,), 2, 8), cand((2,), 4, 16),
                                                       cand(('x',), 5, 20, outcome='death')], [1, 2, 3, 4, 5]), 1)
        self.assertEqual((new, imp), (2, 0))
        self.assertNotIn(('x',), a.index)
        c1 = a.cells[a.index[(1,)]]
        self.assertEqual((c1.actions, c1.frames), (bytes([1, 2]), 8))
        # from cell (1,) (prefix [1, 2]) a faster way to (2,) and a slower one to (1,)
        new, imp = a.merge((1,), bytes([1, 2]), result([cand((2,), 1, 12), cand((1,), 3, 20)], [9, 9, 9]), 2)
        self.assertEqual((new, imp), (0, 1))
        c2 = a.cells[a.index[(2,)]]
        self.assertEqual((c2.actions, c2.frames, c2.improvements), (bytes([1, 2, 9]), 12, 1))
        self.assertEqual(a.cells[a.index[(1,)]].frames, 8)
        # a tie in frames: more monster damage wins; less damage taken beats fewer frames
        self.assertTrue(better((0.0, 12, -1.0), c2))
        self.assertFalse(better((0.0, 12, 0.0), c2))
        a.merge((1,), bytes([1, 2]), result([cand((4,), 1, 10, hurt=1.0)], [9]), 3)
        self.assertEqual(a.cells[a.index[(4,)]].hurt, 1.0)
        new, imp = a.merge(self.root_key, b'', result([cand((4,), 3, 30)], [5, 5, 5]), 4)
        self.assertEqual((new, imp, a.cells[a.index[(4,)]].frames), (0, 1, 30))
        self.assertEqual([k for k, _ in a.log], [self.root_key, (1,), (2,), (2,), (4,), (4,)])

    def test_the_replayed_prefix_is_used(self):
        a = self.a
        a.merge(self.root_key, b'', result([cand((1,), 3, 12)], [1, 1, 1]), 1)
        # a task left for (1,) with prefix [1, 1, 1]; meanwhile (1,) got a faster trajectory [5]
        a.merge(self.root_key, b'', result([cand((1,), 1, 4)], [5]), 2)
        a.merge((1,), bytes([1, 1, 1]), result([cand((3,), 2, 20)], [7, 8]), 3)
        self.assertEqual(a.cells[a.index[(3,)]].actions, bytes([1, 1, 1, 7, 8]))

    def test_seen_since_new_and_selection(self):
        a = self.a
        rng = np.random.default_rng(0)
        self.assertIs(a.select(rng), a.cells[0])
        self.assertEqual((a.cells[0].chosen, a.cells[0].since_new), (1, 1))
        a.merge(self.root_key, b'', result([cand((1,), 1, 4)], [1], seen={(1,): 3, self.root_key: 1}), 1)
        self.assertEqual(a.cells[0].since_new, 0)
        self.assertEqual(a.cells[a.index[(1,)]].seen, 3)
        a.merge(self.root_key, b'', result([], [], seen={}), 2)
        self.assertEqual(a.cells[0].since_new, 0)          # not chosen in between
        a.select(np.random.default_rng(5))
        w = a.weights[:len(a.cells)]
        self.assertTrue(np.all(w > 0))

    def test_wins_are_kept_never_chosen(self):
        a = self.a
        a.merge(self.root_key, b'', result([cand((WIN, 6), 2, 30, health=6, progress=1.0, outcome=WIN)], [1, 2]), 1)
        win = a.cells[a.index[(WIN, 6)]]
        self.assertEqual(a.weights[a.index[(WIN, 6)]], 0.0)
        self.assertEqual(a.wins(), [win])
        self.assertIs(a.best(), win)
        rng = np.random.default_rng(3)
        for _ in range(50):
            self.assertIsNot(a.select(rng), win)

    def test_failed_returns_retire(self):
        a = self.a
        a.merge(self.root_key, b'', result([cand((1,), 1, 4)], [1]), 1)
        i = a.index[(1,)]
        w0 = a.weights[i]
        a.failed((1,))
        self.assertAlmostEqual(a.weights[i], w0 * 0.5)
        a.merge((1,), bytes([1]), result([], []), 2)        # a good return starts the count over
        self.assertEqual(a.cells[i].fails, 0)
        a.failed((1,))
        a.failed((1,), error=True)
        self.assertEqual(a.weights[i], 0.0)
        self.assertEqual((a.stats['return_fails'], a.stats['return_errors'], a.stats['retirements']), (2, 1, 1))
        self.assertEqual(a.summary()['retired'], 1)
        # a better trajectory gives the cell another chance
        a.merge(self.root_key, b'', result([cand((1,), 1, 2)], [3]), 3)
        self.assertGreater(a.weights[i], 0.0)

    def test_progress_and_health_factors(self):
        a = self.a
        a.merge(self.root_key, b'', result([cand((1,), 1, 4, progress=0.5), cand((2,), 1, 4, health=4, hurt=2.0)], [1]), 1)
        c1, c2 = a.cells[a.index[(1,)]], a.cells[a.index[(2,)]]
        base = 1 + 1 + 0.5
        self.assertAlmostEqual(a.weight(c1), base * (1 + 2.0 * 0.5))
        self.assertAlmostEqual(a.weight(c2), base * 0.5 ** 2)


class KnownTest(unittest.TestCase):
    def test_the_local_copy_keeps_the_best_rank(self):
        from isaac_bridge.abplus_goexplore import apply_delta, rank
        known = {}
        apply_delta(known, [((1,), (0.0, 20, -1.0)), ((1,), (0.0, 16, 0.0)), ((1,), (1.0, 4, 0.0)), ((2,), (0.0, 8, 0.0))])
        self.assertEqual(known, {(1,): (0.0, 16, 0.0), (2,): (0.0, 8, 0.0)})
        self.assertEqual(rank(dict(hurt=1, dealt=2.5), 12), (1.0, 12, -2.5))


class FakeBridge:
    action_repeat = 4

    def __init__(self):
        self.sent = None

    def play(self, codes, repeat=None, repeats=None, stop_clear=True):
        self.sent = (list(codes), repeats)
        frames = sum(repeats) if repeats is not None else len(codes) * self.action_repeat
        return obs(frames=100 + frames), len(codes), 'done'


class EnvPlayTest(unittest.TestCase):
    def env(self, max_frames=30):
        env = ExploreEnv(port=1, max_episode_frames=max_frames, frames_per_decision=4)
        env.bridge = FakeBridge()
        env.raw_obs, env.finished, env.elapsed_frames = obs(frames=100), False, 0
        return env

    def test_frame_budget_at_the_deadline(self):
        env = self.env(max_frames=30)
        _, played, outcome = env.play([(7, 0, 0)] * 10)
        codes, repeats = env.bridge.sent
        self.assertEqual(len(codes), 8)                   # 7 x 4 + 2 = 30 frames
        self.assertEqual(repeats, [4] * 7 + [2])
        self.assertEqual((played, outcome, env.elapsed_frames, env.finished), (8, 'time_limit', 30, True))
        self.assertEqual(codes[0], joint_code(7))

    def test_uniform_repeats_are_not_sent(self):
        env = self.env(max_frames=300)
        _, played, outcome = env.play([(3, 1, 0), (44, 0, 0)])
        self.assertEqual(env.bridge.sent, ([joint_code(3, 1), joint_code(44)], None))
        self.assertEqual((played, outcome, env.elapsed_frames, env.finished), (2, 'running', 8, False))
        self.assertEqual(env.history.previous_action.tolist(), [44, 0, 0, 1])

    def test_frame_mismatch_raises(self):
        env = self.env(max_frames=300)
        env.bridge.play = lambda codes, repeat=None, repeats=None, stop_clear=True: (obs(frames=101), len(codes), 'done')
        with self.assertRaises(RuntimeError):
            env.play([(0, 0, 0), (0, 0, 0)])


class FakeInstance:
    """run_task's view of an instance: reset, play and scripted digests."""

    def __init__(self, digests):
        self.digests = list(digests)
        self.env = SimpleNamespace(elapsed_frames=0, raw_obs=obs(), play=self.play)
        self.resets = 0

    def reset(self, seed):
        self.resets += 1
        self.env.elapsed_frames = 0
        return self.env.raw_obs, {}

    def play(self, actions):
        self.env.elapsed_frames += 4 * len(actions)
        return self.env.raw_obs, len(actions), 'running'

    def digest(self):
        return self.digests.pop(0)


class ReturnCheckTest(unittest.TestCase):
    def task(self):
        cfg = GxConfig()
        key = describe(obs(), room_context(obs()), cfg)[0]
        return cfg, dict(kind='verify', seed=1, key=key, actions=bytes([3, 4]), digest='good', frames=8,
                         outcome='running', ctx=room_context(obs()))

    def test_a_mismatch_is_repeated_once(self):
        from isaac_bridge.abplus_goexplore import run_task
        cfg, task = self.task()
        inst = FakeInstance(['bad', 'good'])
        out = run_task(inst, task, {}, cfg)
        self.assertEqual((out['status'], out['retry'], inst.resets), ('ok', dict(match=True, same=False, ended=False), 2))
        inst = FakeInstance(['bad', 'bad'])
        out = run_task(inst, task, {}, cfg)
        self.assertEqual((out['status'], out['reason'], out['retry']['same']), ('fail', 'digest', True))
        inst = FakeInstance(['bad'])
        out = run_task(inst, task, {}, GxConfig(retry_mismatch=0))
        self.assertEqual((out['status'], 'retry' in out, inst.resets), ('fail', False, 1))

    def test_a_short_or_ended_replay_fails_without_digest(self):
        from isaac_bridge.abplus_goexplore import run_task
        cfg, task = self.task()
        out = run_task(FakeInstance([]), dict(task, frames=12), {}, cfg)
        self.assertEqual((out['status'], out['reason'], out['frames']), ('fail', 'ended', 8))


class FakeQueue(list):
    def put(self, x):
        self.append(x)


class DriverFlowTest(unittest.TestCase):
    def test_room_from_root_to_written(self):
        from goexplore_abplus import Manager
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=tmp, rooms_at_once=1, rng_seed=0, workers=1, iterations=3, progress_s=1e9,
                                   name='t', port=1)
            cfg = GxConfig()
            m = Manager(args, cfg, dict(name='g'), [11])
            q = FakeQueue()
            m.task_queues[0] = q
            m.idle.add(0)
            m.activate()
            self.assertTrue(m.dispatch(0))
            self.assertFalse(m.dispatch(0))              # the root is in flight; nothing else to hand out
            t = q[-1]
            self.assertEqual(t['kind'], 'root')
            base = dict(worker=0, kind='result', seed=11, status='ok', return_frames=0, explore_frames=8,
                        return_s=0.0, explore_s=0.1, digest_s=0.01, digests=2, seconds=0.2)
            m.handle(dict(base, id=t['id'], task='root', root=root(cfg), **result([cand((1,), 2, 8)], [1, 2])))
            archive = m.rooms[11]['archive']
            self.assertEqual(len(archive.cells), 2)
            # exploration: the delta carries the archive log; the win found is verified before the room is written
            self.assertTrue(m.dispatch(0))
            t = q[-1]
            self.assertEqual(t['kind'], 'explore')
            self.assertEqual(len(t['delta']), 2)
            m.handle(dict(base, id=t['id'], task='explore', return_frames=t['frames'],
                          **result([cand((WIN, 5), 1, t['frames'] + 4, health=5, progress=1.0, outcome=WIN)], [9],
                                   ends={'win': 1})))
            self.assertTrue(m.dispatch(0))
            t = q[-1]
            self.assertEqual(len(t['delta']), 1)             # only what the worker had not seen
            m.handle(dict(base, id=t['id'], task='explore', status='fail', reason='digest'))
            self.assertTrue(m.dispatch(0))
            t = q[-1]
            m.handle(dict(base, id=t['id'], task='explore', **result([], [])))
            self.assertTrue(m.dispatch(0))
            t = q[-1]
            self.assertEqual((t['kind'], t['outcome']), ('verify', WIN))
            m.finish_rooms()
            self.assertIn(11, m.rooms)                       # the check is in flight
            m.handle(dict(base, id=t['id'], task='verify', **result([], [])))
            m.finish_rooms()
            self.assertNotIn(11, m.rooms)
            self.assertEqual(m.drops[0], [11])
            rec = json.loads((Path(tmp) / 'results.jsonl').read_text().splitlines()[0])
            self.assertEqual((rec['wins'], rec['verify']['status'], rec['return_fails']), (1, 'ok', 1))
            detail = json.loads((Path(tmp) / 'rooms' / '11.json').read_text())
            self.assertEqual(bytes.fromhex(detail['best']['actions']), t['actions'])   # the verified win
            self.assertEqual(t['actions'][-1], 9)


    def test_a_failed_final_check_moves_to_the_next_win(self):
        from goexplore_abplus import Manager
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(out=tmp, rooms_at_once=1, rng_seed=0, workers=1, iterations=1, progress_s=1e9,
                                   name='t', port=1)
            cfg = GxConfig()
            m = Manager(args, cfg, dict(name='g'), [12])
            q = FakeQueue()
            m.task_queues[0] = q
            m.idle.add(0)
            m.activate()
            m.dispatch(0)
            base = dict(worker=0, kind='result', seed=12, status='ok', return_frames=0, explore_frames=8)
            wins = [cand((WIN, 6), 1, 40, health=6, progress=1.0, outcome=WIN),
                    cand((WIN, 4), 2, 20, health=4, progress=1.0, outcome=WIN)]
            m.handle(dict(base, id=q[-1]['id'], task='root', root=root(cfg), **result(wins, [1, 2])))
            m.dispatch(0)
            t = q[-1]
            self.assertEqual(t['kind'], 'explore')
            m.handle(dict(base, id=t['id'], task='explore', return_frames=t['frames'], **result([], [])))
            m.dispatch(0)
            t = q[-1]
            self.assertEqual((t['kind'], t['key']), ('verify', (WIN, 6)))        # more health first
            m.handle(dict(base, id=t['id'], task='verify', status='fail', reason='digest'))
            m.finish_rooms()
            self.assertIn(12, m.rooms)
            m.dispatch(0)
            t = q[-1]
            self.assertEqual((t['kind'], t['key']), ('verify', (WIN, 4)))
            m.handle(dict(base, id=t['id'], task='verify', **result([], [])))
            m.finish_rooms()
            rec = json.loads((Path(tmp) / 'results.jsonl').read_text().splitlines()[0])
            self.assertEqual((rec['verify'], rec['best_verified'], rec['best_cell']['health']),
                             (dict(status='ok', attempts=2), True, 4))
            self.assertEqual([a['status'] for a in rec['verify_attempts']], ['fail', 'ok'])

if __name__ == '__main__':
    unittest.main()
