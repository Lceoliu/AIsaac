"""CPU smoke test of B39 (2026-10-18): no-active starts (train_tok --no-active-prob p, eval_tok --no-active 1;
tok_pickup.NO_ACTIVE_LUA / parse_no_active, tok_floor.no_active_start and its call site, the starting room's restore
point of a changed start, the trainer's progress / episodes bookkeeping).

No game, no GPU needed. Plain script:
  cd rl/bridge/python && python tests/test_no_active_smoke.py
Checks:
  (a) NO_ACTIVE_LUA's text (balanced blocks, GetActiveItem / RemoveCollectible / the answer) and parse_no_active on
      good and bad answers (an active still held is a ValueError);
  (b) tok_floor.no_active_start on a fake episode clone: the snippet is sent once, no_actives counts every start,
      no_active_removed only those that held an item, a failed removal raises and counts nothing;
  (c) TokSamplerConfig defaults (off), the STATS columns (A28's np_runs / np_removed stay last);
  (d) tok_floor's call site: training draws from an rng of its own and only at floor starts (not archive starts),
      evaluation removes at every start; the removal before the start build, the build before the first observation;
      a changed start (no-active, build, stat augmentation) parks the episode as the starting room's restore point
      (reseed None) when searches are on, an unchanged one keeps src with the episode's reseed;
  (e) train_tok: --no-active-prob reaches TokSamplerConfig (main() up to the config, captured; nothing launched),
      default 0.0, refused outside floor / run modes and outside [0, 1];
  (f) eval_tok: --no-active 1 reaches TokSamplerConfig(no_active=True), off by default, refused in room mode;
  (g) train_tok.Episodes: active0 (has_active at the first record) in the finished episodes when on, absent when off.
"""
import contextlib
import io
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parent
sys.path.insert(0, str(PY))
try:   # tok_obs imports transformer_obs, which imports gymnasium at module level (not used here)
    import gymnasium  # noqa: F401
except ImportError:
    g, s = types.ModuleType('gymnasium'), types.ModuleType('gymnasium.spaces')

    class _Any:
        def __init__(self, *a, **k):
            pass
    for nm in ('Box', 'Dict', 'Discrete', 'MultiDiscrete', 'Tuple', 'MultiBinary'):
        setattr(s, nm, _Any)
    g.spaces, g.Env, g.Wrapper = s, object, object
    sys.modules['gymnasium'], sys.modules['gymnasium.spaces'] = g, s

import numpy as np   # noqa: E402

from isaac_bridge import tok_floor as TF   # noqa: E402
from isaac_bridge import tok_pickup as PK   # noqa: E402
from isaac_bridge.tok_obs import ROW   # noqa: E402
from isaac_bridge.tok_sampler import ST, STATS, TokSamplerConfig   # noqa: E402

FAIL = []
GROUPS = str(PY.parent / 'abplus' / 'catalog' / 'scaling2_groups.json')


def check(name, ok, detail=''):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f'  ({detail})' if detail else ''), flush=True)
    if not ok:
        FAIL.append(name)


class FakeEpisode:
    def __init__(self, answer):
        self.answer, self.sent = answer, []

    def lua(self, code):
        self.sent.append(code)
        return self.answer


class Captured(Exception):
    pass


def capture_config(module, run):
    """Run `run()` with module.TokSamplerConfig replaced by a recorder that raises Captured: the kwargs."""
    got = {}
    real = module.TokSamplerConfig

    def fake(**kw):
        got.update(kw)
        raise Captured()
    module.TokSamplerConfig = fake
    try:
        run()
    except Captured:
        pass
    finally:
        module.TokSamplerConfig = real
    return got


def refused(run):
    try:
        with contextlib.redirect_stderr(io.StringIO()):   # (argparse's usage text)
            run()
    except SystemExit as exc:
        return exc.code not in (0, None)
    except Captured:
        return False
    return False


def main():
    # (a) the snippet and its answer
    lua = PK.NO_ACTIVE_LUA
    check('(a) NO_ACTIVE_LUA: GetActiveItem, RemoveCollectible(a) when held, the three-part answer, blocks balanced',
          'Isaac.GetPlayer(0)' in lua and 'p:GetActiveItem()' in lua and 'p:RemoveCollectible(a)' in lua
          and 'if a ~= 0 then' in lua and 'p:GetCollectibleNum(a)' in lua and lua.count('"|"') == 2
          and len(re.findall(r'\b(?:do|then|function)\b', lua)) == len(re.findall(r'\bend\b', lua)) == 1)
    ok_parse = PK.parse_no_active('105|0|0') == (105, 0, 0) and PK.parse_no_active('0|0|0') == (0, 0, 0) \
        and PK.parse_no_active(' 105|0|0\n') == (105, 0, 0)
    bad = []
    for ans in ('105|105|1', '105|0|1', '', None, '105|0', 'x|0|0'):
        try:
            PK.parse_no_active(ans)
            bad.append(ans)
        except ValueError:
            pass
    check('(a) parse_no_active: good answers parsed, held / malformed answers refused', ok_parse and not bad, str(bad))

    # (b) the worker's helper and its counters
    mine = np.zeros(len(STATS))
    ep = FakeEpisode('105|0|0')
    r1 = TF.no_active_start(ep, mine)
    ep0 = FakeEpisode('0|0|0')   # a character without an active item
    r0 = TF.no_active_start(ep0, mine)
    check('(b) no_active_start: the snippet sent once per start, counters no_actives 2 / no_active_removed 1',
          r1 == 105 and r0 == 0 and ep.sent == [PK.NO_ACTIVE_LUA] and ep0.sent == [PK.NO_ACTIVE_LUA]
          and mine[ST['no_actives']] == 2 and mine[ST['no_active_removed']] == 1
          and mine.sum() == 3, f"{mine[ST['no_actives']]}, {mine[ST['no_active_removed']]}")
    before = mine.copy()
    try:
        TF.no_active_start(FakeEpisode('105|105|1'), mine)
        raised = False
    except ValueError:
        raised = True
    check('(b) no_active_start: an active still held raises, nothing counted', raised and (mine == before).all())

    # (c) defaults and stats columns
    c = TokSamplerConfig()
    check('(c) TokSamplerConfig: no_active_prob 0.0, no_active False by default',
          c.no_active_prob == 0.0 and c.no_active is False)
    check('(c) STATS: no_actives / no_active_removed present, A28 np_runs / np_removed still last',
          'no_actives' in ST and 'no_active_removed' in ST and STATS[-2:] == ('np_runs', 'np_removed')
          and len(set(STATS)) == len(STATS))

    # (d) tok_floor's call site
    src = (PY / 'isaac_bridge' / 'tok_floor.py').read_text(encoding='utf-8')
    i_fork = src.find('episode = src.fork(alarm=cfg.clone_alarm, reseed=reseed, lean=True)')
    i_na = src.find('no_active_start(episode, mine)', i_fork)
    i_build = src.find('if building and not from_archive and build_rng.random() < cfg.start_build_prob:', i_fork)
    i_aug = src.find('if augmenting and not from_archive and aug_rng.random() < cfg.stat_aug_prob:', i_fork)
    i_obs = src.find('episode._send({"cmd": "obs"})', i_fork)
    i_snap = src.find('if rebase and start_mod:', i_fork)
    i_exc = src.find('except (BridgeError, OSError, RuntimeError, ValueError) as exc:', i_fork)
    check('(d) tok_floor: fork -> no-active -> build -> stat aug -> first observation -> start snapshot, in the try',
          0 < i_fork < i_na < i_build < i_aug < i_obs < i_snap < i_exc)
    check('(d) tok_floor: training only at floor starts with its own rng (seed tag 41), evaluation at every start',
          "na_train = float(getattr(cfg, 'no_active_prob', 0.0) or 0.0) > 0 and not evaluation" in src
          and "na_eval = bool(getattr(cfg, 'no_active', False)) and evaluation" in src
          and 'na_rng = np.random.default_rng([cfg.seed, index, 41])' in src
          and src.count('default_rng([cfg.seed, index, 41])') == 1
          and 'if na_eval or (na_train and not from_archive and na_rng.random() < cfg.no_active_prob):' in src)
    seg = src[i_fork:i_exc]
    check('(d) tok_floor: every start change sets start_mod (no-active, build, stat aug)', seg.count('start_mod = True') == 3)
    check('(d) tok_floor: a changed start = the episode parked as the starting room\'s base (reseed None); else src',
          'base, base_reseed, base_t = start_base, None, 0' in src
          and 'base, base_reseed, base_t = src.hold(), reseed, 0' in src
          and "start_base = Parked(episode.fork(tag='snap', alarm=0)" in src
          and src.count('start_base.release()') == 1)

    # (e) train_tok plumbing
    import train_tok as T
    tmp = Path(tempfile.mkdtemp(prefix='b39-'))
    try:
        base = ['--groups-file', GROUPS, '--mode', 'run', '--items', '--workers', '2']
        got = capture_config(T, lambda: T.main(base + ['--no-active-prob', '0.5', '--out', str(tmp / 'a')]))
        got0 = capture_config(T, lambda: T.main(base + ['--out', str(tmp / 'b')]))
        check('(e) train_tok --no-active-prob 0.5 -> TokSamplerConfig(no_active_prob=0.5); default 0.0',
              got.get('no_active_prob') == 0.5 and got0.get('no_active_prob') == 0.0 and 'no_active' not in got,
              f"{got.get('no_active_prob')}, {got0.get('no_active_prob')}")
        cfg = TokSamplerConfig(**got)
        check('(e) the captured kwargs build a TokSamplerConfig', cfg.no_active_prob == 0.5 and cfg.no_active is False)
        r_room = refused(lambda: T.main(['--groups-file', GROUPS, '--mode', 'room', '--workers', '2',
                                         '--no-active-prob', '0.5', '--out', str(tmp / 'c')]))
        r_range = refused(lambda: T.main(base + ['--no-active-prob', '1.5', '--out', str(tmp / 'd')]))
        check('(e) train_tok refuses --no-active-prob in room mode and outside [0, 1]', r_room and r_range)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # (f) eval_tok plumbing
    import eval_tok as E
    tmp = Path(tempfile.mkdtemp(prefix='b39e-'))
    argv0 = sys.argv
    try:
        def ev(extra, out, mode='run'):
            def run():
                sys.argv = ['eval_tok.py', '--random', '--groups-file', GROUPS, '--mode', mode, '--items',
                            '--workers', '2', '--seeds', '2147490000:4', '--out', str(tmp / out)] + extra
                E.main()
            return run
        got = capture_config(E, ev(['--no-active', '1'], 'a'))
        got0 = capture_config(E, ev([], 'b'))
        check('(f) eval_tok --no-active 1 -> TokSamplerConfig(no_active=True); off by default',
              got.get('no_active') is True and got0.get('no_active') is False and 'no_active_prob' not in got)
        check('(f) eval_tok refuses --no-active in room mode', refused(ev(['--no-active', '1'], 'c', mode='room')))
        ev_src = (PY / 'eval_tok.py').read_text(encoding='utf-8')
        check('(f) eval_tok records no_active and the counters in summary.json',
              "info['no_active'] = True" in ev_src and "summary['no_active_starts']" in ev_src
              and "summary['no_active_removed']" in ev_src)
    finally:
        sys.argv = argv0
        shutil.rmtree(tmp, ignore_errors=True)

    # (g) the trainer's episode bookkeeping
    rw = dict(room=0.0, explore=0.0, boss=0.0, exit=1.0, heal=0.0, resource=0.0, damage=0.0, hurt=0.0, time=0.0,
              clear=1.0, death=1.0, timeout=1.0)

    def episodes(active0_on):
        book = T.Episodes(2, rw, True, run=True, items=True)
        if active0_on:
            book.active0 = {}
        roll = types.SimpleNamespace(open=np.full(2, -1), reward=np.zeros((2, 4)), terminal=np.zeros((2, 4), bool))
        pending = np.zeros((2, 5), np.int64)
        r = np.zeros(2, ROW)
        r['first'] = 1
        r['player'][0, 21] = 1.0   # worker 0 holds an active item (the D6), worker 1 does not
        r['player'][1, 21] = 0.0
        book.arrive(roll, r, np.arange(2), pending)
        r2 = np.zeros(2, ROW)
        r2['done'] = [1, 3]
        r2['t'] = 5
        book.arrive(roll, r2, np.arange(2), pending)
        return book.finished
    fin_on, fin_off = episodes(True), episodes(False)
    check('(g) Episodes: active0 1 / 0 by the first record\'s has_active when on; absent when off',
          [e.get('active0') for e in fin_on] == [1, 0] and all('active0' not in e for e in fin_off),
          str([e.get('active0') for e in fin_on]))
    tsrc = (PY / 'train_tok.py').read_text(encoding='utf-8')
    check('(g) progress.csv: no_actives / no_active_removed / noact/* / act/* written when on',
          "row['no_actives'] = float(stats[:, ST['no_actives']].sum())" in tsrc
          and "for k_a, a_ in (('noact', 0), ('act', 1)):" in tsrc and 'if cfg.no_active_prob > 0:' in tsrc)

    print('FAILED: ' + ', '.join(FAIL) if FAIL else 'all ok', flush=True)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
