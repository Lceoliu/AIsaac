"""整局录制格式的单元测试（不启动游戏）。

python -m pytest rl/bridge/python/test_run_recording.py -q
真实录制文件的 round-trip（load_run.roundtrip）在 experiments/run-recorder/phase2 有文件时一并检查。
"""
import json
import random
from pathlib import Path

import pytest

import load_run
import replay_run
from isaac_bridge import run_recording as rr

PHASE2 = Path(__file__).resolve().parents[3] / 'experiments/run-recorder/phase2/sessions'
PHASE3 = Path(__file__).resolve().parents[3] / 'experiments/run-recorder/phase3/sessions'


@pytest.mark.parametrize('pcol', [None, 2])
def test_row_delta_roundtrip_random(pcol):
    rng = random.Random(1)
    prev_full = {}
    for frame in range(200):
        kf = frame % 50 == 0
        rows = []
        for idx in range(rng.randint(0, 8)):
            base = prev_full.get(idx)
            if base is None or rng.random() < 0.1:
                row = [idx, rng.randint(1, 1000)] + [rng.choice([None, 'Idle', rng.randint(-5000, 5000)]) for _ in range(20)]
            else:
                row = list(base)
                for c in range(2, len(row)):
                    if rng.random() < 0.3:
                        row[c] = rng.choice([None, 'Walk', rng.randint(-5000, 5000)])
            rows.append(row)
        packed = load_run.compress_rows(rows, prev_full, kf, seed_col=1, pcol=pcol)
        text = json.dumps(packed)
        back, prev_full = load_run.expand_rows(json.loads(text), {} if kf else prev_full, pcol)
        assert back == rows


def test_tick_recompress():
    state = {}
    tk = ['i0', ['u', [[0, [[0, 0, None]] * 12]]], 'i1', 'i2', 'i3', ['i4', [], [[1, 15, -1, 1]]], 'u']
    tk2 = ['i0', ['u', [[0, [[0, 0, None]] * 12]]], 'i1-3', ['i4', [], [[1, 15, -1, 1]]], 'u']
    ticks = load_run.Run.expand_ticks(1, tk2, state)
    assert [t.kind for t in ticks] == ['i0', 'u', 'i1', 'i2', 'i3', 'i4', 'u']
    assert load_run.recompress_ticks(ticks, {}) == tk2
    assert len(load_run.Run.expand_ticks(1, tk, {})) == 7


def test_validate_lines_detects_gap():
    hdr = json.dumps({'k': 'header', 'schema': rr.SCHEMA, 'meta': {'seeds': {}}})
    f1 = json.dumps({'k': 'f', 'n': 1, 'gf': 1})
    f3 = json.dumps({'k': 'f', 'n': 3, 'gf': 3})
    st = rr.validate_lines([hdr.encode(), f1.encode(), f3.encode()])
    assert st['truncated'] and st['errors']


def real_runs():
    if not PHASE2.exists():
        return []
    return sorted(PHASE2.glob('*/runs/run-*.jsonl.zst'))[:6]


@pytest.mark.parametrize('path', real_runs() or [None])
def test_real_file_roundtrip(path):
    if path is None:
        pytest.skip('没有 phase2 录制文件')
    try:
        run = load_run.Run(path)
    except KeyError:
        pytest.skip('早期格式的测试文件')
    if not run.keyframes:
        pytest.skip('早期格式的测试文件')
    assert load_run.roundtrip(path, verbose=False)['ok']


# ---------------------------------------------------------------- 第三阶段（recorder 0.2.0）
def test_repair_line():
    bad = b'{"k":"f","n":3,"g":[[55,2,1003,0x1,3,0]],"x":0x-12}'
    fixed, ok = load_run.repair_line(bad)
    assert ok and json.loads(fixed)['g'] == [[55, 2, 1003, 1, 3, 0]] and json.loads(fixed)['x'] == -12
    good = b'{"k":"f","s":"0xff"}'
    assert load_run.repair_line(good) == (good, False)
    st = rr.validate_lines([json.dumps({'k': 'header', 'schema': rr.SCHEMA, 'meta': {'seeds': {}}}).encode(),
                            b'{"k":"f","n":1,"gf":0x1}', b'{"k":"end","frames":1}'])
    assert st['repaired_lines'] == 1 and st['bad_json'] == 0 and st['frames'] == 1 and not st['errors']


def test_mt_salts_and_reseeds_match():
    rec = {'at': 'POST_PLAYER_INIT', 'salt': 7, 'seeded_with': 9, 'before': [1, 2], 'after': [624, 5]}
    late = {'at': 'POST_GAME_STARTED', 'salt': 7, 'seeded_with': 9, 'before': [40, 6], 'after': [624, 5]}
    ha = {'mt': {'available': True, 'reseeds': [rec, late]}}
    assert replay_run.mt_salts(ha) == {'POST_PLAYER_INIT': 7, 'POST_GAME_STARTED': 7}
    assert replay_run.mt_salts({'mt': {'available': False}}) == {}
    hb = {'mt': {'available': True, 'reseeds': [dict(rec, before=[99, 99]), late]}}
    assert replay_run.reseeds_match(ha, hb) is True          # 第一次 reseed 前的 MT 本来就不同
    hc = {'mt': {'available': True, 'reseeds': [rec, dict(late, before=[41, 6])]}}
    assert replay_run.reseeds_match(ha, hc) is False
    assert replay_run.reseeds_match(ha, {'mt': None}) is None


def test_turbo_env_default_off():
    """--turbo 默认关：不加参数时不给游戏进程任何 RR_MT_TURBO* 变量（录制/原速回放行为不变）。"""
    assert replay_run.turbo_env(None) == {}
    assert replay_run.turbo_env('clock') == {'RR_MT_TURBO': '1'}
    assert replay_run.turbo_env('gate') == {'RR_MT_TURBO': '1', 'RR_MT_TURBO_GPU_GATE': '1'}
    with pytest.raises(ValueError):
        replay_run.turbo_env('skip')
    with pytest.raises(ValueError):  # 加速需要 DLL
        replay_run.run_replay('unused.jsonl', 'x', '.', use_dll=False, turbo='gate')


def phase3_runs():
    if not PHASE3.exists():
        return []
    return sorted(PHASE3.glob('*/runs/run-*.jsonl.zst'))[:4] + sorted(PHASE3.glob('kA2-rec/runs/run-*.jsonl.zst'))


@pytest.mark.parametrize('path', phase3_runs() or [None])
def test_phase3_file_roundtrip(path):
    """带 mt 的帧行、IRR_OBS=keyframe（非关键帧无 e）的文件也能逐字节 round-trip。"""
    if path is None:
        pytest.skip('没有 phase3 录制文件')
    run = load_run.Run(path)
    assert load_run.roundtrip(path, verbose=False)['ok']
    if run.header.get('obs') == 'keyframe':
        fs = run.frame(run.frames)
        assert fs.entities is None or fs.keyframe
    assert run.format == 1 and run.app_names is None          # v1 文件：没有外观字段
    fs = run.frame(min(run.frames, 50))
    assert fs.player_appearance is None and fs.pa_rows is None and 'grid_spr' not in (fs.room or {})


# ---------------------------------------------------------------- v2（recorder 0.3.0，header.format = 2）：外观字段
PHASE6 = Path(__file__).resolve().parents[3] / 'experiments/run-recorder/phase6-art/sessions'
APP = ['key', 'file', 'ov_anim', 'ov_frame', 'flip', 'sx', 'sy', 'rot', 'r', 'g', 'b', 'a', 'ro', 'go', 'bo',
       'sr', 'sg', 'sb', 'sa', 'sro', 'sgo', 'sbo', 'ox', 'oy', 'pox', 'poy']
APP_FIX = ['sx', 'sy', 'rot', 'r', 'g', 'b', 'a', 'ro', 'go', 'bo', 'sr', 'sg', 'sb', 'sa', 'sro', 'sgo', 'sbo', 'ox', 'oy', 'pox', 'poy']


def test_sparse_appearance_roundtrip_random():
    """外观行的稀疏差分：关键帧/新对象完整行、只写变化的对象、消失的对象丢弃；compress_sparse -> expand_sparse 还原。"""
    rng = random.Random(3)
    prev = {}
    for frame in range(300):
        kf = frame % 60 == 0
        keys = sorted(rng.sample(range(12), rng.randint(0, 10)))
        rows, new = [], set()
        for k in keys:
            base = prev.get(k)
            if base is None or rng.random() < 0.05:
                row = [k] + [rng.choice([None, rng.randint(0, 30), rng.randint(-500, 500)]) for _ in range(25)]
                new.add(k)
            else:
                row = list(base)
                for c in range(1, len(row)):
                    if rng.random() < 0.08:
                        row[c] = rng.choice([None, rng.randint(-500, 500)])
            rows.append(row)
        packed = load_run.compress_sparse(rows, {} if kf else prev, kf, new)
        # 未变的对象不写
        assert all(isinstance(r[1], list) or len(r) == 26 for r in packed)
        back, prev = load_run.expand_sparse(json.loads(json.dumps(packed)), {} if kf else prev, keys)
        assert back == rows


def _v2_lines():
    hdr = {'k': 'header', 'schema': 'isaac-run-v1', 'recorder': '0.3.0', 'format': 2,
           'fields': {'player': ['num', 'x', 'y', 'vx', 'vy', 'anim'], 'entity': ['idx', 'type', 'variant', 'subtype', 'x', 'y', 'vx', 'vy',
                                                                        'hp', 'max_hp', 'frame', 'init_seed', 'drop_seed', 'spawner_type',
                                                                        'parent_idx', 'visible', 'coll', 'anim', 'aframe'],
                      'entity_extra': {}, 'fixed_x100': {'player': ['x', 'y', 'vx', 'vy'], 'entity': ['x', 'y', 'vx', 'vy', 'hp', 'max_hp'],
                                                         'entity_extra': {}},
                      'room_frame': ['room_frame', 'clear', 'alive_enemies', 'room_index'],
                      'position_prediction': {'player': 1, 'entity': 4},
                      'appearance': APP, 'player_appearance_extra': ['body_color', 'head_color'], 'appearance_fixed_x100': APP_FIX,
                      'appearance_string_columns': ['file', 'ov_anim'], 'grid_sprite': ['index', 'file', 'anim', 'frame'],
                      'door_sprite': ['slot', 'file', 'anim', 'frame', 'rot', 'ox', 'oy']}}
    room = {'k': 'room', 'n': 1, 'gf': 0, 'room_index': 84, 'top_left': [6000, 14000], 'bottom_right': [58000, 42000],
            'grid': [[22, 2, 0, 1, 2, 0]], 'doors': [[0, 83, 1, 1, 0, 2, 8]],
            'sd': [[0, 'gfx/grid/grid_rock.anm2'], [1, 'normal'], [2, 'gfx/grid/Door_01_NormalDoor.anm2'], [3, 'Opened']],
            'grid_spr': [[22, 0, 1, 2]], 'door_spr': [[0, 2, 3, 0, -9000, 1500, 0]]}
    pa0 = [0, 4, 5, 0, 0, 100, 100, 0, 100, 100, 100, 100, 0, 0, 0, 100, 100, 100, 100, 0, 0, 0, 0, 0, 0, 0, -1, -1]
    ea0 = [7, 6, None, -1, 1, 100, 100, 0, 100, 100, 100, 100, 0, 0, 0, None, None, None, None, None, None, None, 0, 0, 0, -2458]
    e0 = [7, 2, 0, 0, 30000, 30000, 100, 0, 0, 0, 1, 11, 11, 1, -1, 1, 4, 'RegularTear6', 0]
    f1 = {'k': 'f', 'n': 1, 'gf': 1, 't': 0, 'kf': 1, 'tk': ['u'], 'r': [1, 1, 0, 84], 'p': [[0, 32000, 38000, 0, 0, 'WalkDown']],
          'e': [e0], 'sd': [[4, 'gfx/001.000_Player.anm2'], [5, 'HeadDown'], [6, 'gfx/002.000_Tear.anm2']], 'pa': [pa0], 'ea': [ea0],
          'dg': [1, 2, 3]}
    f2 = {'k': 'f', 'n': 2, 'gf': 2, 't': 33, 'tk': ['u'], 'r': [2, 1, 0, 84], 'p': [0], 'e': [[7, [[18, 1]]]],
          'sd': [[7, 'HeadLeft']], 'pa': [[0, [[2, 2], [3, 1], [4, 1]]]], 'ea': [[7, [[25, -50]]]],
          'gs': [[22, 0, 1, 0]], 'ds': [[0, 2, 3, 1, -9000, 1500, 0]], 'dg': [1, 2, 3]}
    f3 = {'k': 'f', 'n': 3, 'gf': 3, 't': 66, 'tk': ['u'], 'r': [3, 1, 0, 84], 'p': [0], 'e': [], 'gs': [[22]], 'dg': [1, 2, 3]}
    return [json.dumps(x, separators=(',', ':'), ensure_ascii=False) for x in (hdr, room, f1, f2, f3)]


def test_v2_synthetic_file(tmp_path):
    p = tmp_path / 'run-v2.jsonl'
    p.write_text('\n'.join(_v2_lines()) + '\n', encoding='utf-8')
    run = load_run.Run(p)
    assert run.format == 2 and run.strings[6] == 'gfx/002.000_Tear.anm2'
    a = run.frame(2)
    pa = a.player_appearance[0]
    assert pa['file'] == 'gfx/001.000_Player.anm2' and pa['ov_anim'] == 'HeadLeft' and pa['ov_frame'] == 1 and pa['flip'] == 1
    assert pa['sx'] == 1.0 and pa['body_color'] == -1
    ea = a.entity_appearance[0]
    assert ea['file'] == 'gfx/002.000_Tear.anm2' and ea['ov_anim'] is None and ea['poy'] == -25.08 and ea['sr'] is None
    assert a.room['grid_spr'][22] == [22, 0, 1, 0] and a.room['door_spr'][0][3] == 1
    b = run.frame(3)
    assert b.entity_appearance == [] and 22 not in b.room['grid_spr']   # 实体消失、网格精灵消失
    assert b.player_appearance[0]['ov_anim'] == 'HeadLeft'              # 未列出 = 沿用
    seq = list(run.iter_frames())
    assert [s.pa_rows for s in seq] == [run.frame(n).pa_rows for n in (1, 2, 3)]
    res = load_run.roundtrip(p, sample_every=1, verbose=False)
    assert res['ok'], res


def test_v2_string_dict_conflict(tmp_path):
    lines = _v2_lines()
    f = json.loads(lines[3])
    f['sd'] = [[6, 'something else']]
    lines[3] = json.dumps(f, separators=(',', ':'))
    p = tmp_path / 'bad.jsonl'
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with pytest.raises(ValueError):
        load_run.Run(p)


def phase6_runs():
    if not PHASE6.exists():
        return []
    return sorted(PHASE6.glob('v2-*/runs/run-*.jsonl.zst'))[:4]


@pytest.mark.parametrize('path', phase6_runs() or [None])
def test_phase6_v2_roundtrip(path):
    """真实 v2 机器人录制：逐字节 round-trip（含 sd/pa/ea/gs/ds），且每个实体/玩家都有外观行。"""
    if path is None:
        pytest.skip('没有 phase6 v2 录制文件')
    run = load_run.Run(path)
    assert run.format == 2
    assert load_run.roundtrip(path, verbose=False)['ok']
    for fs in run.iter_frames():
        assert all(a is not None for a in fs.player_appearance)
        if fs.entities is not None:
            assert len(fs.entity_appearance) == len(fs.entities) and all(a is not None for a in fs.entity_appearance)



def test_replay_table_start_players(tmp_path):
    """续局（header.continued）的回放表带 start_players = 录制第 1 帧各玩家位置（定点 x100）；非续局没有这个字段。"""
    lines = _v2_lines()
    hdr = json.loads(lines[0])
    hdr['meta'] = {'seeds': {'start_seed_string': 'ABCD EFGH'}}
    for continued in (True, False):
        hdr['continued'] = continued
        lines[0] = json.dumps(hdr, separators=(',', ':'))
        p = tmp_path / f'run-c{int(continued)}.jsonl'
        p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        run = load_run.Run(p)
        out = tmp_path / f'irr_replay-c{int(continued)}.lua'
        t = replay_run.build_replay_table(run, out)
        text = out.read_text(encoding='utf-8')
        if continued:
            assert replay_run.continue_start_players(run) == [(0, 32000, 38000)]
            assert t['start'] == 'menu:continue' and t['start_players'] == [{'num': 0, 'x': 320.0, 'y': 380.0}]
            assert 'start_players = {[0]={x100=32000,y100=38000}},' in text
        else:
            assert replay_run.continue_start_players(run) == []
            assert t['start'] == 'seed ABCD EFGH' and t['start_players'] == []
            assert 'start_players' not in text


def _mouse_lines():
    """_v2_lines 加鼠标流：第 1 帧两条（第一条 = 开局光标）、第 3 帧一条（带左键）。"""
    lines = _v2_lines()
    hdr = json.loads(lines[0])
    hdr['meta'] = {'seeds': {'start_seed_string': 'ABCD EFGH'}}
    lines[0] = json.dumps(hdr, separators=(',', ':'))
    f1 = json.loads(lines[2])
    f1['ms'] = [[1, 75500, 700, 21692, 8446, 0], [2, 75600, 700, 21723, 8446, 0]]
    lines[2] = json.dumps(f1, separators=(',', ':'), ensure_ascii=False)
    f3 = json.loads(lines[4])
    f3['ms'] = [[3, 152600, 38400, 45415, 20046, 1]]
    lines[4] = json.dumps(f3, separators=(',', ':'), ensure_ascii=False)
    return lines


def test_replay_table_mouse_stream(tmp_path):
    """phase8-mouse：回放表的 mouse_init = 录制第 1 帧第一条 ms（屏幕坐标定点原值 + 按键），mouse[n] = 其余条目
    [r, sx100, sy100, buttons]（r = 上一次 MC_POST_UPDATE 之后第几次渲染）；mouse=False 时没有这两个字段。"""
    p = tmp_path / 'run-ms.jsonl'
    p.write_text('\n'.join(_mouse_lines()) + '\n', encoding='utf-8')
    run = load_run.Run(p)
    init, per = replay_run.mouse_stream(run)
    assert init == [75500, 700, 0]
    assert per == {1: [[2, 75600, 700, 0]], 3: [[3, 152600, 38400, 1]]}
    out = tmp_path / 'irr_replay.lua'
    t = replay_run.build_replay_table(run, out)
    text = out.read_text(encoding='utf-8')
    assert t['mouse'] is True and t['mouse_init'] == [75500, 700, 0] and t['mouse_entries'] == 2 and t['mouse_buttons_pressed'] == 1
    assert 'mouse_init = {75500,700,0},' in text and '[1]={{2,75600,700,0}},' in text and '[3]={{3,152600,38400,1}},' in text
    t2 = replay_run.build_replay_table(run, out, mouse=False)
    assert t2['mouse'] is False and 'mouse_init' not in out.read_text(encoding='utf-8') and 'mouse = {' not in out.read_text(encoding='utf-8')
    # 没有任何 ms 的录制：不写鼠标字段
    q = tmp_path / 'run-noms.jsonl'
    q.write_text('\n'.join(_v2_lines()) + '\n', encoding='utf-8')
    assert replay_run.mouse_stream(load_run.Run(q)) == (None, {})


def test_replay_env_mouse():
    """回放进程环境：mouse=True 时多 RR_MT_MOUSE=1（rr_mt.dll 才导出可用的 RRMT_SetMouse），其余同 turbo_env。"""
    assert replay_run.replay_env(None, False) == {}
    assert replay_run.replay_env(None, True) == {'RR_MT_MOUSE': '1'}
    assert replay_run.replay_env('gate', True) == {'RR_MT_TURBO': '1', 'RR_MT_TURBO_GPU_GATE': '1', 'RR_MT_MOUSE': '1'}

def test_affinity_mask():
    """回放进程 CPU 亲和（phase8-mouse §H4）：auto = 逻辑 CPU 0；0/off/None = 不限制（不设 RR_MT_AFFINITY）；其它按掩码解析。"""
    assert replay_run.affinity_mask('auto') == 1
    assert replay_run.affinity_mask('0') == 0 and replay_run.affinity_mask('off') == 0 and replay_run.affinity_mask(None) == 0
    assert replay_run.affinity_mask('0x10') == 16 and replay_run.affinity_mask('3') == 3
    assert replay_run.replay_env(None, False, '0x10') == {'RR_MT_AFFINITY': '0x10'}
    assert 'RR_MT_AFFINITY' not in replay_run.replay_env('gate', True, '0')


def test_recording_affinity_env():
    """录制侧 CPU 亲和（phase8-mouse §H4）：带 DLL 的会话默认 RR_MT_AFFINITY=0x1；-NoDll 或 --affinity 0 不设；
    DLL 日志里的 affinity: 行能被解析，FAILED 要能看出来。"""
    assert rr.affinity_env(True) == {'RR_MT_AFFINITY': '0x1'}
    assert rr.affinity_env(True, '0x6') == {'RR_MT_AFFINITY': '0x6'}
    assert rr.affinity_env(True, '0') == {} and rr.affinity_env(False) == {}
    assert replay_run.affinity_mask('auto') == rr.affinity_mask('auto') == 1
    log = 'status=1 ready\naffinity: RR_MT_AFFINITY=0x1 -> SetProcessAffinityMask ok (process mask now 0x1, system 0xFFFFF)\nmore'
    assert rr.dll_affinity_line(log).startswith('affinity: RR_MT_AFFINITY=0x1 -> SetProcessAffinityMask ok')
    assert rr.dll_affinity_line('status=1 ready\n') is None
    assert 'FAILED' in rr.dll_affinity_line('affinity: RR_MT_AFFINITY=0x1 -> SetProcessAffinityMask FAILED (process mask now 0xFF, system 0xFF)')


def test_sparse_bare_int_tolerated():
    """ea/pa 里格式外的裸整数（数字格式损坏修复后的残留，human2b 第 10522 帧）按"该对象外观未变"处理。"""
    prev = {5: [5] + [1] * 25, 9: [9] + [2] * 25}
    rows, cur = load_run.expand_sparse([[5, [[3, 4]]], 9], prev, [5, 9])
    assert rows[0][3] == 5 and rows[1] == prev[9]
