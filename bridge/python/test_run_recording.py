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
    """回放进程环境：mouse=True 时多 RR_MT_MOUSE=1（rr_mt.dll 才导出可用的 RRMT_SetMouse），其余同 turbo_env；
    phase12 起总有 RR_MT_FORCEPAUSE=1（装 Game::IsPaused 钩子，只在 start_paused 的局开局窗口由 mod 打开）。"""
    fp = {'RR_MT_FORCEPAUSE': '1'}
    assert replay_run.replay_env(None, False) == fp
    assert replay_run.replay_env(None, True) == dict(fp, RR_MT_MOUSE='1')
    assert replay_run.replay_env('gate', True) == dict(fp, RR_MT_TURBO='1', RR_MT_TURBO_GPU_GATE='1', RR_MT_MOUSE='1')

def test_affinity_mask():
    """回放进程 CPU 亲和（phase8-mouse §H4）：auto = 逻辑 CPU 0；0/off/None = 不限制（不设 RR_MT_AFFINITY）；其它按掩码解析。"""
    assert replay_run.affinity_mask('auto') == 1
    assert replay_run.affinity_mask('0') == 0 and replay_run.affinity_mask('off') == 0 and replay_run.affinity_mask(None) == 0
    assert replay_run.affinity_mask('0x10') == 16 and replay_run.affinity_mask('3') == 3
    assert replay_run.replay_env(None, False, '0x10') == {'RR_MT_AFFINITY': '0x10', 'RR_MT_FORCEPAUSE': '1'}
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


# ---------------------------------------------------------------- phase9-robust（强杀/崩溃截断的局）
def _mt_lines():
    """_v2_lines 加 MT：header.mt 可用、每帧 mt（compare 判 exact 需要）。"""
    lines = _v2_lines()
    hdr = json.loads(lines[0])
    hdr['mt'] = {'available': True, 'reseeds': [{'at': 'POST_PLAYER_INIT', 'seeded_with': 5, 'after': [624, 1]},
                                                {'at': 'POST_GAME_STARTED', 'seeded_with': 5, 'before': [30, 2], 'after': [624, 1]}]}
    hdr['meta'] = {'seeds': {'start_seed_string': 'ABCD EFGH'}}
    lines[0] = json.dumps(hdr, separators=(',', ':'))
    for i in (2, 3, 4):
        f = json.loads(lines[i])
        f['mt'] = [100 + i, 7 * i]
        lines[i] = json.dumps(f, separators=(',', ':'), ensure_ascii=False)
    return lines


def test_partial_tail_dropped(tmp_path):
    """进程被强杀时 mod 的 256 KB 写缓冲可能停在行中间（c2-kill 实测）：最后一行没有换行且解析不了 -> 读取时丢掉并记在
    Run.partial_tail；帧数 = 完整帧行数，顺序重建不出错；会话校验报 last_line_partial、不报错误。"""
    lines = _mt_lines()
    p = tmp_path / 'run-cut.jsonl'
    cut = lines[4][:len(lines[4]) // 2]
    p.write_bytes(('\n'.join(lines[:4]) + '\n' + cut).encode('utf-8'))
    run = load_run.Run(p)
    assert run.frames == 2 and run.end is None
    assert run.partial_tail == [{'line': 4, 'bytes': len(cut.encode('utf-8')), 'kind': 'f'}]
    assert [fs.n for fs in run.iter_frames()] == [1, 2]
    st = rr.validate_lines(rr.iter_lines_raw(p))
    assert st['frames'] == 2 and st['truncated'] and st['last_line_partial'] and st['bad_json'] == 1 and not st['errors']
    # 最后一行完整但缺换行（恰好在行尾截断）：保留
    q = tmp_path / 'run-nonl.jsonl'
    q.write_bytes('\n'.join(lines).encode('utf-8'))
    r2 = load_run.Run(q)
    assert r2.frames == 3 and r2.partial_tail == []
    # 正常文件不受影响
    n = tmp_path / 'run-ok.jsonl'
    n.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert load_run.Run(n).frames == 3 and load_run.Run(n).partial_tail == []


def test_compare_truncated_record(tmp_path):
    """录制截断（没有 end 行）+ 回放在同一帧数被帧上限截止（frames）：三层 + MT 全同时 exact = True（以前 end_compatible
    只认 exit/exit_save/restart 等结束方式，截断局永远 exact = False）。长度不同仍不相容。"""
    lines = _mt_lines()
    rec = tmp_path / 'rec.jsonl'
    rec.write_bytes(('\n'.join(lines[:4]) + '\n' + lines[4][:20]).encode('utf-8'))   # 第 3 帧被截断（不完整行）
    rep = tmp_path / 'rep.jsonl'
    rep.write_text('\n'.join(lines[:4] + ['{"k":"end","n":2,"reason":"frames","frames":2}']) + '\n', encoding='utf-8')
    c = replay_run.compare(rec, rep)
    assert c['record_truncated'] and c['record_partial_tail'] and c['end_compatible'] and c['exact']
    assert c['record_frames'] == c['replay_frames'] == 2
    rep3 = tmp_path / 'rep3.jsonl'
    rep3.write_text('\n'.join(lines + ['{"k":"end","n":3,"reason":"frames","frames":3}']) + '\n', encoding='utf-8')
    c3 = replay_run.compare(rec, rep3)
    assert not c3['same_length'] and not c3['end_compatible'] and not c3['exact']
    # 有 end 行的录制：record_truncated = False，规则不变
    full = tmp_path / 'full.jsonl'
    full.write_text('\n'.join(lines + ['{"k":"end","n":3,"reason":"exit_save","frames":3}']) + '\n', encoding='utf-8')
    cf = replay_run.compare(full, rep3)
    assert cf['record_truncated'] is False and cf['end_compatible'] and cf['exact']


def test_process_exit_summary():
    """session.json 的 game_exit（phase9-robust）：TerminateProcess(1) = 异常；正常关窗口 = 0；录制端自己强制结束另有标记。"""
    assert rr.process_exit_summary(None) is None
    killed = rr.process_exit_summary({'pid': 5, 'exit_code': 1, 'exit_hex': '0x00000001', 'seconds': 70.04})
    assert killed == dict(pid=5, exit_code=1, exit_hex='0x00000001', wm_close_sent=False, forced_by_recorder=False,
                          seconds=70.0, abnormal=True)
    ok = rr.process_exit_summary({'pid': 6, 'exit_code': 0, 'exit_hex': '0x00000000', 'wm_close': True, 'forced': False, 'seconds': 3})
    assert ok['abnormal'] is False and ok['wm_close_sent'] is True
    assert '0x00000001' in rr.truncated_note(killed) and '退出码' not in rr.truncated_note(ok) and rr.truncated_note(None)


# ---------------------------------------------------------------- phase10-mods-console（第三方 mod 调用、控制台命令）
GT = 'goodtrip_1630477831'


def _api_lines():
    """_mt_lines 加 api 行（GoodTrip 一次传送 + 一次射击冷却 + 结束本局的 restart）、另一个 mod 的调用、来源不明的调用。"""
    lines = _mt_lines()
    api = [
        {'k': 'api', 'n': 2, 'gf': 1, 'ph': 'r', 'r': 2, 'id': 1, 'm': 'EntityPlayer.SetShootingCooldown', 'self': {'p': 0},
         'a': [2], 'src': GT},
        {'k': 'api', 'n': 2, 'gf': 1, 'ph': 'r', 'r': 2, 'id': 2, 'm': 'Level.EnterDoor', 'set': 1, 'a': [-1], 'src': GT},
        {'k': 'api', 'n': 2, 'gf': 1, 'ph': 'r', 'r': 2, 'id': 3, 'm': 'Game.ChangeRoom', 'a': [84, -1], 'src': GT},
        {'k': 'api', 'n': 2, 'gf': 1, 'ph': 'u', 'rc': 1, 'id': 4, 'm': 'Level.AddCurse', 'a': [2, False], 'src': GT, 'in': 3},
        {'k': 'api', 'n': 2, 'gf': 1, 'ph': 'r', 'r': 2, 'id': 5, 'm': 'Game.StartRoomTransition',
         'a': [86, -1, 1, {'p': 0}, -1], 'src': GT},
        {'k': 'api', 'n': 3, 'gf': 2, 'ph': 'r', 'r': 1, 'af': 1, 'id': 6, 'm': 'EntityPlayer.TakeDamage', 'self': {'p': 0},
         'a': [{'f': 1.5}, 268435456, {'ref': {'p': 0}}, 0], 'src': GT},
        {'k': 'api', 'n': 3, 'gf': 2, 'ph': 'p', 'r': 0, 'id': 7, 'm': 'Isaac.ExecuteCommand', 'a': ['restart'], 'src': GT},
        {'k': 'api', 'n': 3, 'gf': 2, 'ph': 'r', 'r': 1, 'id': 8, 'm': 'Game.ChangeRoom', 'a': [70, -1], 'src': 'other_mod'},
        {'k': 'api', 'n': 3, 'gf': 2, 'ph': 'r', 'r': 1, 'id': 9, 'm': 'Isaac.ExecuteCommand', 'a': ['giveitem c1'], 'src': '?',
         'srcfile': '=(console)'},
    ]
    # api 行写在所属帧的 f 行之前（lines[3] = 第 2 帧，lines[4] = 第 3 帧）
    out = lines[:3] + [json.dumps(a, separators=(',', ':')) for a in api[:5]] + [lines[3]] + \
        [json.dumps(a, separators=(',', ':')) for a in api[5:]] + lines[4:]
    return out


def test_api_lines_indexed_and_roundtrip(tmp_path):
    """recorder 0.4.0 的 api 行：load_run 按 n 归进帧事件、另记 Run.api_lines；帧重建与 round-trip 不受影响；会话校验计数。"""
    p = tmp_path / 'run-api.jsonl'
    p.write_text('\n'.join(_api_lines()) + '\n', encoding='utf-8')
    run = load_run.Run(p)
    assert run.frames == 3 and len(run.api_lines) == 9
    fs = run.frame(2)
    assert [e['id'] for e in fs.events if e['k'] == 'api'] == [1, 2, 3, 4, 5]
    assert [e['id'] for f in run.iter_frames() for e in (f.events or []) if e.get('k') == 'api'] == list(range(1, 10))
    assert replay_run.api_mods(run) == [GT, 'other_mod']
    st = rr.validate_lines(rr.iter_lines_raw(p))
    assert st['api_lines'] == 9 and st['kinds']['api'] == 9 and not st['errors']
    assert load_run.roundtrip(p, sample_every=1, verbose=False)


def test_api_replay_table(tmp_path):
    """回放表 api 段：只含被禁用的 mod 的调用；参数按类型编码（整数/浮点/玩家/EntityRef/布尔/字符串）；嵌套调用带 parent；
    结束本局的命令（restart）标 skip；来源不明的调用进 issues。"""
    p = tmp_path / 'run-api.jsonl'
    p.write_text('\n'.join(_api_lines()) + '\n', encoding='utf-8')
    run = load_run.Run(p)
    out = tmp_path / 'irr_replay.lua'
    t = replay_run.build_replay_table(run, out, api=[GT])
    text = out.read_text(encoding='utf-8')
    assert t['api_calls'] == 7 and t['api_other_mods_calls'] == 2 and t['api_mods'] == [GT]
    assert [x['id'] for x in t['api_skipped']] == [7]
    assert '{id=1,ph="r",m="EntityPlayer.SetShootingCooldown",src="goodtrip_1630477831",r=2,self=0,a={{"i",2}}}' in text
    assert '{id=2,ph="r",m="Level.EnterDoor",src="goodtrip_1630477831",r=2,set=1,a={{"i",-1}}}' in text
    assert 'id=4,ph="u",m="Level.AddCurse",src="goodtrip_1630477831",rc=1,parent=3,a={{"i",2},{"b",false}}}' in text
    assert 'a={{"i",86},{"i",-1},{"i",1},{"p",0},{"i",-1}}' in text
    assert 'a={{"f",1.5},{"i",268435456},{"ref",0},{"i",0}}' in text and 'af=1' in text
    assert 'skip="结束本局的命令（restart）' in text and 'other_mod' not in text
    assert replay_run.build_replay_table(run, out)['api_calls'] == 0 and 'api = {' not in out.read_text(encoding='utf-8')
    assert any('来源不明' in x for x in replay_run.replayability(run))
    with pytest.raises(replay_run.Unreplayable):
        replay_run.api_arg_lua({'u': 'GridEntity'})
    assert replay_run.api_arg_lua({'v': [1.5, -2.0]}) == '{"v",1.5,-2.0}' and replay_run.api_arg_lua({'ref': None}) == '{"ref"}'
    assert replay_run.api_arg_lua('restart') == '{"s","restart"}' and replay_run.api_arg_lua(None) == '{"n"}'


def test_api_compare_report(tmp_path):
    """compare 的 api 对照（诊断，不进 exact）：回放局里 rp 行按录制 id 对上，执行位置不同的列出。"""
    lines = _api_lines()
    rec = tmp_path / 'rec.jsonl'
    rec.write_text('\n'.join(lines + ['{"k":"end","n":3,"reason":"exit","frames":3}']) + '\n', encoding='utf-8')
    rep_lines = [ln for ln in lines if '"k":"api"' not in ln]
    rp = [{'k': 'api', 'n': 2, 'gf': 1, 'ph': 'r', 'r': 2, 'id': 5, 'm': 'Game.StartRoomTransition', 'src': GT, 'rp': 1,
           'where': 'render'},
          {'k': 'api', 'n': 2, 'gf': 1, 'ph': 'q', 'r': 2, 'id': 3, 'm': 'Game.ChangeRoom', 'src': GT, 'rp': 1, 'where': 'render'}]
    rep_lines = rep_lines[:3] + [json.dumps(x, separators=(',', ':')) for x in rp] + rep_lines[3:]
    rep = tmp_path / 'rep.jsonl'
    rep.write_text('\n'.join(rep_lines + ['{"k":"end","n":3,"reason":"frames","frames":3}']) + '\n', encoding='utf-8')
    c = replay_run.compare(rec, rep)
    assert c['exact'] and c['api']['record_calls'] == 9 and c['api']['replay_executed'] == 2
    assert c['api']['position_changed_count'] == 1 and c['api']['position_changed'][0]['id'] == 3
    assert 1 in c['api']['not_executed'] and c['api']['replay_live_calls'] == 0


def test_disable_and_restore_mods(tmp_path):
    """回放前给要禁用的 mod 放 disable.it（已禁用/不存在的跳过），回放后挪进回放会话目录（不删）。"""
    copy = rr.GameCopy(tmp_path / 'copy')
    for m in (GT, 'already_off'):
        (copy.mods / m).mkdir(parents=True)
    (copy.mods / 'already_off' / 'disable.it').write_bytes(b'')
    placed = replay_run.disable_mods_in_copy(copy, [GT, 'already_off', 'missing_mod'])
    assert placed == [GT] and (copy.mods / GT / 'disable.it').exists()
    moved = replay_run.restore_disabled_mods(copy, placed, tmp_path / 'sess' / 'disabled-mods')
    assert not (copy.mods / GT / 'disable.it').exists() and (copy.mods / 'already_off' / 'disable.it').exists()
    assert moved == [dict(mod=GT, moved_to=str(tmp_path / 'sess' / 'disabled-mods' / GT / 'disable.it'))]
    assert (tmp_path / 'sess' / 'disabled-mods' / GT / 'disable.it').exists()


def _console_lines(spanning=False):
    """_mt_lines 加一次控制台暂停：第 2 帧第 3 次渲染打开（ev pause 第三参数 3）、提交 giveitem c1、同帧解除（spanning=True 时
    unpause 记在第 3 帧：关控制台与第 2 帧的更新在同一次迭代）。"""
    lines = _mt_lines()
    f2 = json.loads(lines[3])
    f2['tk'] = ['i1-2', ['i2', [], [[1, 0, -1, 1]]], 'i3-40', 'u'] if not spanning else ['i1-40', 'u']
    lines[3] = json.dumps(f2, separators=(',', ':'), ensure_ascii=False)
    evs = [{'k': 'ev', 'n': 2, 'gf': 1, 'ev': 'pause', 'a': [100, 5000, 3]},
           {'k': 'ev', 'n': 2, 'gf': 1, 'ev': 'cmd', 'a': ['giveitem', 'c1'], 'src': 'console', 'line': 'giveitem c1', 'r': 3,
            'paused': True, 'k_console': 1}]
    unp = {'k': 'ev', 'n': 2, 'gf': 1, 'ev': 'unpause', 'a': [138, 5600, 39]}
    if spanning:
        unp = {'k': 'ev', 'n': 3, 'gf': 2, 'ev': 'unpause', 'a': [138, 5600, 1]}
        return lines[:3] + [json.dumps(e, separators=(',', ':')) for e in evs] + [lines[3]] + \
            [json.dumps(unp, separators=(',', ':'))] + lines[4:]
    return lines[:3] + [json.dumps(e, separators=(',', ':')) for e in evs + [unp]] + lines[3:]


def test_console_commands_replay_table(tmp_path):
    """控制台命令（ev cmd src=console）：同帧解除的控制台暂停按菜单暂停注入（console=True，open = 暂停开始渲染序号 - 1 的 tick），
    命令在第 r 次渲染执行；暂停跨过本帧更新的不注入暂停，命令在第 min(r, 2) 次渲染执行。会话校验计数。"""
    for spanning in (False, True):
        p = tmp_path / f'run-con-{int(spanning)}.jsonl'
        p.write_text('\n'.join(_console_lines(spanning)) + '\n', encoding='utf-8')
        run = load_run.Run(p)
        assert replay_run.console_commands(run) == [(2, 3, 'giveitem c1', True, not spanning)]
        plans, _ = replay_run.menu_pauses(run)
        out = tmp_path / f'irr_replay-{int(spanning)}.lua'
        t = replay_run.build_replay_table(run, out)
        text = out.read_text(encoding='utf-8')
        if spanning:
            assert plans == [] and '[2]={{r=2,line="giveitem c1"}},' in text
        else:
            assert len(plans) == 1 and plans[0]['console'] and plans[0]['open'] == 'i2' and plans[0]['open_action'] == 15
            assert '[2]={{r=3,line="giveitem c1"}},' in text and 'open="i2",open_action=15' in text
        assert t['console_cmds'][0]['pause_same_frame'] is (not spanning)
        assert any('控制台命令' in x for x in replay_run.replayability(run))
        st = rr.validate_lines(rr.iter_lines_raw(p))
        assert st['console_cmds'] == 1 and st['events']['cmd'] == 1


def test_session_allow_console(tmp_path):
    """--allow-console：不强制 EnableDebugConsole=0，游戏进程环境多 RR_MT_CONSOLE=1（DLL 挂控制台提交函数）；默认会话不变；
    不带 DLL 时拒绝。"""
    s0 = rr.Session('plain', tmp_path, copy_root=tmp_path / 'copy')
    assert s0.forced_options == {'EnableDebugConsole': '0'} and 'RR_MT_CONSOLE' not in s0.extra_env and not s0.allow_console
    s1 = rr.Session('con', tmp_path, copy_root=tmp_path / 'copy', allow_console=True)
    assert s1.forced_options == {} and s1.extra_env.get('RR_MT_CONSOLE') == '1' and s1.info['allow_console'] is True
    s2 = rr.Session('con2', tmp_path, copy_root=tmp_path / 'copy', allow_console=True, forced_options={'PauseOnFocusLost': '0'})
    assert s2.forced_options == {'PauseOnFocusLost': '0'}
    with pytest.raises(ValueError):
        rr.Session('con3', tmp_path, copy_root=tmp_path / 'copy', allow_console=True, use_dll=False)


def _meta_run(tmp_path, name, meta, continued=False, n_frames=3, end=None):
    """_mt_lines 的变体：header.meta 换成给定内容，只留前 n_frames 帧，可加 end 行。"""
    lines = _mt_lines()
    hdr = json.loads(lines[0])
    hdr['meta'] = dict({'seeds': {'start_seed_string': 'ABCD EFGH'}}, **meta)
    hdr['continued'] = continued
    lines[0] = json.dumps(hdr, separators=(',', ':'))
    out = lines[:2 + n_frames]
    if end:
        out.append(json.dumps(dict({'k': 'end', 'n': n_frames, 'frames': n_frames}, reason=end), separators=(',', ':')))
    p = tmp_path / f'{name}.jsonl'
    p.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return p


def test_start_character_and_issues(tmp_path):
    """phase11-human3：回放开局按录制的角色 `restart <玩家 0 角色>` 开预备局再 `seed`（表里 pre 字段）；续局没有 pre；
    Jacob & Esau（19/20）不算多玩家；合作多玩家、贪婪、困难各有 issue（困难的 issue 说明沿用存档里的难度）。"""
    p = _meta_run(tmp_path, 'aza', {'players': [{'type': 7}], 'difficulty': 1, 'challenge': 0})
    run = load_run.Run(p)
    assert replay_run.start_character(run) == 'restart 7'
    t = replay_run.build_replay_table(run, tmp_path / 'irr_replay.lua')
    assert t['pre'] == 'restart 7' and t['start'] == 'seed ABCD EFGH'
    issues = replay_run.replayability(run)
    assert any(s.startswith('难度 1') and '--init-save' in s for s in issues)
    assert not any('角色' in s or '多玩家' in s for s in issues)
    cont = load_run.Run(_meta_run(tmp_path, 'cont', {'players': [{'type': 7}]}, continued=True))
    assert replay_run.start_character(cont) is None
    assert replay_run.build_replay_table(cont, tmp_path / 'irr_replay2.lua')['pre'] is None
    je = load_run.Run(_meta_run(tmp_path, 'je', {'players': [{'type': 19}, {'type': 20}]}))
    assert replay_run.start_character(je) == 'restart 19'
    assert not any('多玩家' in s for s in replay_run.replayability(je))
    coop = load_run.Run(_meta_run(tmp_path, 'coop', {'players': [{'type': 0}, {'type': 7}], 'greed_mode': True}))
    iss = replay_run.replayability(coop)
    assert any('多玩家' in s for s in iss) and any('贪婪' in s for s in iss)
    assert replay_run.start_character(load_run.Run(_meta_run(tmp_path, 'nometa', {}))) is None


def test_start_target_and_name():
    """回放局名预先算：已开局数 + 1，有 pre（预备局）再 + 1。"""
    assert replay_run.start_target(1, None) == 2 and replay_run.start_target(1, 'restart 7') == 3
    assert replay_run.start_target(None, None) == 1
    assert replay_run.replay_run_name('p1-01', 3) == 'run-p1-01-0003'


class _FakeGame:
    """start_replay_run 用的假游戏进程：command 之后按脚本给出 status 序列（模拟极短回放局在轮询前就结束）。"""

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.sent = []
        self.cur = self.statuses.pop(0)

    def status(self):
        return self.cur

    def command(self, **c):
        self.sent.append(c)
        return len(self.sent)

    def wait_ack(self, cid, timeout=30):
        return self.cur

    def wait(self, pred, timeout, poll=0.1):
        while True:
            if self.cur is not None and pred(self.cur):
                return self.cur
            if not self.statuses:
                raise TimeoutError(f'等待超时，最后状态 {self.cur!r}')
            self.cur = self.statuses.pop(0)


def test_start_replay_run_short_run_already_ended():
    """phase11-human3 的 7 局：回放局（9 帧）在 Python 看到 in_game 之前就结束了（status：in_game False、run None、
    runs 已 +1）。新逻辑按 runs 计数认出回放局（名字预先算好），不再超时重试；带 pre 时目标序号再 +1。"""
    base = {'proc': 'p1-01', 'in_game': True, 'run': 'run-p1-01-0001', 'runs': 1}
    g = _FakeGame([base, dict(base, in_game=False, run=None, runs=2)])
    name, st = replay_run.start_replay_run(g, {'start': 'seed ABCD EFGH', 'pre': None}, 9)
    assert name == 'run-p1-01-0002' and st['runs'] == 2
    assert g.sent == [{'cmd': 'start', 'start': 'seed ABCD EFGH', 'replay': True, 'frames': 9}]
    # 带 pre：预备局（runs=2）不算，等到 runs=3
    g2 = _FakeGame([base, dict(base, run='run-p1-01-0002', runs=2), dict(base, in_game=True, run='run-p1-01-0003', runs=3)])
    name2, st2 = replay_run.start_replay_run(g2, {'start': 'seed ABCD EFGH', 'pre': 'restart 7'}, 100)
    assert name2 == 'run-p1-01-0003' and st2['in_game']
    assert g2.sent[0]['pre'] == 'restart 7'
    # 一直没开出来：照旧超时（外层换进程重试）
    g3 = _FakeGame([base, dict(base)])
    with pytest.raises(TimeoutError):
        replay_run.start_replay_run(g3, {'start': 'seed ABCD EFGH', 'pre': None}, 9)


def test_compare_zero_frame_record(tmp_path):
    """录制 0 帧（按住 R 在第一次逻辑更新前重开，human3 run-0025/0031）：回放 0 帧（0.4.1）或 1 帧（0.4.0）且被帧上限截止
    （frames）都算等长、exact（只比 MT reseed 与开局元数据）；回放 2 帧、或 1 帧但不是帧上限截止的，不相容。"""
    meta = {'players': [{'type': 7}], 'difficulty': 1}
    rec = _meta_run(tmp_path, 'rec0', meta, n_frames=0, end='exit')
    for k in (0, 1):
        c = replay_run.compare(rec, _meta_run(tmp_path, f'rep{k}', meta, n_frames=k, end='frames'))
        assert c['zero_frame_record'] and c['compared'] == 0 and c['same_length'] and c['end_compatible'] and c['exact'], k
    c2 = replay_run.compare(rec, _meta_run(tmp_path, 'rep2', meta, n_frames=2, end='frames'))
    assert not c2['same_length'] and not c2['exact']
    cd = replay_run.compare(rec, _meta_run(tmp_path, 'repd', meta, n_frames=1, end='death'))
    assert not cd['exact']
    # 角色不同（回放开成了 Isaac）：没有帧可比也判不精确
    ci = replay_run.compare(rec, _meta_run(tmp_path, 'repi', {'players': [{'type': 0}], 'difficulty': 1}, n_frames=0, end='frames'))
    assert not ci['start_meta_match'] and not ci['exact']
    assert ci['start_meta']['record']['players'] == [7] and ci['start_meta']['replay']['players'] == [0]


def test_custom_run_flag(tmp_path):
    """phase11-human3：录制是自然局（is_custom_run = false）时回放表写 custom_run = false（回放局在楼层生成前清标记）；
    自定义种子局写 true；续局 / 没有字段时不写。compare 的 start_meta 带 is_custom_run，不同则不 exact。"""
    def seeds(v):
        return {'seeds': {'start_seed_string': 'ABCD EFGH', 'is_custom_run': v}, 'players': [{'type': 7}]}
    for v in (False, True):
        run = load_run.Run(_meta_run(tmp_path, f'c{int(v)}', seeds(v)))
        out = tmp_path / f'irr_replay-c{int(v)}.lua'
        t = replay_run.build_replay_table(run, out)
        assert t['custom_run'] is v and replay_run.record_custom_run(run) is v
        assert f'custom_run = {"true" if v else "false"},' in out.read_text(encoding='utf-8')
    cont = load_run.Run(_meta_run(tmp_path, 'cc', seeds(False), continued=True))
    out = tmp_path / 'irr_replay-cc.lua'
    assert replay_run.build_replay_table(cont, out)['custom_run'] is None
    assert 'custom_run =' not in out.read_text(encoding='utf-8')
    rec = _meta_run(tmp_path, 'rec-nat', seeds(False), n_frames=3, end='exit')
    c = replay_run.compare(rec, _meta_run(tmp_path, 'rep-cust', seeds(True), n_frames=3, end='frames'))
    assert not c['start_meta_match'] and not c['exact'] and c['start_meta']['replay']['is_custom_run'] is True
    assert replay_run.compare(rec, _meta_run(tmp_path, 'rep-nat', seeds(False), n_frames=3, end='frames'))['exact']


def test_inject_restart_extras(tmp_path):
    """phase11-human3：按住 R 重开的局，录制最后一帧的更新里重开已经触发；回放默认注入 ACTION_RESTART（16）的全局查询
    （extra），其它 extra（地图键 13 等）不注入；inject_restart=False 时不注入；inject_extras=True 时全部注入。"""
    lines = _mt_lines()
    for i, extra in ((2, [[0, 16, -1, 1], [0, 13, -1, 1]]), (3, [[0, 13, -1, 1]])):
        f = json.loads(lines[i])
        f['tk'] = [['i1', [], extra], 'u']
        lines[i] = json.dumps(f, separators=(',', ':'), ensure_ascii=False)
    p = tmp_path / 'run-r.jsonl'
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    run = load_run.Run(p)
    out = tmp_path / 'irr_replay.lua'
    t = replay_run.build_replay_table(run, out)
    text = out.read_text(encoding='utf-8')
    assert t['inject_restart'] and t['extra_ticks'] == 1
    assert 'extras = {' in text and '[1]={["i1"]={{0,16,-1,1}}},' in text and '13,-1' not in text
    t0 = replay_run.build_replay_table(run, out, inject_restart=False)
    assert t0['extra_ticks'] == 0 and 'extras = {' not in out.read_text(encoding='utf-8')
    ta = replay_run.build_replay_table(run, out, inject_extras=True)
    assert ta['extra_ticks'] == 2 and '{0,13,-1,1}' in out.read_text(encoding='utf-8')


def _paused_run(tmp_path, name, header_paused=None, events=(), end=None, mt_shift=False):
    """_mt_lines 的变体：reseed 记录可带 paused（recorder 0.4.2），可在帧后插 ev 行 (n, ev)，可加 end 行（dict），
    mt_shift = 第 2 帧 MT 不同（模拟分歧）。"""
    lines = _mt_lines()
    hdr = json.loads(lines[0])
    hdr['continued'] = True
    if header_paused is not None:
        for r in hdr['mt']['reseeds']:
            r['paused'] = header_paused
    lines[0] = json.dumps(hdr, separators=(',', ':'))
    if mt_shift:
        f = json.loads(lines[3])
        f['mt'] = [999, 999]
        lines[3] = json.dumps(f, separators=(',', ':'), ensure_ascii=False)
    out = lines[:]
    for n, ev in events:
        out.append(json.dumps({'k': 'ev', 'n': n, 'gf': n, 'ev': ev, 'a': [0, 0, 1]}, separators=(',', ':')))
    if end:
        out.append(json.dumps(dict({'k': 'end', 'n': 3, 'frames': 3}, **end), separators=(',', ':')))
    p = tmp_path / f'{name}.jsonl'
    p.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return p


def test_record_start_paused(tmp_path):
    """phase12-human4：录制开局时是否暂停。0.4.2 起看 header.mt.reseeds[0].paused；更早的录制按事件推断（本局第一条
    pause/unpause 是 unpause = 开局时已暂停，控制台 rewind 续上的局）。为真时回放表写 start_paused = true。"""
    cases = [('h1', dict(header_paused=True), (True, 'header')),
             ('h0', dict(header_paused=False, events=[(2, 'unpause')]), (False, 'header')),
             ('e1', dict(events=[(2, 'unpause'), (5, 'pause')]), (True, 'events')),
             ('e0', dict(events=[(2, 'pause'), (2, 'unpause')]), (False, 'events')),
             ('none', dict(), (None, None))]
    for name, kw, want in cases:
        run = load_run.Run(_paused_run(tmp_path, name, **kw))
        assert replay_run.record_start_paused(run) == want, name
        out = tmp_path / f'irr_replay-{name}.lua'
        t = replay_run.build_replay_table(run, out)
        assert t['start_paused'] == want[0] and t['start_paused_from'] == want[1]
        assert ('start_paused = true,' in out.read_text(encoding='utf-8')) == bool(want[0]), name


def test_compare_start_paused(tmp_path):
    """compare 的 start_paused 报告；录制开局暂停、回放没有强制暂停（旧 DLL）且不精确 → first_divergence_category =
    start_paused_not_replayed；回放强制暂停了（end.replay.start_paused.ok）但仍分歧 → 不归这一类；精确时类别为 None。"""
    rec = _paused_run(tmp_path, 'rec', header_paused=True, end=dict(reason='exit_save'))
    forced = {'replay': {'start_paused': {'ok': True, 'hits': 27, 'left_on': False}}}
    ok = replay_run.compare(rec, _paused_run(tmp_path, 'rep-ok', header_paused=True, end=dict(reason='frames', **forced)))
    assert ok['exact'] and ok['first_divergence_category'] is None
    assert ok['start_paused'] == dict(record=True, record_from='header', replay=True, replay_forced=True,
                                      replay_stats=forced['replay']['start_paused'])
    bad = replay_run.compare(rec, _paused_run(tmp_path, 'rep-old', header_paused=False, mt_shift=True, end=dict(reason='frames')))
    assert not bad['exact'] and not bad['start_paused']['replay_forced']
    assert bad['first_divergence_category'] == 'start_paused_not_replayed'
    bad2 = replay_run.compare(rec, _paused_run(tmp_path, 'rep-forced', header_paused=True, mt_shift=True,
                                               end=dict(reason='frames', **forced)))
    assert not bad2['exact'] and bad2['first_divergence_category'] != 'start_paused_not_replayed'
    # 录制开局没暂停：不归这一类
    rec0 = _paused_run(tmp_path, 'rec0', header_paused=False, end=dict(reason='exit_save'))
    c0 = replay_run.compare(rec0, _paused_run(tmp_path, 'rep0', header_paused=False, mt_shift=True, end=dict(reason='frames')))
    assert not c0['exact'] and c0['first_divergence_category'] != 'start_paused_not_replayed'


def test_replay_lua_start_pause(tmp_path):
    """replay.lua（Lua 5.3，lupa）：回放表 start_paused = true 时 start_pause_on/off 各调一次 mt.force_pause(true/false)，
    stats_json 的 start_paused 带 ok、强制次数 hits、left_on；表里没有 start_paused 时不调用、stats 里没有这一项。"""
    lupa = pytest.importorskip('lupa.lua53')
    mod = Path(__file__).resolve().parents[1] / 'mod' / 'isaac_run_recorder' / 'replay.lua'
    for paused in (True, False):
        tbl = tmp_path / f'tbl{int(paused)}.lua'
        tbl.write_text('return {start = "menu:continue", frames = 3' + (', start_paused = true' if paused else '') + '}\n',
                       encoding='utf-8')
        rt = lupa.LuaRuntime()
        R = rt.execute(f'return dofile([[{mod}]])').load(str(tbl).replace('\\', '/'))
        calls = []
        mt = rt.eval('''function(log) local h = 100
            return {force_pause = function(on) log(on); if not on then h = h + 27 end; return true, 0, h end} end''')(calls.append)
        R.start_pause_on(mt)
        R.start_pause_off(mt)
        st = json.loads(R.stats_json()).get('start_paused')
        if paused:
            assert calls == [True, False] and st == {'ok': True, 'hits': 27, 'left_on': False}
        else:
            assert calls == [] and st is None


# ---------------------------------------------------------------------------------------------- phase13：重建安全（recorder 0.5.0）
def _deg_lines():
    """_mt_lines 改成：第 2 帧是退化帧（"err":1，只有 tk/mt/sd），其后补写 resync room 行，第 3 帧是关键帧（完整行）。"""
    lines = _mt_lines()
    f2 = json.loads(lines[3])
    deg = {'k': 'f', 'n': 2, 'gf': 2, 't': 33, 'err': 1, 'tk': f2['tk'], 'mt': f2['mt'], 'sd': f2['sd']}
    room0 = json.loads(lines[1])
    room0.pop('sd', None)
    room = {'k': 'room', 'n': 3, 'gf': 3, 'resync': 1}   # mod 把 resync 标记写在 gf 之后
    room.update({k: v for k, v in room0.items() if k not in ('k', 'n', 'gf')})
    f30 = json.loads(lines[4])
    f1 = json.loads(lines[2])
    f3 = {'k': 'f', 'n': 3, 'gf': 3, 't': f30['t'], 'kf': 1}   # 键序同 mod：kf 在 t 之后
    f3.update({k: v for k, v in f30.items() if k not in f3})
    f3.update(p=[[0, 32000, 38000, 0, 0, 'WalkDown']], pa=[f1['pa'][0]])
    def mod_order(d):   # 帧行键序同 mod（round-trip 逐字节比较要求）
        head = ['k', 'n', 'gf', 't', 'kf', 'err', 'tk', 'sp', 'ms', 'mt', 'r', 'p', 'e', 'sd', 'pa', 'ea', 'g', 'dr', 'gs', 'ds', 'dg']
        o = {k: d[k] for k in head if k in d}
        o.update({k: v for k, v in d.items() if k not in o})
        return json.dumps(o, separators=(',', ':'), ensure_ascii=False)
    return [lines[0], lines[1], mod_order(f1), mod_order(deg),
            json.dumps(room, separators=(',', ':'), ensure_ascii=False), mod_order(f3)]


def test_degraded_frame_decode(tmp_path):
    """0.5.0 退化帧：读取端标 degraded（观测为 None，不拿上一帧冒充），输入/MT 照常；resync room 行之后关键帧解码正常、
    不标 suspect；round-trip 逐字节；会话校验报 data_issues。"""
    p = tmp_path / 'deg.jsonl'
    end = json.dumps({'k': 'end', 'n': 3, 'reason': 'exit', 'frames': 3, 'errors': 1, 'err_frames': 1, 'write_errors': 0},
                     separators=(',', ':'))
    p.write_text('\n'.join(_deg_lines() + [end]) + '\n', encoding='utf-8')
    run = load_run.Run(p)
    assert run.degraded_frames == [2] and run.input_missing_frames == [] and run.frame_gaps == [] and run.suspect_frames == []
    assert run.resync_rooms and run.resync_rooms[0][1] == 3
    d = run.frame(2)
    assert d.degraded and d.players is None and d.entities is None and d.digest is None and d.room_frame is None
    assert d.mt == [103, 21] and len(d.ticks) == 1 and d.suspect == ['degraded']
    f3 = run.frame(3)
    assert not f3.degraded and f3.suspect is None and f3.players[0]['x'] == 320.0 and f3.room['grid'][22][0] == 22
    seq = list(run.iter_frames())
    assert [s.degraded for s in seq] == [False, True, False] and seq[2].players == f3.players
    assert run.strings[7] == 'HeadLeft'          # 退化帧里的字典条目照样收
    res = load_run.roundtrip(p, sample_every=1, verbose=False)
    assert res['ok'] and res['degraded_frames'] == 1, res
    st = rr.validate_lines(rr.iter_lines_raw(p))
    assert st['degraded_frames'] == [2] and st['resync_rooms'] == 1 and not st['errors']
    issues = rr.data_issues(st)
    assert any('退化帧' in x for x in issues) and any('end.errors=1' in x for x in issues)
    assert any('退化帧' in x for x in run.data_issues())


def test_gap_marks_suspect_frames(tmp_path):
    """帧号缺口（帧行没写出）且之后不是关键帧：缺口之后到下一个关键帧/room 行之前的帧标 suspect（解不出就给 None），
    不静默返回错值；归档校验报帧号缺口与 end.frames 不符。"""
    lines = _mt_lines()
    end = json.dumps({'k': 'end', 'n': 3, 'reason': 'exit', 'frames': 3, 'errors': 0, 'write_errors': 1}, separators=(',', ':'))
    p = tmp_path / 'gap.jsonl'
    p.write_text('\n'.join([lines[0], lines[1], lines[2], lines[4], end]) + '\n', encoding='utf-8')   # 缺第 2 帧
    run = load_run.Run(p)
    assert run.frame_gaps == [(1, 3)] and run.suspect_frames == [3] and run.last_n == 3 and run.frames == 2
    f3 = run.frame(3)
    assert f3.suspect and 'base_broken' in f3.suspect
    assert [s.suspect is not None for s in run.iter_frames()] == [False, True]
    st = rr.validate_lines(rr.iter_lines_raw(p))
    assert st['frame_gaps'] == [[1, 3]]
    issues = rr.data_issues(st)
    assert any('帧号缺口' in x for x in issues) and any('end.frames=3' in x for x in issues) and any('write_errors' in x for x in issues)


def test_compare_skips_degraded_frames(tmp_path):
    """录制里的退化帧没有 digest：compare 跳过这些帧的三层比较（MT 照比），其余全同 -> exact，degraded_skipped 列出；
    录制有帧号缺口 -> exact 一律 False。"""
    rec = tmp_path / 'rec.jsonl'
    rec.write_text('\n'.join(_deg_lines() + ['{"k":"end","n":3,"reason":"exit","frames":3}']) + '\n', encoding='utf-8')
    lines = _mt_lines()
    f3 = json.loads(lines[4])
    rep_lines = list(lines[:4]) + [json.dumps(f3, separators=(',', ':'), ensure_ascii=False),
                                   '{"k":"end","n":3,"reason":"frames","frames":3}']
    rep = tmp_path / 'rep.jsonl'
    rep.write_text('\n'.join(rep_lines) + '\n', encoding='utf-8')
    c = replay_run.compare(rec, rep)
    assert c['exact'] and c['degraded_skipped'] == [2] and c['exact_note']
    # MT 在退化帧上不同 -> 不 exact
    bad = _deg_lines()
    d = json.loads(bad[3])
    d['mt'] = [1, 1]
    bad[3] = json.dumps(d, separators=(',', ':'))
    rec2 = tmp_path / 'rec2.jsonl'
    rec2.write_text('\n'.join(bad + ['{"k":"end","n":3,"reason":"exit","frames":3}']) + '\n', encoding='utf-8')
    assert not replay_run.compare(rec2, rep)['exact']
    gap = tmp_path / 'gap.jsonl'
    gap.write_text('\n'.join([lines[0], lines[1], lines[2], lines[4], '{"k":"end","n":3,"reason":"exit","frames":3}']) + '\n',
                   encoding='utf-8')
    assert not replay_run.compare(gap, rep)['exact']


def test_replay_table_with_degraded_frame(tmp_path):
    """退化帧的 tk 照样进回放表；帧上限 = 最后帧号；replayability 报退化帧（不阻止回放）。"""
    p = tmp_path / 'deg.jsonl'
    p.write_text('\n'.join(_deg_lines()) + '\n', encoding='utf-8')
    run = load_run.Run(p)
    info = replay_run.build_replay_table(run, tmp_path / 'irr_replay.lua', mouse=False)
    text = (tmp_path / 'irr_replay.lua').read_text(encoding='utf-8')
    assert info['frames'] == 3 and 'frames = 3,' in text
    assert any('退化帧' in x for x in replay_run.replayability(run))


def test_truncated_note_write_failure():
    """有结束信号却没有 end 行 = end 行没写出去（写失败），不是进程异常结束；flush 间隔决定"丢多少"的措辞。"""
    n = rr.truncated_note({'abnormal': True, 'exit_hex': '0xC0000005'}, {'reason': 'exit', 'write_errors': 3})
    assert '写失败' in n and '异常结束' not in n.split('不是')[0]
    assert '约 1 秒' in rr.truncated_note(None, None, 30) and '约 10 秒' in rr.truncated_note(None, None, None)
    assert '0xC0000005' in rr.truncated_note({'abnormal': True, 'exit_hex': '0xC0000005'})


def test_snapshot_blobs_dedup_and_old_layout(tmp_path):
    """快照按内容去重存 save/blobs/<sha256>（两份同内容快照共用一个 blob），snapshot_files 新旧布局都认，
    replay_run.restore_init_save 从去重布局拷回存档目录。"""
    sd = tmp_path / 'savedir'
    sd.mkdir()
    (sd / 'persistentgamedata1.dat').write_bytes(b'P1' * 100)
    (sd / 'gamestate1.dat').write_bytes(b'G' * 50)
    (sd / 'log.txt').write_text('log')
    blobs = tmp_path / 'save' / 'blobs'
    m1 = rr.snapshot_save(tmp_path / 'save' / 'run-a', sd, blobs=blobs)
    m2 = rr.snapshot_save(tmp_path / 'save' / 'run-b', sd, blobs=blobs)
    assert m1['files']['gamestate1.dat']['sha256'] == m2['files']['gamestate1.dat']['sha256']
    assert len(list(blobs.iterdir())) == 2 and not (tmp_path / 'save' / 'run-a' / 'gamestate1.dat').exists()
    fa = rr.snapshot_files(tmp_path / 'save' / 'run-a')
    assert sorted(fa) == ['gamestate1.dat', 'persistentgamedata1.dat'] and fa['gamestate1.dat'].read_bytes() == b'G' * 50
    old = tmp_path / 'old'
    rr.snapshot_save(old, sd)          # 旧布局：文件直接放在快照目录
    fo = rr.snapshot_files(old)
    assert sorted(fo) == ['gamestate1.dat', 'persistentgamedata1.dat'] and fo['gamestate1.dat'].parent == old

    class FakeSess:
        pass
    s = FakeSess()
    s.save_dir = tmp_path / 'target'
    s.save_dir.mkdir()
    (s.save_dir / 'gamestate1.dat').write_bytes(b'old')
    s.dir = tmp_path / 'replaysess'
    s.init_save_from = tmp_path / 'save' / 'run-a'
    s.init_save_exclude = {'gamestate1.dat'}
    s.enforce_options = False
    r = replay_run.restore_init_save(s, 1)
    assert r['copied'] == ['persistentgamedata1.dat'] and r['moved'] == ['gamestate1.dat']
    assert (s.save_dir / 'persistentgamedata1.dat').read_bytes() == b'P1' * 100


def _write_signal(irr, run, save_copy, mtime=None):
    p = irr / f'signal-{run}.start.json'
    p.write_text(json.dumps({'run': run, 'continued': False, 'save_copy': save_copy}), encoding='utf-8')
    if mtime is not None:
        import os
        os.utime(p, (mtime, mtime))
    return p


def test_savewatcher_mod_copy_and_rapid_restarts(tmp_path):
    """mod 的开局拷贝（信号 save_copy + savecopy-<run>-<文件>）直接入库为该局快照（kind=mod、不会晚），连续 3 个开局信号
    一次处理、内容各自正确；同内容去重。"""
    irr, save_root, sd = tmp_path / 'lua', tmp_path / 'save', tmp_path / 'savedir'
    irr.mkdir()
    sd.mkdir()
    for k, content in enumerate([b'A' * 10, b'B' * 10, b'A' * 10], 1):
        run = f'run-p1-01-{k:04d}'
        (irr / f'savecopy-{run}-gamestate1.dat').write_bytes(content)
        (irr / f'savecopy-{run}-persistentgamedata1.dat').write_bytes(b'P')
        _write_signal(irr, run, {'files': {'gamestate1.dat': 10, 'persistentgamedata1.dat': 1, 'gamestate2.dat': None},
                                 'bytes': 11, 'ms': 1})
    recs = {}
    w = rr.SaveWatcher(irr, save_root, recs, sd, check=False)
    w.poll_once()
    assert sorted(recs) == ['run-p1-01-0001', 'run-p1-01-0002', 'run-p1-01-0003']
    for run, content in (('run-p1-01-0001', b'A' * 10), ('run-p1-01-0002', b'B' * 10), ('run-p1-01-0003', b'A' * 10)):
        assert recs[run]['kind'] == 'mod' and recs[run]['late'] is False and recs[run]['absent'] == ['gamestate2.dat']
        assert rr.snapshot_files(save_root / run)['gamestate1.dat'].read_bytes() == content
        assert rr.snapshot_status(save_root / run)['late'] is False
    assert len(list((save_root / 'blobs').iterdir())) == 3   # A、B、P


def test_savewatcher_python_fallback_detects_late(tmp_path):
    """没有 mod 拷贝（旧 mod）时退回 Python 快照：存档文件 mtime 晚于本局开局信号 = late，并写进清单；
    snapshot_status 对旧会话（清单里没有 late）按开局信号推断出同样结论。"""
    import os
    import time as _t
    irr, save_root, sd = tmp_path / 'lua', tmp_path / 'save', tmp_path / 'savedir'
    irr.mkdir()
    sd.mkdir()
    now = _t.time()
    (sd / 'gamestate1.dat').write_bytes(b'later')
    os.utime(sd / 'gamestate1.dat', (now, now))
    _write_signal(irr, 'run-p1-01-0001', None, mtime=now - 5)
    recs = {}
    w = rr.SaveWatcher(irr, save_root, recs, sd, check=False)
    rr_wait = rr.wait_save_quiet
    try:
        rr.wait_save_quiet = lambda *a, **k: 0.0
        w.poll_once()
    finally:
        rr.wait_save_quiet = rr_wait
    rec = recs['run-p1-01-0001']
    assert rec['kind'] == 'python' and rec['late'] and rec['late_reasons']
    assert rr.snapshot_status(save_root / 'run-p1-01-0001')['late'] is True
    # 旧会话布局：清单里没有 late 字段 -> 按 lua/signal-<run>.start.json 推断
    old = tmp_path / 'oldsess'
    (old / 'lua').mkdir(parents=True)
    _write_signal(old / 'lua', 'run-x', None, mtime=now - 5)
    _write_signal(old / 'lua', 'run-y', None, mtime=now + 5)
    for run in ('run-x', 'run-y'):
        rr.snapshot_save(old / 'save' / run, sd)
        man = json.loads((old / 'save' / run / 'manifest.json').read_text(encoding='utf-8'))
        assert 'late' not in man
    assert rr.snapshot_status(old / 'save' / 'run-x')['late'] is True
    assert rr.snapshot_status(old / 'save' / 'run-y')['late'] is False


def test_recover_session(tmp_path):
    """启动器被杀之后 --recover：归档 lua/ 里的原始 run 文件（截断局标出）、mod 开局拷贝入库、session.json 补成
    state=recovered、按 recover.json 还原 options.ini；已收尾的会话拒绝重做；find_unfinished 找得到未收尾的会话。"""
    out_root = tmp_path / 'sessions'
    sdir = out_root / 'kill1'
    (sdir / 'lua').mkdir(parents=True)
    (sdir / 'game').mkdir()
    copy = tmp_path / 'copy'
    save_dir = copy / 'profile' / rr.SAVE_SUBDIR
    save_dir.mkdir(parents=True)
    (save_dir / 'options.ini').write_text('[Options]\nEnableDebugConsole=0\nPauseOnFocusLost=1\n', encoding='utf-8')
    (save_dir / 'persistentgamedata1.dat').write_bytes(b'P' * 7)
    rec = dict(schema='isaac-run-recover-v1', state='recording', name='kill1', session_dir=str(sdir), out_root=str(out_root),
               copy=str(copy), save_dir=str(save_dir), use_dll=True, raw_policy='stage', allow_console=False,
               launcher_pid=1, options_file=str(save_dir / 'options.ini'), forced_before={'EnableDebugConsole': '1'},
               forced={'EnableDebugConsole': '0'}, cmd_history_before=[])
    (sdir / 'recover.json').write_text(json.dumps(rec), encoding='utf-8')
    (sdir / 'session.json').write_text(json.dumps(dict(schema='isaac-run-session-v1', name='kill1', state='recording',
                                                       launches=[], runs=[], notes=[],
                                                       options=dict(before={'EnableDebugConsole': '1'}))), encoding='utf-8')
    lines = _mt_lines()
    (sdir / 'lua' / 'run-p1-01-0001.jsonl').write_bytes(('\n'.join(lines[:4]) + '\n' + lines[4][:30]).encode('utf-8'))
    (sdir / 'lua' / 'savecopy-run-p1-01-0001-persistentgamedata1.dat').write_bytes(b'P' * 7)
    _write_signal(sdir / 'lua', 'run-p1-01-0001', {'files': {'persistentgamedata1.dat': 7, 'gamestate1.dat': None}})
    assert rr.find_unfinished(out_root) == [sdir]
    sess, info = rr.recover_session(sdir)
    assert info['state'] == 'recovered' and info['recovered']['raw_runs_found'] == ['run-p1-01-0001.jsonl']
    r = info['runs'][0]
    assert r['verified'] and r['validation']['truncated'] and r['validation']['frames'] == 2 and r['data_ok'] is True
    assert r['save_snapshot']['kind'] == 'mod' and '异常结束' in r['note']
    assert 'EnableDebugConsole=1' in (save_dir / 'options.ini').read_text(encoding='utf-8')
    assert info['options']['restored'] == {'EnableDebugConsole': '1'}
    assert json.loads((sdir / 'recover.json').read_text(encoding='utf-8'))['state'] == 'recovered'
    assert json.loads((sdir / 'session.json').read_text(encoding='utf-8'))['state'] == 'recovered'
    assert (sdir / 'runs' / 'run-0001.jsonl.zst').exists() and not (sdir / 'lua' / 'run-p1-01-0001.jsonl').exists()
    assert rr.find_unfinished(out_root) == []
    with pytest.raises(RuntimeError):
        rr.recover_session(sdir)


def test_game_process_env_drops_fault(monkeypatch):
    """IRR_FAULT 只在调用方显式给出时传给游戏进程（不从外层环境继承，免得回放进程也注入）；IRR_SAVE_DIR/FILES 总是给。"""
    captured = {}

    class Boom(Exception):
        pass

    def fake_popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        captured['env'] = env
        raise Boom()
    monkeypatch.setenv('IRR_FAULT', 'frame:1')
    monkeypatch.setattr(rr.subprocess, 'Popen', fake_popen)
    import tempfile
    d = Path(tempfile.mkdtemp())
    copy = rr.GameCopy(d / 'copy')
    with pytest.raises(Boom):
        rr.GameProcess(copy, d / 'irr', 'p1', d / 'logs', use_dll=False, save_dir=d / 'save')
    env = captured['env']
    assert 'IRR_FAULT' not in env and env['IRR_SAVE_DIR'].endswith('/save') and 'gamestate1.dat' in env['IRR_SAVE_FILES']
    with pytest.raises(Boom):
        rr.GameProcess(copy, d / 'irr', 'p2', d / 'logs', use_dll=False, save_dir=d / 'save', extra_env={'IRR_FAULT': 'room:2'})
    assert captured['env']['IRR_FAULT'] == 'room:2'
