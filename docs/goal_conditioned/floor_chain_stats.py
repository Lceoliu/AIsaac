import sys, json, random, time, collections
sys.path.insert(0, r'D:/Projects/fortune/Isaac/rl/macro')
from isaac_macro.roomconfig import default_room_config
from isaac_macro.level import GameContext, generate_floor
from isaac_macro.floor import Floor, START_INDEX
cat = json.load(open(r'D:/Projects/fortune/Isaac/rl/bridge/abplus/catalog/abplus_basement1_rooms.json'))
normal = {r['variant']: r for r in cat['normal']}
print('catalog stage', cat.get('stage'), 'engine', cat.get('engine'))
rc = default_room_config()
rng = random.Random(1)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
st = collections.Counter()
door_hist = collections.Counter(); nb_hist = collections.Counter()
t = time.time()
for i in range(N):
    seed = rng.getrandbits(32)
    try:
        lv = generate_floor(rc, GameContext(), stage=1, stage_type=0, stage_seed=seed)
    except Exception as e:
        st['gen_error'] += 1; continue
    f = Floor.from_level(lv)
    st['floors'] += 1
    start = f.room_at(START_INDEX)
    def combat1x1(r):
        if r is None or r.type != 1 or r.shape != 1: return False
        c = normal.get(r.variant)
        if c is None: st['variant_missing'] += 1; return False
        return bool(c['enemies']) and not c['clear']
    snb = f.neighbors(start)
    st['start_doors_sum'] += len(snb)
    if any(combat1x1(r) for _, r in snb): st['start_has_1x1_combat_nb'] += 1
    has_pair = False
    for r in f.rooms:
        if not combat1x1(r) or r.index == start.index: continue
        st['rooms_1x1_combat'] += 1
        nbs = f.neighbors(r)
        door_hist[len(nbs)] += 1
        k = sum(1 for _, x in nbs if combat1x1(x) and x.index != start.index)
        nb_hist[k] += 1
        if k: has_pair = True
        if any(x.type in (7, 8) for _, x in nbs): st['1x1_adjacent_secret_or_super'] += 1
        if any(x.type == 7 for _, x in nbs): st['1x1_adjacent_secret'] += 1
        if any(x.type in (2, 9, 12) for _, x in nbs): st['1x1_adjacent_locked_type(shop/arcade/library)'] += 1
    if has_pair: st['floor_has_adjacent_1x1_combat_pair'] += 1
print('time/floor ms', 1000*(time.time()-t)/max(1,N))
print(dict(st))
print('doors per 1x1 combat room (non-start):', sorted(door_hist.items()))
print('1x1 combat neighbours per 1x1 combat room:', sorted(nb_hist.items()))
