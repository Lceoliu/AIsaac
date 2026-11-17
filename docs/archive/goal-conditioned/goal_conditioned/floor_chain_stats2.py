import sys, json, random, collections
sys.path.insert(0, r'D:/Projects/fortune/Isaac/rl/macro')
from isaac_macro.roomconfig import default_room_config
from isaac_macro.level import GameContext, generate_floor
from isaac_macro.floor import Floor, START_INDEX
cat = json.load(open(r'D:/Projects/fortune/Isaac/rl/bridge/abplus/catalog/abplus_basement1_rooms.json'))
normal = {r['variant']: r for r in cat['normal']}
train = set(json.load(open(r'D:/Projects/fortune/Isaac/rl/bridge/abplus/catalog/scaling_normal_rooms.json')).get('normal', []))
rc = default_room_config(); rng = random.Random(2); N = 2000
st = collections.Counter(); cand_hist = collections.Counter()
def combat1x1(r):
    if r is None or r.type != 1 or r.shape != 1: return False
    c = normal.get(r.variant); return bool(c and c['enemies'] and not c['clear'])
for i in range(N):
    f = Floor.from_level(generate_floor(rc, GameContext(), stage=1, stage_type=0, stage_seed=rng.getrandbits(32)))
    st['floors'] += 1
    start = f.room_at(START_INDEX)
    ok = False
    for s, a in f.neighbors(start):
        if not combat1x1(a): continue
        st['start_nb_1x1_combat'] += 1
        if a.variant in train: st['start_nb_1x1_combat_in_C39_set'] += 1
        # candidate doors after clear: visible, openable without key/bomb: exclude secret/supersecret, shop/arcade/library(locked),
        vis = [(sl, x) for sl, x in f.neighbors(a) if x.type not in (7, 8)]
        openable = [(sl, x) for sl, x in vis if x.type not in (2, 9, 12)]
        cand_hist[len(openable)] += 1
        bs = [x for sl, x in openable if combat1x1(x) and x.index != start.index]
        if bs:
            ok = True; st['A_with_B'] += 1
            if a.variant in train and any(x.variant in train for x in bs): st['A_and_B_in_C39_set'] += 1
    if ok: st['floor_ok_start_nb_chain'] += 1
print(dict(st), 'train set size', len(train))
print('visible openable doors of start-neighbour 1x1 combat rooms (incl. back to start):', sorted(cand_hist.items()))
