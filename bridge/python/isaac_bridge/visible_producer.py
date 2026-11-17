"""visible-v2 producer half: the bridge observation -> a complete visible frame (observation_frame,
prioritize_entities, pack_frame, write_frame, the canvas packers). Moved verbatim from structured_input.py
(2026-11-12, DESIGN.md section 5: the producer side is package C's); structured_input re-exports every name,
so the existing imports keep working. The decoder half (unpack_*, decode_*, StructuredSpatial) stays there."""
import numpy as np

from .schema_v2 import (SCHEMA, VERSION, FRAME_BYTES, P_FIELDS, STAGE_INDEX, CURSE_START, OVERFLOW_START,
                        ENTITY_CLASSES, PF, EF, NF, SF, RC, HEADER, ROOM_BITS, ROOM_SCALES)


def pack_canvas(x,bits,scales):
    values=np.moveaxis(np.asarray(x),-3,-1)*np.asarray(scales)
    integer=np.rint(values).astype(np.int64)
    if not np.allclose(values,integer,atol=1e-5) or np.any(integer<0) or np.any(integer>=2**np.array(bits)):
        raise ValueError('canvas values are outside the lossless visible terrain schema')
    shifts=np.cumsum((0,)+tuple(bits[:-1]))
    return (integer<<shifts).sum(-1).astype('<f4')


def pack_room(x):
    # Thirteen discrete public channels packed losslessly, plus explicit visible
    # grid type/variant/state. Four float32 words per cell rather than sixteen.
    x=np.asarray(x)
    return np.concatenate((pack_canvas(x[...,:13,:,:],ROOM_BITS,ROOM_SCALES)[...,None],
                           np.moveaxis(x[...,13:,:,:],-3,-1)),axis=-1)


def pack_frame(body, room, local, minimap, entity, inventory, slots):
    counts = (len(body), len(entity), len(inventory), len(slots))
    if not counts[0] or counts[1] > 128:
        raise ValueError('visible-v2 requires bodies and at most 128 visible entities')
    arrays = [np.asarray(x, dtype='<f4').reshape(-1) for x in
              (pack_room(room),pack_canvas(minimap,(1,)*8,(1,)*8),body,
               pack_room(local),entity,inventory,slots)]
    size = HEADER + sum(a.size for a in arrays)
    if size*4 > FRAME_BYTES:
        raise ValueError(f'complete visible-v2 frame needs {size*4} bytes; transport budget {FRAME_BYTES}; no truncation')
    # Float header magic is exactly representable; the separate ROW version is int32.
    return np.concatenate((np.array([61107, size, *counts, 1, 0], '<f4'), *arrays)).tobytes()


def write_frame(row, payload):
    if len(payload) > FRAME_BYTES:
        raise ValueError('visible-v2 frame exceeds transport byte budget')
    row['visible_bytes'] = len(payload)
    row['visible_frame'][0, :len(payload)] = np.frombuffer(payload, np.uint8)
    row['visible_schema'] = VERSION


def observation_frame(o):
    """Only public raw bridge fields. Reward/privileged totals never enter this path."""
    if o.visible is None:
        raise ValueError('bridge did not send visible-v2 extension')
    ext=o.visible
    if ext['schema'] != SCHEMA or len(ext['bodies']) != len(o.players):
        raise ValueError('bridge visible-v2 schema/body mismatch')
    body=np.zeros((len(o.players),PF),np.float32)
    room=np.zeros((RC,16,28),np.float32)
    terrain=o.terrain
    if terrain is None:
        raise ValueError('visible-v2 requires room terrain')
    width,height=terrain['width'],terrain['height']
    if width>28 or height>16: raise ValueError('room canvas overflow')
    positions=[]
    for index,x,y,collision,inside,walkable,pit,hazard,solid,destructible in terrain['cells']:
        iy,ix=divmod(index,width)
        # Objective grid collision; never player-0 CanFly/size-dependent pathfinding.
        room[:6,iy,ix]=(inside,collision/6,solid,pit,destructible,hazard)
        positions.append((index,x,y))
    def cell(x,y):
        index,_,_=min(positions,key=lambda c:(c[1]-x)**2+(c[2]-y)**2)
        return divmod(index,width)
    for d in o.doors:
        iy,ix=cell(d['x'],d['y'])
        room[6:12,iy,ix]=(1,d['open'],d['locked'],d['target_type']/30,int(d['seen'])&1,(int(d['seen'])>>1)&1)
    for g in o.grid or ():
        iy,ix=cell(g[5],g[6])
        room[13:16,iy,ix]=(g[1]/32,g[2]/256,g[3]/256)
        if g[1] in (17,18):
            iy,ix=cell(g[5],g[6]);room[12,iy,ix]=1
    local=np.zeros((len(body),RC,9,9),np.float32)
    scales=np.array([520,280,10,10,20,12,12,12,65535,6,2,2,4,100,100,100,10,20,2,500,2,5,
                     1,1,1,4,4,4,60,1,20,60,60,2048,32],np.float32)
    health=[P_FIELDS.index(k) for k in ('hearts','max_hearts','soul','black','bone','eternal','golden')]
    for i,(p,meta) in enumerate(zip(o.players,ext['bodies'])):
        values=np.array([0 if k.startswith('unknown_') else p[k] for k in P_FIELDS],np.float32)
        values[0]-=o.room[8];values[1]-=o.room[9]
        known=not(ext['curses'] & 8)
        if not known: values[health]=0
        body[i,:35]=values/scales
        body[i,35:42]=(i,meta['controller'],meta['controller_known'],meta['animation']/32,
                       meta['animation_known'],float(known),ext['capabilities'])
        body[i,STAGE_INDEX]=ext['stage']/16
        body[i,CURSE_START:OVERFLOW_START]=[(int(ext['curses'])>>bit)&1 for bit in range(32)]
        iy,ix=cell(p['x'],p['y'])
        padded=np.pad(room,((0,0),(4,4),(4,4)))
        local[i]=padded[:,iy:iy+9,ix:ix+9]
    # Legal minimap only, never room descriptors that the player has not seen.
    minimap=np.zeros((8,13,13),np.float32)
    from .tok_obs import SHAPE_CELLS
    for m in o.map:
        idx=int(m['idx'])
        for delta in (0,)+SHAPE_CELLS.get(int(m['shape']),()):
            j=idx+delta
            if 0<=j<169:
                y,x=divmod(j,13);kind=int(m['type']);flags=int(m['flags'])
                minimap[:,y,x]=(1,flags&1,(flags>>1)&1,idx==o.room[4],kind==5,kind==4,kind==2,kind not in (1,2,4,5))
    entities,overflow=prioritize_entities(o.entities,o.players)
    body[:,OVERFLOW_START:]=overflow/128
    ef=('type','variant','subtype','x','y','vx','vy','size','size_mx','size_my','coll','gcoll','cdmg',
        'aframe','age','kind','flags','anim','flip','champion','hp','height','fall','scale',
        'l_cos','l_sin','l_len','l_ex','l_ey','l_radius','l_circle','price')
    entity=np.zeros((len(entities),EF),np.float32)
    for i,en in enumerate(entities):
        vals=np.array([en[k] for k in ef],np.float32)
        identity_known=not(ext['curses']&64 and en['type']==5 and en['variant']==100)
        if not identity_known:vals[2]=0
        vals[3]=(vals[3]-o.room[8])/520;vals[4]=(vals[4]-o.room[9])/280
        vals[5:10]/=20;vals[13:15]/=60;vals[20]=vals[20] if int(en['flags'])&4 else 0
        vals[21:24]/=30;vals[26:30]/=520;vals[31]/=100
        entity[i,:32]=vals
        word=int(en['tflags'])
        entity[i,32:47]=[(word>>k)&1 for k in (0,1,2,3,4,5,6,7,8,10,12,13,16,19,22)]
        entity[i,47]=identity_known
    inv=np.asarray(ext['inventory'],np.float32).reshape(-1,NF)
    slots=np.asarray(ext['slots'],np.float32).reshape(-1,SF)
    return pack_frame(body,room,local,minimap,entity,inv,slots)


def prioritize_entities(entities,players):
    """Keep interaction targets before combat traffic; count every discarded class.

    Doors/grid exits already live in uncapped R channels. Entity-form exits use
    the existing bridge synthetic identity1023, and receive the same top priority.
    Type values follow the AB+ EntityType enum; kind/flags are lean protocol enums.
    """
    kind,etype=entities['kind'],entities['type']
    priority=np.full(len(entities),4,dtype=np.int64)
    priority[np.isin(kind,(1,2,3,4))|np.isin(etype,(2,4,7,8,9))|
             ((etype==1000)&(entities['cdmg']>0))]=3
    priority[etype==3]=2
    priority[(kind==6)|((entities['flags']&1)!=0)]=1
    priority[(kind==5)|np.isin(etype,(5,6,1022,1023))]=0
    distances=np.min(np.stack([(entities['x']-p['x'])**2+(entities['y']-p['y'])**2
                              for p in players]),axis=0)
    order=np.lexsort((np.arange(len(entities)),distances,priority))[:128]
    overflow=np.bincount(priority,minlength=len(ENTITY_CLASSES))-np.bincount(priority[order],minlength=len(ENTITY_CLASSES))
    return entities[order],overflow


# ================================================================================================ visible-v3 (package C)
# The reference encoder (2026-11-17; schema_v3 VERSION 2026111702, batches 1 / 2): abp_bridge.lua visible_v3_raw()'s
# JSON (raw game values, world positions) -> the v3 frame bytes. The native encoder (abp_visible.c v3) must give the
# same bytes. Frame order (float32, little-endian): header 8 | room canvas 16 x 28 cells x 5 words (row-major, a cell's
# 5 words together) | minimap 13 x 13 | P p x 192 | local p x 9 x 9 x 5 (row-major) | E e x 64 | N n x 8 | S s x 24.
# Wire values are raw (the model scales and clamps, schema_v3.NUM_CLAMP); positions in the canvas frame (world -
# origin, origin = GetGridPosition(0) - (CELL / 2, CELL / 2), sent by the bridge); a cell = floor(position / CELL).
import math as _math

from . import schema_v3 as V3

# effect variants never sent (abp_bridge.lua V3_DECORATIVE; tests compare the two lists): the recorder's fx_omit list
V3_DECORATIVE = (2, 3, 4, 5, 7, 11, 12, 13, 14, 15, 16, 17, 18, 21, 27, 33, 35, 38, 41, 43, 50, 58, 62, 63, 64, 66, 68,
                 69, 70, 75, 77, 79, 80, 85, 86, 87, 88, 95, 98, 99, 121, 130, 131, 132, 133, 135, 138, 140, 145)
_P = {c[0]: i for i, c in enumerate(V3.P_COLS) if c[1] != 'reserved'}
_E = {c[0]: i for i, c in enumerate(V3.E_COLS) if c[1] != 'reserved'}
_N = {c[0]: i for i, c in enumerate(V3.N_COLS) if c[1] != 'reserved'}
_S = {c[0]: i for i, c in enumerate(V3.S_COLS) if c[1] != 'reserved'}
_HEALTH = ('hearts', 'max_hearts', 'soul', 'bone', 'eternal', 'golden', 'broken', 'rotten')
_P_RAW = ('vx', 'vy', 'size') + _HEALTH + ('extra_lives', 'coins', 'bombs', 'keys', 'giga_bombs', 'golden_bomb',
          'golden_key', 'poop_mana', 'soul_charge', 'blood_charge', 'damage', 'max_fire_delay', 'shot_speed', 'range',
          'move_speed', 'luck', 'can_fly', 'invulnerable', 'controls', 'dead', 'head_dir', 'fire_dir', 'move_dir',
          'sprite_frame', 'weapon_charge')
_E_RAW = ('type', 'variant', 'subtype', 'vx', 'vy', 'size', 'size_mx', 'size_my', 'coll', 'gcoll', 'cdmg',
          'sprite_frame', 'age', 'npc_flags', 'anim', 'flip_x', 'champion', 'boss_hp', 'height', 'fall', 'scale',
          'l_cos', 'l_sin', 'l_len', 'l_radius', 'l_circle', 'price', 'identity_known', 'flags_known')


def cell_of(x, y, gw, gh):
    """The canvas cell (col, row) of a canvas-frame position, None outside the grid (schema_v3 batch 1, A1)."""
    col, row = _math.floor(x / V3.CELL), _math.floor(y / V3.CELL)
    return (col, row) if 0 <= col < gw and 0 <= row < gh else None


def _pack(fields, layout):
    """A bit word from {name: value} by (name, bits) LSB first; a value outside its bits is an error (lossless)."""
    word, shift = 0, 0
    for name, bits in layout:
        v = int(fields.get(name, 0))
        if v < 0 or v >= (1 << bits):
            raise ValueError(f'visible-v3: {name} = {v} does not fit {bits} bits')
        word |= v << shift
        shift += bits
    if word >= 1 << 24:
        raise ValueError('visible-v3: a bit word past 24 bits')
    return float(word)


def v3_priority(e):
    """The entity's priority class (ENTITY_CLASSES order: 0 interactable, 1 monster, 2 familiar, 3 projectile, 4
    other), v2's rule on the raw kind (0 other, 1 tear, 2 projectile, 3 laser, 4 bomb, 5 pickup, 6 npc, 7 knife, 8
    familiar, 9 effect)."""
    k, t = int(e['kind']), int(e['type'])
    if k == 5 or t in (5, 6, 1022, 1023):
        return 0
    if k == 6 or int(e.get('npc_flags', 0)) & 1:
        return 1
    if t == 3:
        return 2
    if k in (1, 2, 3, 4, 7) or t in (2, 4, 7, 8, 9) or (t == 1000 and float(e.get('cdmg', 0)) > 0):
        return 3
    return 4


def v3_prioritize(entities, bodies):
    """At most CAP_ENTITY entities: by class, then the distance to the nearest body, then the bridge's order; the
    dropped count per class."""
    keyed = []
    for i, e in enumerate(entities):
        d = min(((e['x'] - b['x']) ** 2 + (e['y'] - b['y']) ** 2 for b in bodies), default=0.0)
        keyed.append((v3_priority(e), d, i))
    keyed.sort()
    kept = keyed[:V3.CAP_ENTITY]
    overflow = [0] * len(V3.ENTITY_CLASSES)
    for c, _, _ in keyed[V3.CAP_ENTITY:]:
        overflow[c] += 1
    return [entities[i] for _, _, i in kept], overflow


def observation_frame_v3(raw, stall_share=0.0, stall_known=0, producer=V3.PRODUCER['abplus']):
    """visible_v3_raw()'s dict (or JSON text) -> the v3 frame bytes. stall_share / stall_known: the worker's P cols 40
    / 53 (encode_row writes them again after its progress update; here for the fixtures)."""
    if isinstance(raw, (str, bytes)):
        import json
        raw = json.loads(raw)
    if raw.get('schema') != 'isaac-visible-raw-v3':
        raise ValueError('visible-v3: not a v3 raw block')
    ox, oy = float(raw['origin'][0]), float(raw['origin'][1])
    gw, gh = int(raw['gw']), int(raw['gh'])
    if not (0 < gw <= V3.CANVAS_W and 0 < gh <= V3.CANVAS_H):
        raise ValueError(f'visible-v3: grid {gw} x {gh} past the canvas')
    curses = int(raw['curses'])
    unknown = bool(curses & 8)
    # room canvas
    canvas = np.zeros((V3.CANVAS_H, V3.CANVAS_W, V3.ROOM_WORDS), '<f4')
    w0 = [dict() for _ in range(gw * gh)]
    w1 = [dict() for _ in range(gw * gh)]
    raws = [(0, 0, 0)] * (gw * gh)
    for i, c in enumerate(raw['cells'][:gw * gh]):
        inside, coll, solid, pit, destr, hazard, gt, gv, gs = c
        w0[i].update(inside=inside, collision=coll, solid=solid, pit=pit, destructible=destr, hazard=hazard,
                     exit=int(gt in (17, 18)))
        raws[i] = (gt, gv, gs)
    for d in raw['doors']:
        at = cell_of(d['x'] - ox, d['y'] - oy, gw, gh)
        if at is None:
            continue
        i = at[1] * gw + at[0]
        w0[i].update(door=1, door_open=d['open'], door_locked=d['locked'], door_target=d['target'],
                     visited=d['visited'], clear=d['clear'])
        w1[i].update(door_variant=d['variant'], door_state=d['state'])
    for i in range(gw * gh):
        row, col = divmod(i, gw)
        canvas[row, col] = (_pack(w0[i], V3.ROOM_W0), _pack(w1[i], V3.ROOM_W1)) + tuple(float(v) for v in raws[i])
    # minimap
    from .tok_obs import SHAPE_CELLS
    minimap = np.zeros(V3.MAP_SIDE * V3.MAP_SIDE, '<f4')
    for gi, shape, icon, visited, clear, current in raw['map']:
        for delta in (0,) + SHAPE_CELLS.get(int(shape), ()):
            j = int(gi) + delta
            if 0 <= j < V3.MAP_SIDE * V3.MAP_SIDE:
                minimap[j] = _pack(dict(shown=1, visited=visited, clear=clear, current=current, icon=icon), V3.MAP_BITS)
    # bodies
    bodies = raw['bodies']
    if not bodies:
        raise ValueError('visible-v3: no body')
    entities, overflow = v3_prioritize(raw['entities'], bodies)
    P = np.zeros((len(bodies), V3.PF), '<f4')
    local = np.zeros((len(bodies), V3.LOCAL_SIDE, V3.LOCAL_SIDE, V3.ROOM_WORDS), '<f4')
    padded = np.zeros((V3.CANVAS_H + 8, V3.CANVAS_W + 8, V3.ROOM_WORDS), '<f4')
    padded[4:4 + V3.CANVAS_H, 4:4 + V3.CANVAS_W] = canvas
    for i, b in enumerate(bodies):
        p = P[i]
        known_pos = int(b.get('position_known', 1))
        x, y = float(b['x']) - ox, float(b['y']) - oy
        for k in _P_RAW:
            p[_P[k]] = float(b.get(k, 0))
        if known_pos:
            p[_P['x']], p[_P['y']] = x, y
        else:   # (schema batch 3: no position in this room -- x, y, vx, vy 0 and the local patch 0)
            p[_P['vx']] = p[_P['vy']] = 0.0
        p[_P['position_known']] = known_pos
        if unknown:
            for k in _HEALTH:
                p[_P[k]] = 0.0
        p[_P['body_index']] = i
        p[_P['linked_body']] = float(b.get('linked') or 0)
        p[_P['stall_share']] = stall_share
        p[_P['grid_w']], p[_P['grid_h']] = gw, gh
        p[_P['health_known']] = 0.0 if unknown else 1.0
        p[_P['controller_known']] = b['controller_known']
        p[_P['animation_known']] = b['animation_known']
        p[_P['weapon_charge_known']] = b['weapon_charge_known']
        p[_P['bone_slots_known']] = b['bone_slots_known']
        p[_P['stall_known']] = stall_known
        p[_P['ptype']] = int(b['ptype']) + 2
        p[_P['body_role']] = b['role']
        p[_P['controller']] = int(b['controller']) + 2 if b['controller_known'] else 0
        p[_P['animation']] = b['animation']
        p[_P['stage']] = raw['stage']
        p[_P['stage_type']] = int(raw['stage_type']) + 1
        p[_P['room_type']] = raw['room_type']
        p[_P['difficulty']] = int(raw['difficulty']) + 1
        p[_P['game_version']] = raw['game_version']
        p[_P['control_scheme']] = raw['control_scheme']
        black, bone, weapons = int(b['black']), int(b['bone_bits']), int(b['weapon_bits'])
        for k in range(32):
            p[_P[f'curse{k}']] = (curses >> k) & 1
        for k in range(24):
            p[_P[f'black_slot{k}']] = 0 if unknown else (black >> k) & 1
            p[_P[f'bone_slot{k}']] = 0 if unknown else (bone >> k) & 1
            p[_P[f'weapon{k}']] = (weapons >> k) & 1
        for k, c in enumerate(V3.ENTITY_CLASSES):
            p[_P[f'overflow_{c}']] = overflow[k]
        if known_pos:
            col = min(max(_math.floor(x / V3.CELL), 0), gw - 1)
            row = min(max(_math.floor(y / V3.CELL), 0), gh - 1)
            local[i] = padded[row:row + V3.LOCAL_SIDE, col:col + V3.LOCAL_SIDE]
    # entities
    E = np.zeros((len(entities), V3.EF), '<f4')
    for i, e in enumerate(entities):
        r = E[i]
        for k in _E_RAW:
            r[_E[k]] = float(e.get(k, 0))
        r[_E['x']], r[_E['y']] = float(e['x']) - ox, float(e['y']) - oy
        r[_E['kind']] = int(e['kind']) + 1
        if int(e['kind']) == 3:   # a laser's end point in the canvas frame
            r[_E['l_ex']], r[_E['l_ey']] = float(e['l_ex']) - ox, float(e['l_ey']) - oy
        tear = sum(int(w) << (16 * k) for k, w in enumerate(e['tear']))
        proj = sum(int(w) << (16 * k) for k, w in enumerate(e['proj']))
        eflags = sum(int(w) << (16 * k) for k, w in enumerate(e['eflags']))
        for k, w in enumerate(V3.flag_words(V3.tear_canonical(tear), V3.TEAR_WORDS)):
            r[_E[f'tear_w{k}']] = w
        for k, w in enumerate(V3.flag_words(proj & V3.PROJ_VISIBLE_MASK, V3.PROJ_WORDS)):
            r[_E[f'proj_w{k}']] = w
        for k, w in enumerate(V3.flag_words(eflags & V3.ENTITY_VISIBLE_MASK, V3.EFLAG_WORDS)):
            r[_E[f'eflag_w{k}']] = w
    # inventory and slots
    N = np.zeros((len(raw['inventory']), V3.NF), '<f4')
    for i, (owner, item, count) in enumerate(raw['inventory']):
        N[i, _N['owner']], N[i, _N['id']], N[i, _N['count']], N[i, _N['id_known']] = owner, item, count, 1
    S = np.zeros((len(raw['slots']), V3.SF), '<f4')
    for i, sl in enumerate(raw['slots']):
        for k in ('owner', 'family', 'slot', 'id', 'occupied', 'selected', 'charge', 'max_charge', 'pill_colour',
                  'battery', 'golden', 'giant'):
            S[i, _S[k]] = float(sl.get(k, 0))
        S[i, _S['id_known']] = sl.get('id_known', 1)
        S[i, _S['selected_known']] = sl.get('selected_known', 1)
        S[i, _S['charge_known']] = sl.get('charge_known', 1)
    counts = (len(bodies), len(entities), len(N), len(S))
    size = V3.frame_floats(*counts)
    header = np.array([V3.MAGIC, size, *counts, V3.LAYOUT_GEN, producer], '<f4')
    out = np.concatenate([header, canvas.reshape(-1), minimap, P.reshape(-1), local.reshape(-1), E.reshape(-1),
                          N.reshape(-1), S.reshape(-1)]).astype('<f4')
    if out.size != size or out.size * 4 > V3.FRAME_BYTES:
        raise ValueError(f'visible-v3: frame of {out.size} floats (expected {size}, budget {V3.FRAME_BYTES} bytes)')
    return out.tobytes()


def unpack_frame_v3(payload):
    """The v3 frame bytes -> dict of arrays (header, canvas [16, 28, 5], minimap [169], P [p, 192], local
    [p, 9, 9, 5], E, N, S) -- the reference reader for tests and fixtures (package A decodes on its own)."""
    f = np.frombuffer(payload, '<f4')
    if int(f[0]) != V3.MAGIC or int(f[6]) != V3.LAYOUT_GEN:
        raise ValueError('visible-v3: bad header')
    p, e, n, s = (int(v) for v in f[2:6])
    if int(f[1]) != V3.frame_floats(p, e, n, s) or f.size < int(f[1]):
        raise ValueError('visible-v3: size mismatch')
    at = V3.HEADER
    out = dict(header=f[:V3.HEADER])
    for name, count, shape in (('canvas', 1, (V3.CANVAS_H, V3.CANVAS_W, V3.ROOM_WORDS)),
                               ('minimap', 1, (V3.MAP_SIDE * V3.MAP_SIDE,)), ('P', p, (V3.PF,)),
                               ('local', p, (V3.LOCAL_SIDE, V3.LOCAL_SIDE, V3.ROOM_WORDS)), ('E', e, (V3.EF,)),
                               ('N', n, (V3.NF,)), ('S', s, (V3.SF,))):
        k = count * int(np.prod(shape))
        out[name] = f[at:at + k].reshape((count,) + shape if name not in ('canvas', 'minimap') else shape)
        at += k
    return out
