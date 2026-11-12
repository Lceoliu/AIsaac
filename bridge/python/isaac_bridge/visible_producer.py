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
