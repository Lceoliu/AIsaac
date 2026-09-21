"""Visible observation adapter for the Rust *fixed empty Isaac/Monstro room*.

No game process, assets or hidden target/RNG fields are needed by the actor.
This is an observation adapter, not the future high-throughput batched worker.
"""

def rust_visible_observation(state):
    p=state['player']
    entities=[]
    for source,kind,size in [('bosses',20,40),('tears',2,7),('projectiles',9,5)]:
        for e in state[source]:
            if e.get('dead',False):continue
            row=dict(id=e['id'],type=kind,variant=0,subtype=0,age=e['age'],
                     pos=e['pos'],size=e.get('size',size),size_multi=[1,1],
                     anim=e['anim'] if kind==20 else 'RegularTear6',
                     aframe=e['frame'] if kind==20 else 0,flip=e.get('flip',False),
                     height=e.get('height',0),scale=e.get('scale',1),
                     coll=e.get('collision',4),gcoll=(3 if e['collision']==0 else 5) if kind==20 else 4,
                     cdmg=3.5 if kind==2 else 1,enemy=kind==20,boss=kind==20)
            if kind==20:row['boss_hp']=max(0,e['hp'])/250
            entities.append(row)
    cells=[]
    for i in range(135):
        row,col=divmod(i,15);inside=0<row<8 and 0<col<14
        cells.append([i,40+40*col,120+40*row,0 if inside else 4,int(inside),int(inside),0,0,int(not inside),0])
    return dict(combat_schema=3,logic_frames=state['frame'],
                room=dict(top_left=[60,140],bottom_right=[580,420]),doors=[],
                terrain=dict(width=15,height=9,cells=cells),entities=entities,
                players=[dict(pos=p['pos'],size=10,hearts=state['hp'],max_hearts=6,soul=0,
                              bombs=0,keys=0,coins=0,damage=3.5,speed=1,shot_speed=1,
                              fire_delay_max=10,range=260,can_fly=False,active_charge=0,
                              active_ready=False,active=0,anim='',aframe=0,flip=False)])
