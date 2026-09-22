"""Visible observation adapter for the Rust Isaac/Monstro curriculum.

No game process, assets or hidden target/RNG fields are needed by the actor.
The same VisibleHistory encoder is used by the original-engine environment.
"""

def rust_visible_observation(state):
    p=state['player']
    entities=[]
    for source,kind,size in [('bosses',20,40),('tears',2,7),('projectiles',9,5),('bombs',4,16)]:
        for e in state[source]:
            if e.get('dead',False):continue
            row=dict(id=e['id'],type=kind,variant=0,subtype=0,age=e['age'],
                     pos=e['pos'],size=e.get('size',size),size_multi=[1,1],
                     anim=e['anim'] if kind in (20,4) else '',
                     aframe=e['frame'] if kind in (20,4) else 0,flip=e.get('flip',False),
                     height=e.get('height',0),scale=e.get('scale',1),
                     coll=e['collision'],gcoll=e['gcoll'],
                     cdmg=e['cdmg'],enemy=kind==20,boss=kind==20)
            if kind==20:row['boss_hp']=max(0,e['hp'])/250
            entities.append(row)
    return dict(combat_schema=3,logic_frames=state['frame'],
                room=dict(top_left=[60,140],bottom_right=[580,420]),doors=state['doors'],
                terrain=state['terrain'],entities=entities,
                players=[dict(pos=p['pos'],size=10,hearts=state['hp'],max_hearts=6,soul=0,
                              bombs=p['bombs'],keys=0,coins=0,damage=3.5,speed=1,shot_speed=1,
                              fire_delay_max=10,range=260,can_fly=False,active_charge=0,
                              active_ready=False,active=0,anim='',aframe=0,flip=False)])
