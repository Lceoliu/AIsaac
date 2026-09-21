"""Record motion from an isolated, original J460 engine. No gameplay patches.

The sole mod is the existing TCP bridge plus extra privileged calibration fields.
Room changes and player placement happen only before each input sequence.
"""
import argparse, json, random
from pathlib import Path
from isaac_bridge.parallel import prepare_worker
from isaac_bridge.training import IsaacTrainingEnv
from isaac_bridge.turbo import launch_suspended, post_close, wait_process

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--suite',choices=['calibration','heldout'],default='calibration')
args=parser.parse_args()
ROOT=args.out.resolve()
runtime, profile, savedata = prepare_worker(ROOT/'worker',
    Path('D:/Steam/steamapps/common/The Binding of Isaac Rebirth'),
    Path.home()/'Documents/My Games/Binding of Isaac Repentance+')
mod = runtime/'mods/isaac_rl_bridge/main.lua'
text = mod.read_text(encoding='utf-8')
text = text.replace('\tlocal ok, seedstr = pcall', Path(__file__).with_name('calibration_info.lua').read_text(encoding='utf-8')+'\n\tlocal ok, seedstr = pcall')
mod.write_text(text, encoding='utf-8')
pid, ctl, output = launch_suspended(27141, game_dir=str(runtime), worker_profile=str(profile),
    extra_args=('--luadebug', '--set-stage=1'), skip_render=True, virtual_clock=False,
    font_guard=False, file_retry=False, probe_dump=False, log_dir=str(ROOT/'log'),
    engine_velocity=True)
print('LAUNCHED', pid, flush=True)
env = IsaacTrainingEnv(port=27141, action_repeat=1, connect_timeout=90)
cases = {
    'axis_stop': [(3,0,35),(0,0,25)],
    'reverse': [(3,0,25),(7,0,35),(0,0,20)],
    'turn': [(3,0,20),(1,0,20),(7,0,20),(5,0,20),(0,0,20)],
    'diagonal_wall': [(2,0,90),(4,0,30),(0,0,20)],
    'stationary_fire': [(0,2,70)],
    'side_fire': [(1,2,24),(5,2,40),(0,2,30)],
    'forward_back_fire': [(3,2,25),(7,2,45),(0,2,30)],
}
positions={}
if args.suite=='heldout':
    cases={}
    rng=random.Random(20260921)
    for move in range(9):
        for shoot in range(1,5):
            name=f'move{move}_shoot{shoot}'
            cases[name]=[(move,shoot,17),(8-move,shoot,13),(0,shoot,15)]
    for i in range(6):
        cases[f'bursts_{i}']=[(rng.randrange(9),rng.randrange(5),rng.randint(1,13)) for _ in range(65)]+[(0,0,35)]
    for shoot,pos in [(1,(320,400)),(2,(100,280)),(3,(320,160)),(4,(540,280))]:
        name=f'range_{shoot}'
        cases[name]=[(0,shoot,550)]
        positions[name]=pos
try:
    env.connect()
    env.reset_monstro()
    with (ROOT/'native.jsonl').open('w', encoding='utf-8') as out:
        for name, segments in cases.items():
            env.reset_monstro()
            env.exec('lua for _,e in ipairs(Isaac.GetRoomEntities()) do if e.Type~=1 then e:Remove() end end; for i=0,7 do Game():GetRoom():RemoveDoor(i) end')
            for _ in range(45): env.step({}, repeat=1)
            x,y=positions.get(name,(320,280))
            # Clear-room rewards are delayed, and may include a troll bomb.
            # Remove them after settling, before any calibration input begins.
            env.exec('lua for _,e in ipairs(Isaac.GetRoomEntities()) do if e.Type~=1 then e:Remove() end end')
            env.step({}, repeat=1)  # Entity::Remove is deferred, not synchronous.
            env.exec(f'lua local p=Isaac.GetPlayer(0); p.Position=Vector({x},{y}); p.Velocity=Vector(0,0); p.FireDelay=-1')
            initial = {'case':name, 'initial':True, 'obs':env.query_obs(), 'info':env.query_info()}
            assert not initial['info']['npcs']
            assert not initial['obs']['entities']
            assert not initial['obs']['doors']
            grid={g[0]:g[4] for g in initial['obs']['grid']}
            assert all(grid.get(i)==4 for i in range(135) if i%15 in (0,14) or i//15 in (0,8)),grid
            out.write(json.dumps(initial)+'\n'); out.flush()
            for move, shoot, length in segments:
                for _ in range(length):
                    action = [move,shoot,0,0]
                    obs, reward, terminated, truncated, info = env.step(action,repeat=1)
                    assert obs['room']['room_idx']==initial['obs']['room']['room_idx']
                    assert obs['players'][0]['hearts']==6
                    out.write(json.dumps({'case':name,'action':action,'obs':obs,'info':env.query_info()})+'\n')
            out.flush()
            print('COLLECTED',name, flush=True)
finally:
    env.reset_safe()
    env.close()
    ctl.close()
    post_close(pid)
    exit_code=wait_process(pid,40)
    (ROOT/'exit.json').write_text(json.dumps({'pid':pid,'exit_code':exit_code}),encoding='utf-8')
    print('EXIT',exit_code,flush=True)
