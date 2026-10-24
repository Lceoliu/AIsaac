"""Isolated J460 Monstro traces. Hidden AI fields are calibration-only info."""
import argparse,json
from pathlib import Path
from isaac_bridge.parallel import prepare_worker
from isaac_bridge.training import IsaacTrainingEnv
from isaac_bridge.turbo import launch_suspended,post_close,wait_process

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--suite',choices=['actions','collision'],default='actions')
args=parser.parse_args()
root=args.out.resolve()
runtime,profile,_=prepare_worker(root/'worker',Path('D:/Steam/steamapps/common/The Binding of Isaac Rebirth'),Path.home()/'Documents/My Games/Binding of Isaac Repentance+')
mod=runtime/'mods/isaac_rl_bridge/main.lua'
text=mod.read_text(encoding='utf8')
text=text.replace('\tlocal ok, seedstr = pcall',Path(__file__).with_name('monstro_info.lua').read_text(encoding='utf8')+'\n\tlocal ok, seedstr = pcall')
mod.write_text(text,encoding='utf8')
pid,ctl,_=launch_suspended(27141,game_dir=str(runtime),worker_profile=str(profile),extra_args=('--luadebug','--set-stage=1'),skip_render=True,virtual_clock=False,font_guard=False,file_retry=False,probe_dump=False,log_dir=str(root/'log'),engine_velocity=True)
print('LAUNCHED',pid,flush=True)
env=IsaacTrainingEnv(port=27141,action_repeat=1,connect_timeout=90)
try:
 env.connect(); env.reset_monstro()
 with (root/'native.jsonl').open('w',encoding='utf8') as out:
  for case in (('stationary','moving','retarget') if args.suite=='actions' else ('low','overhead','floor')):
   env.reset_monstro()
   if args.suite=='collision':
    h={'low':-30,'overhead':-70,'floor':-5}[case]
    pos='Isaac.GetPlayer(0).Position' if case!='floor' else 'Vector(200,280)'
    env.exec(f'lua local q=Isaac.Spawn(9,0,0,{pos},Vector(0,0),nil):ToProjectile(); q.Height={h}; q.FallingSpeed=0; q.FallingAccel=0')
    for tick in range(6):
     obs,_,_,_,_=env.step([0,0,0,0],repeat=1)
     out.write(json.dumps(dict(case=case,tick=tick,obs=obs,info=env.query_info()))+'\n')
    out.flush();print('COLLECTED',case,flush=True)
    continue
   # Immunity only prevents termination; native movement and AI remain untouched.
   env.exec('lua Isaac.GetPlayer(0):SetMinDamageCooldown(30000)')
   for tick in range(900):
    if case=='retarget' and tick%20==0:
     x=180 if tick%40==0 else 460
     env.exec(f'lua Isaac.GetPlayer(0).Position=Vector({x},380); Isaac.GetPlayer(0).Velocity=Vector(0,0)')
    action=[(3 if tick%120<60 else 7) if case=='moving' else 0,0,0,0]
    obs,_,_,_,_=env.step(action,repeat=1)
    info=env.query_info()
    assert 'monstro' in info, info
    out.write(json.dumps(dict(case=case,tick=tick,action=action,obs=obs,info=info))+'\n')
   out.flush(); print('COLLECTED',case,flush=True)
finally:
 env.reset_safe();env.close();ctl.close();post_close(pid)
 code=wait_process(pid,40)
 (root/'exit.json').write_text(json.dumps(dict(pid=pid,exit_code=code)),encoding='utf8')
 print('EXIT',code,flush=True)
