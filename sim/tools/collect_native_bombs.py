"""Isolated ordinary Isaac bomb calibration; never modifies the installed Mod."""
import argparse, json
from pathlib import Path
from isaac_bridge.parallel import prepare_worker
from isaac_bridge.training import IsaacTrainingEnv
from isaac_bridge.turbo import launch_suspended, post_close, wait_process

ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--extra',action='store_true');ap.add_argument('--layouts',action='store_true');args=ap.parse_args()
root=args.out.resolve()
runtime,profile,_=prepare_worker(root/'worker',Path('D:/Steam/steamapps/common/The Binding of Isaac Rebirth'),Path.home()/'Documents/My Games/Binding of Isaac Repentance+')
mod=runtime/'mods/isaac_rl_bridge/main.lua'
extra='''
info.bombs={}
for _,e in ipairs(Isaac.GetRoomEntities()) do
 local b=e:ToBomb()
 if b then info.bombs[#info.bombs+1]={id=b.Index,age=b.FrameCount,pos=vec(b.Position),vel=vec(b.Velocity),
born=b:GetData().born,damage=b.ExplosionDamage,radius=b.RadiusMultiplier,size=b.Size,
 friction=b.Friction,mass=b.Mass,anim=b:GetSprite():GetAnimation(),aframe=b:GetSprite():GetFrame(),dead=b:IsDead()} end
end
'''
text=mod.read_text(encoding='utf8').replace('\tlocal ok, seedstr = pcall',extra+'\n\tlocal ok, seedstr = pcall')
text=text.replace('local function player_record(p)',"mod:AddCallback(ModCallbacks.MC_POST_BOMB_INIT,function(_,b) b:GetData().born={pos=vec(b.Position),vel=vec(b.Velocity)} end)\nlocal function player_record(p)")
mod.write_text(text,encoding='utf8')
pid,ctl,_=launch_suspended(27143,game_dir=str(runtime),worker_profile=str(profile),extra_args=('--luadebug','--set-stage=1'),skip_render=True,virtual_clock=False,font_guard=False,file_retry=False,probe_dump=False,log_dir=str(root/'log'),engine_velocity=True)
print('LAUNCHED',pid,flush=True)
env=IsaacTrainingEnv(port=27143,action_repeat=1,connect_timeout=90)
try:
 env.connect();env.reset_monstro()
 with (root/'native.jsonl').open('w',encoding='utf8') as out:
  cases=['stationary','moving','held','boss_ground','boss_air','player_radius_80','player_radius_100','rock']
  if args.extra:cases=['moving','release']+[f'player_radius_{d}' for d in (84,85,89,90,94,95)]+[f'boss_radius_{d}' for d in (99,100,109,110,114,115)]+['free_velocity']
  if args.layouts:cases=['layout_1010','layout_1012','layout_1037','layout_1038']
  for case in cases:
   env.reset_monstro()
   if case.startswith('layout_'):
    env.exec('goto s.boss.'+case.rsplit('_',1)[1])
    for tick in range(4):
     obs,_,_,_,_=env.step([0,0,0,0],repeat=1)
     out.write(json.dumps(dict(case=case,tick=tick,obs=obs,info=env.query_info()))+'\n')
    out.flush();print('COLLECTED',case,obs['room'],flush=True);continue
   env.exec('lua for _,e in ipairs(Isaac.GetRoomEntities()) do if e.Type==20 then e:Remove() end end; local p=Isaac.GetPlayer(0); p.Position=Vector(320,280); p.Velocity=Vector(0,0); p:AddBombs(5-p:GetNumBombs())')
   if case=='moving':
    for _ in range(12):env.step([3,0,0,0],repeat=1)
   if case in ('boss_ground','boss_air'):
    state=4 if case=='boss_ground' else 7
    anim='Walk' if case=='boss_ground' else 'JumpDown'
    frame=24 if case=='boss_ground' else 10
    env.exec(f'lua local n=Isaac.Spawn(20,0,0,Vector(360,280),Vector(0,0),nil):ToNPC(); n:ClearEntityFlags(EntityFlag.FLAG_APPEAR); n.State={state}; n:GetSprite():SetFrame("{anim}",{frame}); n:AddEntityFlags(EntityFlag.FLAG_FREEZE); n.EntityCollisionClass={4 if state==4 else 0}; local b=Isaac.Spawn(4,0,0,Vector(320,280),Vector(0,0),Isaac.GetPlayer(0)):ToBomb(); b:SetExplosionCountdown(1); Isaac.GetPlayer(0).Position=Vector(100,380)')
   if case.startswith('player_radius_'):
    distance=int(case.rsplit('_',1)[1])
    env.exec(f'lua local b=Isaac.Spawn(4,0,0,Vector(320,280),Vector(0,0),Isaac.GetPlayer(0)):ToBomb(); b:SetExplosionCountdown(1); Isaac.GetPlayer(0).Position=Vector({320+distance},280)')
   if case.startswith('boss_radius_'):
    distance=int(case.rsplit('_',1)[1])
    env.exec(f'lua local n=Isaac.Spawn(20,0,0,Vector({320+distance},280),Vector(0,0),nil):ToNPC(); n:ClearEntityFlags(EntityFlag.FLAG_APPEAR); n.State=4; n:GetSprite():SetFrame("Walk",24); n:AddEntityFlags(EntityFlag.FLAG_FREEZE); n.EntityCollisionClass=4; local b=Isaac.Spawn(4,0,0,Vector(320,280),Vector(0,0),Isaac.GetPlayer(0)):ToBomb(); b:SetExplosionCountdown(1); Isaac.GetPlayer(0).Position=Vector(100,380)')
   if case=='free_velocity':
    env.exec('lua Isaac.Spawn(4,0,0,Vector(250,280),Vector(5,0),Isaac.GetPlayer(0)); Isaac.GetPlayer(0).Position=Vector(100,380)')
   if case=='rock':
    env.exec('lua Game():GetRoom():SpawnGridEntity(68,GridEntityType.GRID_ROCK,0,1,0); local b=Isaac.Spawn(4,0,0,Vector(320,280),Vector(0,0),Isaac.GetPlayer(0)):ToBomb(); b:SetExplosionCountdown(1); Isaac.GetPlayer(0).Position=Vector(100,380)')
   for tick in range(80 if case in ('stationary','moving','held','release','free_velocity') else 5):
    action=[3 if case=='moving' else 0,0,int(case in ('stationary','moving') and tick==0 or case=='held'),0]
    if case=='release':action[2]=int(tick%2==0)
    obs,reward,done,trunc,info=env.step(action,repeat=1)
    out.write(json.dumps(dict(case=case,tick=tick,action=action,obs=obs,info=env.query_info()))+'\n')
   out.flush();print('COLLECTED',case,flush=True)
finally:
 env.reset_safe();env.close();ctl.close();post_close(pid)
 code=wait_process(pid,40);(root/'exit.json').write_text(json.dumps(dict(pid=pid,exit_code=code)),encoding='utf8');print('EXIT',code,flush=True)
