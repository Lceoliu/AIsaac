info.monstro = { bosses={}, projectiles={} }
for _,e in ipairs(Isaac.GetRoomEntities()) do
 if e.Type==20 then
  local n=e:ToNPC(); local s=e:GetSprite()
  info.monstro.bosses[#info.monstro.bosses+1]={id=e.Index,pos=vec(e.Position),vel=vec(e.Velocity),
   target=vec(n.TargetPosition),offset=vec(e.PositionOffset),sprite_offset=vec(s.Offset),
   anim=s:GetAnimation(),frame=s:GetFrame(),filename=s:GetFilename(),
   jump=s:IsEventTriggered('Jump'),land=s:IsEventTriggered('Land'),shoot=s:IsEventTriggered('Shoot'),
   jumped=s:WasEventTriggered('Jump'),landed=s:WasEventTriggered('Land'),finished=s:IsFinished(s:GetAnimation()),
   state=n.State,i1=n.I1,collision=e.EntityCollisionClass,grid_collision=e.GridCollisionClass,
   hp=e.HitPoints,visible=e.Visible}
 end
 local p=e:ToProjectile()
 if p then info.monstro.projectiles[#info.monstro.projectiles+1]={id=p.Index,age=p.FrameCount,
  pos=vec(p.Position),vel=vec(p.Velocity),height=p.Height,fall=p.FallingSpeed,accel=p.FallingAccel,
  offset=vec(p.PositionOffset),scale=p.Scale,size=p.Size,dead=p:IsDead(),flags=tostring(p.ProjectileFlags)} end
end
