local p = Isaac.GetPlayer(0)
info.calibration = {
 recent=vec(p:GetRecentMovementVector()), before=vec(p:GetVelocityBeforeUpdate()),
 inherit_right=vec(p:GetTearMovementInheritance(Vector(1,0))),
 tears_offset=vec(p.TearsOffset), tear_height=p.TearHeight,
 tear_falling_speed=p.TearFallingSpeed, tear_falling_accel=p.TearFallingAcceleration,
 friction=p.Friction, tears={}
}
for _,e in ipairs(Isaac.GetRoomEntities()) do
 local t=e:ToTear()
 if t then
  info.calibration.tears[#info.calibration.tears+1]={
   id=t.Index, age=t.FrameCount,dead=t:IsDead(),pos=vec(t.Position),vel=vec(t.Velocity),
   height=t.Height,fall=t.FallingSpeed,accel=t.FallingAcceleration,
   flags=tostring(t.TearFlags),offset=vec(t.PositionOffset),
   displacement=vec(t.PosDisplacement),parent_offset=vec(t.ParentOffset),
   tear_index=t.TearIndex,damage=t.CollisionDamage,size=t.Size}
 end
end
