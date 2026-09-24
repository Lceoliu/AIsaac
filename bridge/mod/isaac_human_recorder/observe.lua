-- Visible schema 3 copied from IsaacRLBridge; keep actor filtering in VisibleHistory.
return function(state)
local cfg = {engine_velocity=false, effect_whitelist={ [1]=true,[22]=true,[23]=true,[26]=true,[30]=true,[45]=true,[46]=true,[51]=true,[53]=true,[55]=true }}
local function entity_key(e) return tostring(e.Index) .. ':' .. tostring(e.InitSeed) end

local function record_enemy_loss(previous, hp)
	local loss = math.max(0, previous.hp - math.max(0, hp))
	if loss > 0 then
		state.combat.enemy_damage_events = state.combat.enemy_damage_events + 1
		state.combat.enemy_damage = state.combat.enemy_damage + loss
		state.combat.enemy_damage_fraction = state.combat.enemy_damage_fraction + loss / previous.max_hp
	end
end

-- Read settled health every logic frame, not attempted damage in a pre-damage
-- callback. A damage event means one entity's confirmed health loss in one frame;
-- simultaneous hits on that entity in that frame are one settlement, not N hits.
local function update_combat()
	local game = Game()
	for i = 0, game:GetNumPlayers() - 1 do
		local p = Isaac.GetPlayer(i)
		local total = p:GetTotalDamageTaken()
		local previous = state.health.players[i]
		if previous and total > previous then
			state.combat.player_damage_events = state.combat.player_damage_events + 1
			state.combat.player_damage = state.combat.player_damage + total - previous
		end
		state.health.players[i] = total
	end
	local current = {}
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if e:ToNPC() and e:IsEnemy() and e.MaxHitPoints > 0 then
			local key = entity_key(e)
			local previous = state.health.npcs[key]
			local hp = e:IsDead() and 0 or math.max(0, e.HitPoints)
			if previous then record_enemy_loss(previous, hp) end
			current[key] = { hp = hp, max_hp = previous and previous.max_hp or e.MaxHitPoints }
		end
	end
	for key, previous in pairs(state.health.npcs) do
		if not current[key] and state.health.deaths[key] then record_enemy_loss(previous, 0) end
	end
	state.health.npcs = current
	state.health.deaths = {}
end

local function sprite_anim(spr)
	-- 官方 LuaDocs (J460 随包) 未列出 GetAnimation，Repentance 社区文档有；失败则 nil
	local ok, name = pcall(function() return spr:GetAnimation() end)
	if ok and type(name) == "string" then return name end
	return nil
end

local function vec(v) return { v.X, v.Y } end

local function player_record(p)
	local spr = p:GetSprite()
	local rec = {
		id = p.Index, pos = vec(p.Position), size = p.Size,
		hearts = p:GetHearts(), max_hearts = p:GetMaxHearts(), soul = p:GetSoulHearts(), black = p:GetBlackHearts(),
		bone = p:GetBoneHearts(), eternal = p:GetEternalHearts(), golden = p:GetGoldenHearts(), lives = p:GetExtraLives(),
		coins = p:GetNumCoins(), bombs = p:GetNumBombs(), keys = p:GetNumKeys(),
		damage = p.Damage, fire_delay_max = p.MaxFireDelay, shot_speed = p.ShotSpeed, range = p.TearRange,
		speed = p.MoveSpeed, luck = p.Luck, can_fly = p.CanFly,
		active = p:GetActiveItem(), active_charge = p:GetActiveCharge(),
		active_ready = p:GetActiveItem() ~= 0 and not p:NeedsCharge(),
		invulnerable = p:GetDamageCooldown() > 0, controls = p.ControlsEnabled,
		head_dir = p:GetHeadDirection(), fire_dir = p:GetFireDirection(), move_dir = p:GetMovementDirection(),
		anim = sprite_anim(spr), aframe = spr:GetFrame(), flip = p.FlipX,
		dead = p:IsDead(), ptype = p:GetPlayerType(),
	}
	if cfg.engine_velocity then rec.vel = vec(p.Velocity) end
	return rec
end

local function entity_record(e)
	local t = e.Type
	if t == EntityType.ENTITY_PLAYER then return nil end
	if t == EntityType.ENTITY_EFFECT and not cfg.effect_whitelist[e.Variant] then return nil end
	if not e.Visible then return nil end
	local spr = e:GetSprite()
	local rec = {
		id = e.Index, type = t, variant = e.Variant, subtype = e.SubType,
		pos = vec(e.Position), size = e.Size, size_multi = vec(e.SizeMulti),
		position_offset = vec(e.PositionOffset), sprite_offset = vec(spr.Offset),
		coll = e.EntityCollisionClass, gcoll = e.GridCollisionClass, cdmg = e.CollisionDamage,
		anim = sprite_anim(spr), aframe = spr:GetFrame(), flip = e.FlipX, age = e.FrameCount,
	}
	if cfg.engine_velocity then rec.vel = vec(e.Velocity) end
	if t == EntityType.ENTITY_TEAR then
		local tear = e:ToTear()
		rec.height = tear.Height; rec.fall = tear.FallingSpeed; rec.accel = tear.FallingAcceleration; rec.scale = tear.Scale
	elseif t == EntityType.ENTITY_PROJECTILE then
		local pr = e:ToProjectile()
		rec.projectile = true
		rec.height = pr.Height; rec.fall = pr.FallingSpeed; rec.accel = pr.FallingAccel; rec.scale = pr.Scale
	elseif t == EntityType.ENTITY_LASER then
		local laser = e:ToLaser()
		local points = {}
		local samples = laser:GetSamples()
		for i = 0, #samples - 1 do points[#points + 1] = vec(samples:Get(i)) end
		rec.laser = { circle = laser:IsCircleLaser(), radius = laser.Radius,
			angle = laser.AngleDegrees, length = laser.LaserLength,
			width = 2 * e.Size, ["end"] = vec(laser:GetEndPoint()), samples = points }
	elseif t == EntityType.ENTITY_BOMB then
		rec.bomb = true
	elseif t == EntityType.ENTITY_PICKUP then
		rec.pickup = true
	else
		local npc = e:ToNPC()
		if npc then
			rec.enemy = e:IsEnemy(); rec.vulnerable = e:IsVulnerableEnemy(); rec.boss = e:IsBoss()
			rec.champion = npc:GetChampionColorIdx()
			-- Boss 血条在屏幕上可见；普通敌人的 HP 只进特权 info
			if rec.boss and e.MaxHitPoints > 0 then rec.boss_hp = e.HitPoints / e.MaxHitPoints end
		end
	end
	return rec
end

local function grid_records(room)
	local cells, doors = {}, {}
	local n = room:GetGridSize()
	for i = 0, n - 1 do
		local g = room:GetGridEntity(i)
		if g then
			local gt = g:GetType()
			if gt ~= GridEntityType.GRID_NULL and gt ~= GridEntityType.GRID_DECORATION then
				cells[#cells + 1] = { i, gt, g:GetVariant(), g.State, g.CollisionClass, g.Position.X, g.Position.Y }
			end
		end
	end
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		if d then
			doors[#doors + 1] = { slot = slot, open = d:IsOpen(), locked = d:IsLocked(), pos = vec(d.Position),
				target_type = d.TargetRoomType }
		end
	end
	return cells, doors
end

local function terrain_record(room, player)
	local cells = {}
	local hazards = {}
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if e.Visible and e.Type == EntityType.ENTITY_EFFECT and cfg.effect_whitelist[e.Variant]
			and e.CollisionDamage > 0 then
			hazards[#hazards + 1] = e
		end
	end
	for i = 0, room:GetGridSize() - 1 do
		local pos = room:GetGridPosition(i)
		local collision = room:GetGridCollision(i)
		local inside = room:IsPositionInRoom(pos, player.Size)
		local pit = collision == GridCollisionClass.COLLISION_PIT
		local fly_over = player.CanFly and (pit or collision == GridCollisionClass.COLLISION_OBJECT
			or collision == GridCollisionClass.COLLISION_SOLID)
		local walkable = inside and (collision == GridCollisionClass.COLLISION_NONE
			or collision == GridCollisionClass.COLLISION_WALL_EXCEPT_PLAYER or fly_over)
		local grid = room:GetGridEntity(i)
		local kind = grid and grid:GetType() or GridEntityType.GRID_NULL
		local spikes = kind == GridEntityType.GRID_SPIKES or kind == GridEntityType.GRID_SPIKES_ONOFF
		local solid = collision ~= GridCollisionClass.COLLISION_NONE and not pit
		local destructible = solid and grid and (grid:ToRock() ~= nil or grid:ToPoop() ~= nil or grid:ToTNT() ~= nil)
		local hazard = spikes -- potential spike danger; not a hidden damage-timer read
		for _, e in ipairs(hazards) do
			-- Conservative cell footprint; precise entity geometry remains in obs.entities.
			if pos:Distance(e.Position) <= e.Size + 28.3 then hazard = true end
		end
		cells[#cells + 1] = { i, pos.X, pos.Y, collision, inside and 1 or 0,
			walkable and 1 or 0, pit and 1 or 0, hazard and 1 or 0,
			solid and 1 or 0, destructible and 1 or 0 }
	end
	return { width = room:GetGridWidth(), height = room:GetGridHeight(), cells = cells }
end

local function room_record(game)
	local room = game:GetRoom()
	local level = game:GetLevel()
	local rec = {
		type = room:GetType(), shape = room:GetRoomShape(), gw = room:GetGridWidth(), gh = room:GetGridHeight(),
		top_left = vec(room:GetTopLeftPos()), bottom_right = vec(room:GetBottomRightPos()),
		clear = room:IsClear(), alive = room:GetAliveEnemiesCount(), frame = room:GetFrameCount(),
		stage = level:GetStage(), stage_type = level:GetStageType(), curses = level:GetCurses(),
		room_idx = level:GetCurrentRoomIndex(),
	}
	local ok, desc = pcall(function() return level:GetCurrentRoomDesc() end)
	if ok and desc and desc.Data then
		pcall(function() rec.variant = desc.Data.Variant; rec.name = desc.Data.Name; rec.subtype = desc.Data.Subtype end)
	end
	return rec
end

local function build_obs()
	local game = Game()
	local obs = {
		combat_schema = 3,
		game_frame = game:GetFrameCount(), paused = game:IsPaused(),
		logic_frames = state.logic_frames,
		events = { damage = state.events.damage, tears = state.events.tears,
			npc_deaths = state.events.npc_deaths, clears = state.events.clears },
		players = {}, entities = {},
	}
	for i = 0, game:GetNumPlayers() - 1 do
		obs.players[#obs.players + 1] = player_record(Isaac.GetPlayer(i))
	end
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		local rec = entity_record(e)
		if rec then obs.entities[#obs.entities + 1] = rec end
	end
	obs.room = room_record(game)
	obs.grid, obs.doors = grid_records(game:GetRoom())
	obs.terrain = terrain_record(game:GetRoom(), Isaac.GetPlayer(0))
	obs.combat = { player_damage_events = state.combat.player_damage_events,
		player_damage = state.combat.player_damage, enemy_damage_events = state.combat.enemy_damage_events,
		enemy_damage = state.combat.enemy_damage, enemy_damage_fraction = state.combat.enemy_damage_fraction }
	return obs
end


return {snapshot=build_obs, update=update_combat, key=entity_key}
end
