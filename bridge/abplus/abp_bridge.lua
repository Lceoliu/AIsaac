--[[
AbpRLBridge abp-0.2.1 — IsaacRLBridge 0.2.0 (J460, ../mod/isaac_rl_bridge/main.lua) ported to
The Binding of Isaac: Afterbirth+ v1.06.T1 (native Linux, Steam build 22878971).

Same line protocol and observation schema (combat_schema=3), so the Python side
(isaac_bridge/env.py, training.py, monstro_gym.py, transformer_obs.py) is reused unchanged;
isaac_bridge/abplus.py only adds the AB+ launcher and arena reset.

Loading: not a mod folder (the instance's mods path is blocked to stop the Workshop sync).
The analysis runtime copy's resources/scripts/main.lua runs dofile(ABP_LUA) when the game is
started with --luadebug (analysis/docs/ABP_LINUX_REVERSE_ENGINEERING.md §6, §7).

AB+ differences handled here:
  * Sprite:GetAnimation does not exist in AB+. The current animation of Monstro (type 20) and
    bombs (type 4) — the only kinds whose animation reaches the policy — is found with
    IsPlaying/IsFinished over their anm2 animation names; other kinds report nil.
  * EntityPlayer.TearRange does not exist (AB+ has TearHeight/TearFallingSpeed). range is
    reported Repentance-equivalent: 260 * TearHeight / -23.75, exact for base Isaac (the
    fixed curriculum character) and proportional otherwise.
  * main.lua can run in a Lua state that is discarded during startup. The TCP port is bound
    lazily on the first MC_POST_UPDATE of the live state, so a discarded state never holds it.
  * The console has no lua or rewind command (restart exists). {"cmd":"lua","code":"..."} runs a
    chunk (arena setup); the server listens on 127.0.0.1 only.
  * No rewind means a dead player cannot be restored. While a client is connected, a hit that
    would kill the player is cancelled in MC_ENTITY_TAKE_DMG, counted as the final damage event
    and reported as players[i].dead = true, so the episode ends at the same lethal hit and the
    next reset starts from a living player (ISAAC_RL_BLOCK_LETHAL=0 disables this).

Protocol (one JSON object per line), identical to 0.2.0 plus "lua", "format" and "profile":
  game -> client:  {"type":"obs","event":"step|reset|query","seq":n,"obs":{...}}
                   {"type":"ok"} / {"type":"info","info":{...}} / {"type":"error","msg":"..."}
  client -> game:  {"cmd":"step","move":0-8,"shoot":0-4,"bomb":0|1,"item":0|1,"pill":0|1,"drop":0|1,"repeat":k}
                   {"cmd":"reset","commands":["goto s.boss.1010"],"settle":2,"wait_room":true}
                   {"cmd":"exec","command":"..."}  {"cmd":"lua","code":"..."}  {"cmd":"info"}
                   {"cmd":"obs"}  {"cmd":"control","enabled":true|false}  {"cmd":"close"}
                   {"cmd":"format","version":1|2,"validate":bool}  {"cmd":"profile","n":k}
  format 2 (abp-0.2.0): an observation becomes the line "B <bytes> <event> <seq>" followed by
  <bytes> of binary payload (see pack_obs; decoder python/isaac_bridge/abplus_obs.py). Every new
  client starts at format 1.
  abp-0.2.1: combat gains blocking_hp (summed HP of the living NPCs that keep the doors shut,
  Entity:CanShutDoors, the engine's clear condition) and blocking_points (their daily-run kill
  points, ceil(5 * MaxHitPoints^0.2) as in ScoreSheet::AddKilledEnemy); both are current values,
  not counters. Used by combat-v2 (python/isaac_bridge/abplus_reward.py).
  move: 0 stop 1 up 2 up-right 3 right 4 down-right 5 down 6 down-left 7 left 8 up-left;
  shoot: 0 none 1 up 2 right 3 down 4 left.
]]

local VERSION = "abp-0.2.1"
local mod = RegisterMod("AbpRLBridge", 1)

-- The game ships LuaSocket for both architectures; make sure the 64-bit core is found.
if package and package.cpath then
	package.cpath = "resources/scripts/linux64/?.so;" .. package.cpath
	package.path = "resources/scripts/?.lua;" .. package.path
end
local json = require("json")
local socket_ok, socket = pcall(require, "socket")

local function getenv(name)
	if os and os.getenv then
		local ok, v = pcall(os.getenv, name)
		if ok then return v end
	end
	return nil
end

local cfg = {
	host = "127.0.0.1",
	port = tonumber(getenv("ISAAC_RL_PORT")) or 27015,
	privileged = (getenv("ISAAC_RL_PRIVILEGED") == "1"),
	engine_velocity = (getenv("ISAAC_RL_ENGINE_VEL") == "1"),
	block_lethal = (getenv("ISAAC_RL_BLOCK_LETHAL") ~= "0"),
	default_repeat = 4,
	effect_whitelist = { [1] = true, [22] = true, [23] = true, [26] = true, [30] = true, [45] = true, [46] = true,
		[51] = true, [53] = true, [55] = true },
}

-- anm2 animation names (analysis/resources/animations-a/anm2/020.000_monstro.anm2; bombs as
-- reported by the Repentance bridge and the Rust simulator).
-- Numeric entity types: the bomb enum is ENTITY_BOMBDROP in AB+ and ENTITY_BOMB in Repentance.
local TYPE_BOMB, TYPE_MONSTRO = 4, 20
local ANIMATIONS = {
	[TYPE_MONSTRO] = { "Walk", "JumpUp", "JumpDown", "Taunt", "Appear", "Death" },
	[TYPE_BOMB] = { "Pulse", "Idle", "Explode" },
}

local A = ButtonAction
local CONTROLLED = {
	[A.ACTION_LEFT] = true, [A.ACTION_RIGHT] = true, [A.ACTION_UP] = true, [A.ACTION_DOWN] = true,
	[A.ACTION_SHOOTLEFT] = true, [A.ACTION_SHOOTRIGHT] = true, [A.ACTION_SHOOTUP] = true, [A.ACTION_SHOOTDOWN] = true,
	[A.ACTION_BOMB] = true, [A.ACTION_ITEM] = true, [A.ACTION_PILLCARD] = true, [A.ACTION_DROP] = true,
}
local MOVE = {
	[0] = {}, [1] = { A.ACTION_UP }, [2] = { A.ACTION_UP, A.ACTION_RIGHT }, [3] = { A.ACTION_RIGHT },
	[4] = { A.ACTION_RIGHT, A.ACTION_DOWN }, [5] = { A.ACTION_DOWN }, [6] = { A.ACTION_DOWN, A.ACTION_LEFT },
	[7] = { A.ACTION_LEFT }, [8] = { A.ACTION_LEFT, A.ACTION_UP },
}
local SHOOT = {
	[0] = {}, [1] = { A.ACTION_SHOOTUP }, [2] = { A.ACTION_SHOOTRIGHT }, [3] = { A.ACTION_SHOOTDOWN }, [4] = { A.ACTION_SHOOTLEFT },
}

local state = {
	server = nil, client = nil, seq = 0, bind_attempted = false,
	control = true,
	held = {}, triggered = {},
	frames_left = 0,
	pending_reset = false, room_ready = false, settle = 0,
	stats = { steps = 0, resets = 0 },
	logic_frames = 0,
	events = { damage = 0, tears = 0, npc_deaths = 0, clears = 0 },
	combat = { player_damage_events = 0, player_damage = 0, enemy_damage_events = 0,
		enemy_damage = 0, enemy_damage_fraction = 0, blocking_hp = 0, blocking_points = 0 },
	health = { players = {}, npcs = {}, deaths = {} },
	lethal = {},             -- player index -> true once a lethal hit was cancelled (virtual death)
}

local function zero_events()
	state.events = { damage = 0, tears = 0, npc_deaths = 0, clears = 0 }
	state.combat = { player_damage_events = 0, player_damage = 0, enemy_damage_events = 0,
		enemy_damage = 0, enemy_damage_fraction = 0, blocking_hp = 0, blocking_points = 0 }
	state.health = { players = {}, npcs = {}, deaths = {} }
	state.lethal = {}
end

local function entity_key(e) return tostring(e.Index) .. ':' .. tostring(e.InitSeed) end

local function record_enemy_loss(previous, hp)
	local loss = math.max(0, previous.hp - math.max(0, hp))
	if loss > 0 then
		state.combat.enemy_damage_events = state.combat.enemy_damage_events + 1
		state.combat.enemy_damage = state.combat.enemy_damage + loss
		state.combat.enemy_damage_fraction = state.combat.enemy_damage_fraction + loss / previous.max_hp
	end
end

-- Settled health every logic frame (same definition as 0.2.0).
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
	local blocking_hp, blocking_points = 0, 0
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if e:ToNPC() and e:CanShutDoors() and not e:IsDead() then
			blocking_hp = blocking_hp + math.max(0, e.HitPoints)
			blocking_points = blocking_points + math.ceil(5 * math.max(0, e.MaxHitPoints) ^ 0.2)
		end
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
	state.combat.blocking_hp = blocking_hp
	state.combat.blocking_points = blocking_points
end

local function log(msg)
	Isaac.DebugString("[AbpRLBridge] " .. tostring(msg))
end

------------------------------------------------------------------ network

local function send(tbl)
	if not state.client then return false end
	local ok, payload = pcall(json.encode, tbl)
	if not ok then log("encode failed: " .. tostring(payload)); return false end
	local sent, err = state.client:send(payload .. "\n")
	if not sent then
		log("send failed: " .. tostring(err))
		state.client:close(); state.client = nil
		return false
	end
	return true
end

local function disconnect(reason)
	log("client disconnected: " .. tostring(reason))
	if state.client then state.client:close() end
	state.client = nil
	state.held = {}; state.triggered = {}; state.frames_left = 0; state.pending_reset = false
end

local function try_bind()
	state.bind_attempted = true
	if not socket_ok then
		log("luasocket unavailable (" .. tostring(socket) .. "); start the game with --luadebug. Bridge disabled.")
		return
	end
	local server, err = socket.bind(cfg.host, cfg.port, 1)
	if not server then
		log("bind failed on " .. cfg.host .. ":" .. cfg.port .. " (" .. tostring(err) .. "). Bridge disabled.")
		return
	end
	server:settimeout(0)
	state.server = server
	log("listening on " .. cfg.host .. ":" .. cfg.port .. " (" .. VERSION .. ")")
end

local function try_accept()
	if not state.server then return end
	local c = state.server:accept()
	if c then
		c:settimeout(nil)
		c:setoption("tcp-nodelay", true)
		state.client = c
		state.obs_format, state.obs_validate, state.terrain_dirty = 1, false, true
		log("client connected on port " .. cfg.port)
		send({ type = "hello", version = VERSION, engine = "abplus-1.06", port = cfg.port,
			privileged = cfg.privileged, game_frame = Game():GetFrameCount(), logic_frames = state.logic_frames })
	end
end

------------------------------------------------------------------ actions

local function apply_action(cmd)
	local held = {}
	for _, a in ipairs(MOVE[tonumber(cmd.move) or 0] or {}) do held[a] = true end
	for _, a in ipairs(SHOOT[tonumber(cmd.shoot) or 0] or {}) do held[a] = true end
	if tonumber(cmd.bomb) == 1 then held[A.ACTION_BOMB] = true end
	if tonumber(cmd.item) == 1 then held[A.ACTION_ITEM] = true end
	if tonumber(cmd.pill) == 1 then held[A.ACTION_PILLCARD] = true end
	if tonumber(cmd.drop) == 1 then held[A.ACTION_DROP] = true end
	local trig = {}
	for a in pairs(held) do
		if not state.held[a] then trig[a] = true end
	end
	state.held = held
	state.triggered = trig
end

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, action)
	if not state.client or not state.control then return nil end
	if entity == nil then return nil end
	if not CONTROLLED[action] then return nil end
	local player = entity:ToPlayer()
	if not player then return nil end
	if hook == InputHook.IS_ACTION_PRESSED then
		return state.held[action] == true
	elseif hook == InputHook.IS_ACTION_TRIGGERED then
		return state.triggered[action] == true
	elseif hook == InputHook.GET_ACTION_VALUE then
		return state.held[action] and 1.0 or 0.0
	end
	return nil
end)

------------------------------------------------------------------ observation

local function sprite_anim(spr, etype)
	local names = ANIMATIONS[etype]
	if not names then return nil end
	for _, name in ipairs(names) do
		local ok, current = pcall(function() return spr:IsPlaying(name) or spr:IsFinished(name) end)
		if ok and current then return name end
	end
	return nil
end

local function vec(v) return { v.X, v.Y } end

local function tear_range(p)
	local ok, height = pcall(function() return p.TearHeight end)
	if ok and type(height) == "number" and height < 0 then return 260 * height / -23.75 end
	return 260
end

local function player_record(p)
	local spr = p:GetSprite()
	local rec = {
		id = p.Index, pos = vec(p.Position), size = p.Size,
		hearts = p:GetHearts(), max_hearts = p:GetMaxHearts(), soul = p:GetSoulHearts(), black = p:GetBlackHearts(),
		bone = p:GetBoneHearts(), eternal = p:GetEternalHearts(), golden = p:GetGoldenHearts(), lives = p:GetExtraLives(),
		coins = p:GetNumCoins(), bombs = p:GetNumBombs(), keys = p:GetNumKeys(),
		damage = p.Damage, fire_delay_max = p.MaxFireDelay, shot_speed = p.ShotSpeed, range = tear_range(p),
		speed = p.MoveSpeed, luck = p.Luck, can_fly = p.CanFly,
		active = p:GetActiveItem(), active_charge = p:GetActiveCharge(),
		active_ready = p:GetActiveItem() ~= 0 and not p:NeedsCharge(),
		invulnerable = p:GetDamageCooldown() > 0, controls = p.ControlsEnabled,
		head_dir = p:GetHeadDirection(), fire_dir = p:GetFireDirection(), move_dir = p:GetMovementDirection(),
		anim = nil, aframe = spr:GetFrame(), flip = p.FlipX,
		dead = p:IsDead() or state.lethal[p.Index] == true, ptype = p:GetPlayerType(),
	}
	if cfg.engine_velocity then rec.vel = vec(p.Velocity) end
	return rec
end

local function laser_record(e)
	local laser = e:ToLaser()
	local rec = { circle = false, radius = 0, angle = 0, length = 0, width = 2 * e.Size, ["end"] = vec(e.Position), samples = {} }
	pcall(function()
		rec.circle = laser:IsCircleLaser(); rec.radius = laser.Radius; rec.angle = laser.AngleDegrees
		rec.length = laser.LaserLength; rec["end"] = vec(laser:GetEndPoint())
		local samples = laser:GetSamples()
		for i = 0, #samples - 1 do rec.samples[#rec.samples + 1] = vec(samples:Get(i)) end
	end)
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
		coll = e.EntityCollisionClass, gcoll = e.GridCollisionClass, cdmg = e.CollisionDamage,
		anim = sprite_anim(spr, t), aframe = spr:GetFrame(), flip = e.FlipX, age = e.FrameCount,
	}
	if cfg.engine_velocity then rec.vel = vec(e.Velocity) end
	if t == EntityType.ENTITY_TEAR then
		local tear = e:ToTear()
		rec.height = tear.Height; rec.fall = tear.FallingSpeed; rec.scale = tear.Scale
	elseif t == EntityType.ENTITY_PROJECTILE then
		local pr = e:ToProjectile()
		rec.projectile = true
		rec.height = pr.Height; rec.fall = pr.FallingSpeed; rec.scale = pr.Scale
	elseif t == EntityType.ENTITY_LASER then
		rec.laser = laser_record(e)
	elseif t == TYPE_BOMB then
		rec.bomb = true
	elseif t == EntityType.ENTITY_PICKUP then
		rec.pickup = true
	else
		local npc = e:ToNPC()
		if npc then
			rec.enemy = e:IsEnemy(); rec.vulnerable = e:IsVulnerableEnemy(); rec.boss = e:IsBoss()
			rec.champion = npc:GetChampionColorIdx()
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

local function terrain_record(room, player, static_only)
	-- static_only (binary v2): spikes only; the client adds hazard effects from the entity list.
	local cells = {}
	local hazards = {}
	for _, e in ipairs(static_only and {} or Isaac.GetRoomEntities()) do
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
		local hazard = spikes
		for _, e in ipairs(hazards) do
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
		combat_schema = 3, engine = "abplus-1.06",
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
		enemy_damage = state.combat.enemy_damage, enemy_damage_fraction = state.combat.enemy_damage_fraction,
		blocking_hp = state.combat.blocking_hp, blocking_points = state.combat.blocking_points }
	return obs
end

local function build_info()
	local game = Game()
	local info = { npcs = {}, players = {} }
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		local npc = e:ToNPC()
		if npc then
			info.npcs[#info.npcs + 1] = { id = e.Index, type = e.Type, variant = e.Variant, hp = e.HitPoints, max_hp = e.MaxHitPoints,
				state = npc.State, state_frame = npc.StateFrame, pcool = npc.ProjectileCooldown, pdelay = npc.ProjectileDelay,
				i1 = npc.I1, i2 = npc.I2, v1 = vec(npc.V1), v2 = vec(npc.V2), vel = vec(e.Velocity), visible = e.Visible }
		end
	end
	for i = 0, game:GetNumPlayers() - 1 do
		local p = Isaac.GetPlayer(i)
		info.players[#info.players + 1] = { id = p.Index, fire_delay = p.FireDelay, vel = vec(p.Velocity),
			damage_cooldown = p:GetDamageCooldown(), total_damage_taken = p:GetTotalDamageTaken() }
	end
	local ok, seedstr = pcall(function() return game:GetSeeds():GetStartSeedString() end)
	if ok then info.start_seed = seedstr end
	local ok2, desc = pcall(function() return game:GetLevel():GetCurrentRoomDesc() end)
	if ok2 and desc then info.room_spawn_seed = desc.SpawnSeed; info.room_clear_count = desc.ClearCount end
	return info
end

------------------------------------------------------------------ binary observation (v2)
-- Same content as build_obs(), packed with string.pack instead of JSON. The terrain and grid
-- (static within a room) are sent only when they may have changed: on reset/query events, after
-- lua/exec/reset commands and room changes, or when a cell's grid collision changes. Frame on the
-- wire: "B <bytes> <event> <seq>\n" followed by the payload. Layout: python/isaac_bridge/abplus_obs.py.

local OBS_MAGIC = 0x32504241 -- "ABP2"
local spack = string.pack
local terrain_cache = { sig = nil, version = 0 }

local function I(x)
	local v = math.tointeger(x)
	if v == nil then error("integer field expected, got " .. tostring(x)) end
	return v
end

local function B(x) return x and 1 or 0 end

-- string.pack with the offending record spelled out when a value does not fit its format.
local function packf(fmt, ...)
	local ok, result = pcall(spack, fmt, ...)
	if ok then return result end
	local values = {}
	for i = 1, select("#", ...) do
		local v = select(i, ...)
		values[#values + 1] = tostring(v) .. ":" .. (math.type(v) or type(v))
	end
	error(tostring(result) .. " | " .. fmt .. " | " .. table.concat(values, " "), 2)
end

local function terrain_signature(room, player)
	local parts = { tostring(player.CanFly), tostring(player.Size) }
	for i = 0, room:GetGridSize() - 1 do parts[#parts + 1] = room:GetGridCollision(i) end
	return table.concat(parts, ",")
end

local function pack_laser(e)
	local l = laser_record(e)
	local parts = { packf("<Bddddddi8", B(l.circle), l.radius, l.angle, l.length, l.width, l["end"][1], l["end"][2],
		#l.samples) }
	for _, s in ipairs(l.samples) do parts[#parts + 1] = packf("<dd", s[1], s[2]) end
	return table.concat(parts)
end

local function pack_entity(e)
	local t = e.Type
	if t == EntityType.ENTITY_PLAYER then return nil end
	if t == EntityType.ENTITY_EFFECT and not cfg.effect_whitelist[e.Variant] then return nil end
	if not e.Visible then return nil end
	local spr = e:GetSprite()
	local anim = sprite_anim(spr, t)
	local kind, extra = 0, ""
	if t == EntityType.ENTITY_TEAR then
		local x = e:ToTear()
		kind, extra = 1, packf("<ddd", x.Height, x.FallingSpeed, x.Scale)
	elseif t == EntityType.ENTITY_PROJECTILE then
		local x = e:ToProjectile()
		kind, extra = 2, packf("<ddd", x.Height, x.FallingSpeed, x.Scale)
	elseif t == EntityType.ENTITY_LASER then
		kind, extra = 3, pack_laser(e)
	elseif t == TYPE_BOMB then
		kind = 4
	elseif t == EntityType.ENTITY_PICKUP then
		kind = 5
	else
		local npc = e:ToNPC()
		if npc then
			local boss = e:IsBoss()
			local has_hp = boss and e.MaxHitPoints > 0
			kind, extra = 6, packf("<BBBi8Bd", B(e:IsEnemy()), B(e:IsVulnerableEnemy()), B(boss),
				I(npc:GetChampionColorIdx()), B(has_hp), has_hp and e.HitPoints / e.MaxHitPoints or 0)
		end
	end
	local vel = cfg.engine_velocity and packf("<Bdd", 1, e.Velocity.X, e.Velocity.Y) or packf("<B", 0)
	return packf("<i8i8i8i8dddddi8i8dBs1i8Bi8B", I(e.Index), I(t), I(e.Variant), I(e.SubType),
		e.Position.X, e.Position.Y, e.Size, e.SizeMulti.X, e.SizeMulti.Y,
		I(e.EntityCollisionClass), I(e.GridCollisionClass), e.CollisionDamage,
		B(anim ~= nil), anim or "", I(spr:GetFrame()), B(e.FlipX), I(e.FrameCount), kind) .. vel .. extra
end

local function pack_player(p)
	local vel = cfg.engine_velocity and packf("<Bdd", 1, p.Velocity.X, p.Velocity.Y) or packf("<B", 0)
	return packf("<i8ddd" .. "i8i8i8i8i8i8i8i8i8i8i8" .. "dddddd" .. "Bi8i8B" .. "BB" .. "i8i8i8" .. "i8B" .. "Bi8",
		I(p.Index), p.Position.X, p.Position.Y, p.Size,
		I(p:GetHearts()), I(p:GetMaxHearts()), I(p:GetSoulHearts()), I(p:GetBlackHearts()), I(p:GetBoneHearts()),
		I(p:GetEternalHearts()), I(p:GetGoldenHearts()), I(p:GetExtraLives()), I(p:GetNumCoins()),
		I(p:GetNumBombs()), I(p:GetNumKeys()),
		p.Damage, p.MaxFireDelay, p.ShotSpeed, tear_range(p), p.MoveSpeed, p.Luck,
		B(p.CanFly), I(p:GetActiveItem()), I(p:GetActiveCharge()), B(p:GetActiveItem() ~= 0 and not p:NeedsCharge()),
		B(p:GetDamageCooldown() > 0), B(p.ControlsEnabled),
		I(p:GetHeadDirection()), I(p:GetFireDirection()), I(p:GetMovementDirection()),
		I(p:GetSprite():GetFrame()), B(p.FlipX),
		B(p:IsDead() or state.lethal[p.Index] == true), I(p:GetPlayerType())) .. vel
end

local function pack_obs(event)
	local game = Game()
	local room, level = game:GetRoom(), game:GetLevel()
	local player0 = Isaac.GetPlayer(0)
	local c = state.combat
	local tl, br = room:GetTopLeftPos(), room:GetBottomRightPos()
	local parts = {
		packf("<I4I4I4Bi8i8i8i8", OBS_MAGIC, I(state.logic_frames), I(game:GetFrameCount()), B(game:IsPaused()),
			I(state.events.damage), I(state.events.tears), I(state.events.npc_deaths), I(state.events.clears)),
		packf("<ddddddd", c.player_damage_events, c.player_damage, c.enemy_damage_events, c.enemy_damage,
			c.enemy_damage_fraction, c.blocking_hp, c.blocking_points),
		packf("<i8i8i8i8ddddBi8i8i8i8i8i8", I(room:GetType()), I(room:GetRoomShape()), I(room:GetGridWidth()),
			I(room:GetGridHeight()), tl.X, tl.Y, br.X, br.Y, B(room:IsClear()), I(room:GetAliveEnemiesCount()),
			I(room:GetFrameCount()), I(level:GetStage()), I(level:GetStageType()), I(level:GetCurses()),
			I(level:GetCurrentRoomIndex())),
	}
	local variant, name, subtype
	pcall(function()
		local desc = level:GetCurrentRoomDesc()
		if desc and desc.Data then variant = desc.Data.Variant; name = desc.Data.Name; subtype = desc.Data.Subtype end
	end)
	if variant ~= nil then
		parts[#parts + 1] = packf("<Bi8i8s2", 1, I(variant), I(subtype), tostring(name))
	else
		parts[#parts + 1] = packf("<B", 0)
	end
	local doors = {}
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		if d then
			doors[#doors + 1] = packf("<BBBddi8", slot, B(d:IsOpen()), B(d:IsLocked()), d.Position.X, d.Position.Y,
				I(d.TargetRoomType))
		end
	end
	parts[#parts + 1] = packf("<B", #doors)
	for _, d in ipairs(doors) do parts[#parts + 1] = d end
	local sig = terrain_signature(room, player0)
	if event ~= "step" or state.terrain_dirty or sig ~= terrain_cache.sig then
		terrain_cache.sig, state.terrain_dirty = sig, false
		terrain_cache.version = terrain_cache.version + 1
		local cells = grid_records(room)
		parts[#parts + 1] = packf("<I4Bs4s4", terrain_cache.version, 1,
			json.encode(terrain_record(room, player0, true)), json.encode(cells))
	else
		parts[#parts + 1] = packf("<I4B", terrain_cache.version, 0)
	end
	local n = game:GetNumPlayers()
	parts[#parts + 1] = packf("<B", n)
	for i = 0, n - 1 do parts[#parts + 1] = pack_player(Isaac.GetPlayer(i)) end
	local ents = {}
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		local rec = pack_entity(e)
		if rec then ents[#ents + 1] = rec end
	end
	parts[#parts + 1] = packf("<I2", #ents)
	for _, rec in ipairs(ents) do parts[#parts + 1] = rec end
	return table.concat(parts)
end

local function send_raw(data)
	if not state.client then return false end
	local sent, err = state.client:send(data)
	if not sent then
		log("send failed: " .. tostring(err))
		state.client:close(); state.client = nil
		return false
	end
	return true
end

local function send_obs(event)
	state.seq = state.seq + 1
	if state.obs_format == 2 then
		if state.obs_validate then
			-- Validation: the JSON message first, then the binary frame of the same state.
			local msg = { type = "obs", event = event, seq = state.seq, obs = build_obs() }
			if not send(msg) then return false end
		end
		local ok, payload = pcall(pack_obs, event)
		if not ok then
			log("pack failed: " .. tostring(payload))
			return send({ type = "error", msg = "pack: " .. tostring(payload) })
		end
		return send_raw("B " .. #payload .. " " .. event .. " " .. state.seq .. "\n" .. payload)
	end
	local msg = { type = "obs", event = event, seq = state.seq, obs = build_obs() }
	if cfg.privileged then msg.info = build_info() end
	return send(msg)
end

------------------------------------------------------------------ command loop

local function do_reset(cmd)
	state.held = {}; state.triggered = {}; state.frames_left = 0
	state.terrain_dirty = true
	state.stats.resets = state.stats.resets + 1
	zero_events()
	state.pending_reset = true
	state.room_ready = (cmd.wait_room == false)
	state.settle = tonumber(cmd.settle) or 2
	for _, c in ipairs(cmd.commands or {}) do
		Isaac.ExecuteCommand(tostring(c))
	end
end

local function run_lua(code)
	local chunk, err = load(tostring(code or ""), "=abp_lua", "t")
	if not chunk then return false, err end
	return pcall(chunk)
end

local function wait_command()
	while state.client do
		local line, err = state.client:receive("*l")
		if not line then disconnect(err); return end
		local ok, cmd = pcall(json.decode, line)
		if not ok or type(cmd) ~= "table" then
			send({ type = "error", msg = "bad json" })
		elseif cmd.cmd == "step" then
			apply_action(cmd)
			state.frames_left = math.max(1, math.floor(tonumber(cmd["repeat"]) or cfg.default_repeat))
			state.stats.steps = state.stats.steps + 1
			return
		elseif cmd.cmd == "reset" then
			do_reset(cmd)
			return
		elseif cmd.cmd == "exec" then
			state.terrain_dirty = true
			Isaac.ExecuteCommand(tostring(cmd.command or ""))
			send({ type = "ok", cmd = "exec" })
		elseif cmd.cmd == "lua" then
			state.terrain_dirty = true
			local lua_ok, result = run_lua(cmd.code)
			if lua_ok then
				send({ type = "ok", cmd = "lua", result = result ~= nil and tostring(result) or nil })
			else
				send({ type = "error", msg = "lua: " .. tostring(result) })
			end
		elseif cmd.cmd == "format" then
			-- 1: JSON observations (default for every new client); 2: binary v2 (pack_obs);
			-- validate: send the JSON observation too, before each binary one.
			state.obs_format = (tonumber(cmd.version) == 2) and 2 or 1
			state.obs_validate = cmd.validate == true
			state.terrain_dirty = true
			send({ type = "ok", cmd = "format", version = state.obs_format, validate = state.obs_validate })
		elseif cmd.cmd == "info" then
			send({ type = "info", info = build_info() })
		elseif cmd.cmd == "profile" then
			-- Cost of each observation component in the current game state (os.clock CPU seconds).
			local n = math.max(1, math.floor(tonumber(cmd.n) or 20))
			local game, room, player = Game(), Game():GetRoom(), Isaac.GetPlayer(0)
			local t, ms = nil, {}
			local function timed(name, f)
				t = os.clock()
				for _ = 1, n do f() end
				ms[name] = 1000 * (os.clock() - t) / n
			end
			timed("players", function() player_record(player) end)
			timed("entities", function()
				for _, e in ipairs(Isaac.GetRoomEntities()) do entity_record(e) end
			end)
			timed("room", function() room_record(game) end)
			timed("grid_doors", function() grid_records(room) end)
			timed("terrain", function() terrain_record(room, player) end)
			local obs = build_obs()
			timed("json_all", function() json.encode(obs) end)
			timed("json_terrain", function() json.encode(obs.terrain) end)
			timed("json_entities", function() json.encode(obs.entities) end)
			timed("build_all", build_obs)
			send({ type = "ok", cmd = "profile", ms = ms, entities = #obs.entities,
				bytes = #json.encode(obs) })
		elseif cmd.cmd == "obs" then
			send_obs("query")
		elseif cmd.cmd == "control" then
			state.control = (cmd.enabled ~= false)
			if not state.control then state.held = {}; state.triggered = {} end
			send({ type = "ok", cmd = "control", enabled = state.control })
		elseif cmd.cmd == "close" then
			disconnect("close requested")
			return
		else
			send({ type = "error", msg = "unknown cmd" })
		end
	end
end

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
	state.room_ready = true
	state.terrain_dirty = true
end)

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, from_save)
	state.room_ready = true
end)

local function health_units(p)
	-- Half-heart units that absorb damage before death (Isaac curriculum: red and soul hearts).
	return p:GetHearts() + p:GetSoulHearts() + p:GetEternalHearts() + 2 * p:GetBoneHearts()
end

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount, flags, source, countdown)
	local p = entity and entity:ToPlayer()
	if not p then return nil end
	state.events.damage = state.events.damage + 1
	if cfg.block_lethal and state.client and p:GetExtraLives() == 0 and amount >= health_units(p) then
		if not state.lethal[p.Index] then
			state.lethal[p.Index] = true
			state.combat.player_damage_events = state.combat.player_damage_events + 1
			state.combat.player_damage = state.combat.player_damage + amount
		end
		return false
	end
	return nil
end)
mod:AddCallback(ModCallbacks.MC_POST_FIRE_TEAR, function(_, tear)
	state.events.tears = state.events.tears + 1
end)
mod:AddCallback(ModCallbacks.MC_POST_NPC_DEATH, function(_, npc)
	state.events.npc_deaths = state.events.npc_deaths + 1
	state.health.deaths[entity_key(npc)] = true
end)
mod:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD, function(_, rng, pos)
	state.events.clears = state.events.clears + 1
	return nil
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
	state.logic_frames = state.logic_frames + 1
	update_combat()
	if not state.bind_attempted then try_bind() end
	if not state.server then return end
	if not state.client then
		try_accept()
		if not state.client then return end
	end
	state.triggered = {}
	if state.pending_reset then
		if not state.room_ready then return end
		state.settle = state.settle - 1
		if state.settle > 0 then return end
		state.pending_reset = false
		if send_obs("reset") then wait_command() end
		return
	end
	if state.frames_left > 0 then state.frames_left = state.frames_left - 1 end
	if state.frames_left == 0 then
		if send_obs("step") then wait_command() end
	end
end)

log("loaded " .. VERSION .. " port " .. cfg.port .. " (binds on the first logic frame)")
