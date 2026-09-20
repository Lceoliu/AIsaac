--[[
IsaacRLBridge 0.2.0  —  原版 Repentance+ (J460) 的同步训练桥接 mod，已接入实机单房间 PPO。
  0.2.0：观测附带 logic_frames（本 mod 数到的 MC_POST_UPDATE 次数）与 events（自上次 reset 起的受伤/发射眼泪/
  敌人死亡/清房计数），供 L1 加速可行性实验做"步进精确"与"加速等价"比对（rl/docs/L1_FEASIBILITY_PLAN.md）。

工作方式
  * 游戏每个逻辑帧 (30 Hz) 结束时触发 MC_POST_UPDATE；到达决策边界后本 mod 把"玩家可观察"观测以一行 JSON
    发给 TCP 客户端，然后阻塞等待下一条指令。游戏主线程因此停在 Game::Update 之后，等价于同步单步。
  * 动作通过 MC_INPUT_ACTION 注入：只对玩家实体、只对受控的 ButtonAction 返回值，其余返回 nil 交还原版。
  * 重置通过 Isaac.ExecuteCommand 执行控制台命令 (restart / seed / stage / goto ...)，并等待 MC_POST_NEW_ROOM。
  * 需要 --luadebug 启动参数：require("socket") 依赖游戏自带的 resources/scripts/socket/core.dll。

协议 (每行一个 JSON 对象)
  游戏 -> 客户端:  {"type":"obs","event":"step|reset|query","seq":n,"obs":{...}}
                   {"type":"ok"} / {"type":"info","info":{...}} / {"type":"error","msg":"..."}
  客户端 -> 游戏:  {"cmd":"step","move":0-8,"shoot":0-4,"bomb":0|1,"item":0|1,"pill":0|1,"drop":0|1,"repeat":k}
                   {"cmd":"reset","commands":["restart"],"settle":2,"wait_room":true}
                   {"cmd":"exec","command":"goto d.5"}
                   {"cmd":"info"}            -- 特权真值 (敌人 HP、AI 状态、RNG)，只供评估/校准，不进策略
                   {"cmd":"obs"}             -- 立即重发当前观测
                   {"cmd":"control","enabled":true|false}
                   {"cmd":"close"}
  move: 0 停 1 上 2 右上 3 右 4 右下 5 下 6 左下 7 左 8 左上；shoot: 0 不射 1 上 2 右 3 下 4 左。
]]

local mod = RegisterMod("IsaacRLBridge", 1)
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
	privileged = (getenv("ISAAC_RL_PRIVILEGED") == "1"),      -- obs 附带 info 真值 (仅评估/校准)
	engine_velocity = (getenv("ISAAC_RL_ENGINE_VEL") == "1"), -- 实体附带引擎速度 (默认关闭，速度由历史估计)
	default_repeat = 4,
	-- 允许进入观测的 EffectVariant：爆炸/尖刺/各类地面毒液等对走位有影响且屏幕可见的效果
	effect_whitelist = { [1] = true, [22] = true, [23] = true, [26] = true, [30] = true, [45] = true, [46] = true,
		[51] = true, [53] = true, [55] = true },
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
	server = nil, client = nil, seq = 0,
	control = true,          -- 受控时键盘对玩家无效
	held = {}, triggered = {},
	frames_left = 0,         -- 当前动作还要保持的逻辑帧数；0 表示到达决策边界
	pending_reset = false, room_ready = false, settle = 0,
	stats = { steps = 0, resets = 0 },
	logic_frames = 0,        -- 本 mod 数到的 MC_POST_UPDATE 次数（与 Game():GetFrameCount() 交叉核对步进）
	events = { damage = 0, tears = 0, npc_deaths = 0, clears = 0 },   -- 自上次 reset 起的可见事件计数
	combat = { player_damage_events = 0, player_damage = 0, enemy_damage_events = 0,
		enemy_damage = 0, enemy_damage_fraction = 0 },
	health = { players = {}, npcs = {}, deaths = {} },
}

local function zero_events()
	state.events = { damage = 0, tears = 0, npc_deaths = 0, clears = 0 }
	state.combat = { player_damage_events = 0, player_damage = 0, enemy_damage_events = 0,
		enemy_damage = 0, enemy_damage_fraction = 0 }
	state.health = { players = {}, npcs = {}, deaths = {} }
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

local function log(msg)
	Isaac.DebugString("[IsaacRLBridge] " .. tostring(msg))
end

------------------------------------------------------------------ 网络

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

local function try_accept()
	if not state.server then return end
	local c = state.server:accept()
	if c then
		c:settimeout(nil)      -- 阻塞：游戏主线程等待训练器
		c:setoption("tcp-nodelay", true)
		state.client = c
		log("client connected on port " .. cfg.port)
		send({ type = "hello", version = "0.2.0", port = cfg.port, privileged = cfg.privileged,
			game_frame = Game():GetFrameCount(), logic_frames = state.logic_frames })
	end
end

------------------------------------------------------------------ 动作

local function apply_action(cmd)
	local held = {}
	for _, a in ipairs(MOVE[tonumber(cmd.move) or 0] or {}) do held[a] = true end
	for _, a in ipairs(SHOOT[tonumber(cmd.shoot) or 0] or {}) do held[a] = true end
	if tonumber(cmd.bomb) == 1 then held[A.ACTION_BOMB] = true end
	if tonumber(cmd.item) == 1 then held[A.ACTION_ITEM] = true end
	if tonumber(cmd.pill) == 1 then held[A.ACTION_PILLCARD] = true end
	if tonumber(cmd.drop) == 1 then held[A.ACTION_DROP] = true end
	-- 边沿：本次新按下的键在下一逻辑帧报告一次 triggered
	local trig = {}
	for a in pairs(held) do
		if not state.held[a] then trig[a] = true end
	end
	state.held = held
	state.triggered = trig
end

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, action)
	if not state.client or not state.control then return nil end
	if entity == nil then return nil end                 -- 菜单/非实体查询不干预
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

------------------------------------------------------------------ 观测

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
		coll = e.EntityCollisionClass, gcoll = e.GridCollisionClass, cdmg = e.CollisionDamage,
		anim = sprite_anim(spr), aframe = spr:GetFrame(), flip = e.FlipX, age = e.FrameCount,
	}
	if cfg.engine_velocity then rec.vel = vec(e.Velocity) end
	if t == EntityType.ENTITY_TEAR then
		local tear = e:ToTear()
		rec.height = tear.Height; rec.fall = tear.FallingSpeed; rec.scale = tear.Scale
	elseif t == EntityType.ENTITY_PROJECTILE then
		local pr = e:ToProjectile()
		rec.projectile = true
		rec.height = pr.Height; rec.fall = pr.FallingSpeed; rec.scale = pr.Scale
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
		cells[#cells + 1] = { i, pos.X, pos.Y, collision, inside and 1 or 0,
			walkable and 1 or 0, pit and 1 or 0, spikes and 1 or 0 }
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

-- 特权真值：敌人 HP/AI 状态/冷却、玩家射击冷却、种子。只用于奖励、校准和评估。
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

local function send_obs(event)
	state.seq = state.seq + 1
	local msg = { type = "obs", event = event, seq = state.seq, obs = build_obs() }
	if cfg.privileged then msg.info = build_info() end
	return send(msg)
end

------------------------------------------------------------------ 指令循环

local function do_reset(cmd)
	state.held = {}; state.triggered = {}; state.frames_left = 0
	state.stats.resets = state.stats.resets + 1
	zero_events()
	-- 先武装等待状态，再执行命令：goto 可能在 ExecuteCommand 内同步触发 MC_POST_NEW_ROOM
	state.pending_reset = true
	state.room_ready = (cmd.wait_room == false)
	state.settle = tonumber(cmd.settle) or 2
	for _, c in ipairs(cmd.commands or {}) do
		Isaac.ExecuteCommand(tostring(c))
	end
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
			Isaac.ExecuteCommand(tostring(cmd.command or ""))
			send({ type = "ok", cmd = "exec" })
		elseif cmd.cmd == "info" then
			send({ type = "info", info = build_info() })
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
end)

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function(_, from_save)
	state.room_ready = true
end)

-- 可见事件计数：只计数，不改变任何返回值（返回 nil 交还原版）
if ModCallbacks.MC_ENTITY_TAKE_DMG then
	mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount, flags, source, countdown)
		if entity and entity:ToPlayer() then state.events.damage = state.events.damage + 1 end
		return nil
	end)
end
if ModCallbacks.MC_POST_FIRE_TEAR then
	mod:AddCallback(ModCallbacks.MC_POST_FIRE_TEAR, function(_, tear)
		state.events.tears = state.events.tears + 1
	end)
end
if ModCallbacks.MC_POST_NPC_DEATH then
	mod:AddCallback(ModCallbacks.MC_POST_NPC_DEATH, function(_, npc)
		state.events.npc_deaths = state.events.npc_deaths + 1
		state.health.deaths[entity_key(npc)] = true
	end)
end
if ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD then
	mod:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD, function(_, rng, pos)
		state.events.clears = state.events.clears + 1
		return nil
	end)
end

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
	state.logic_frames = state.logic_frames + 1
	update_combat()
	if not state.server then return end
	if not state.client then
		try_accept()
		if not state.client then return end
	end
	-- 上一帧新按下的键已被本帧 Update 消费，清掉边沿标记
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

------------------------------------------------------------------ 启动

if not socket_ok then
	log("luasocket unavailable (" .. tostring(socket) .. "); start the game with --luadebug. Bridge disabled.")
else
	local server, err = socket.bind(cfg.host, cfg.port, 1)
	if not server then
		log("bind failed on " .. cfg.host .. ":" .. cfg.port .. " (" .. tostring(err) .. "). Bridge disabled.")
	else
		server:settimeout(0)   -- accept 不阻塞，直到有客户端
		state.server = server
		log("listening on " .. cfg.host .. ":" .. cfg.port)
	end
end
