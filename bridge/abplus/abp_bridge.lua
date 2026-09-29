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
  abp-0.2.2: combat gains blocking_count, the number of those NPCs (combat-v3 prices time by the
  enemies still alive and pays a kill per NPC removed).
  abp-0.2.3: the roster lineage (combat-v5). The setup chunk calls AbpRosterMark(mode), which takes
  the doors-blocking NPCs alive at that moment as the roster. Death successors join by frame and
  place (AB+ leaves their SpawnerEntity nil; mode, see LINEAGE_RADIUS); NPCs spawned by a living
  one never join. Morph keeps the entity, so it stays in the lineage (one that follows a death: abp-0.2.8). The HP a lineage NPC loses is counted up to the HP it had when it
  joined (regrowth earns nothing new) into combat.lineage_damage; its death adds 1 to
  combat.lineage_kills. combat also reports lineage_count and lineage_hp (the alive lineage NPCs),
  and every NPC record carries lineage and blocking (CanShutDoors) flags.
  abp-0.2.4 (the hit-rate test, user spec 2026-09-26): the setup chunk calls AbpSetInvincible(on);
  an invincible player takes no damage while a client is connected (MC_ENTITY_TAKE_DMG returns
  false, combat.blocked_hits counts the cancelled hits). combat.tear_hits counts the player tears
  that damaged a lineage NPC, once per tear; the tears fired are events.tears (MC_POST_FIRE_TEAR).
  EntityRef.Type and SpawnerType are eEntityType userdata whose read aborts AB+ (LuaBridge
  assertion, Userdata.h:376), so the damage source is read through source.Entity. Player tears
  have SpawnerEntity nil; in the curriculum rooms only the player fires tears.
  abp-0.2.5 (user decisions 2026-09-26: every enemy earns reward, misses are penalised):
  lineage mode 3 makes every doors-blocking NPC a lineage member from the first frame it is seen,
  spawns of living NPCs included, with its HP then as its budget, so hits, kills and tear hits
  count on every enemy. Every tear the player fires is followed until it is gone; one that is gone
  without having damaged a lineage NPC is a miss. combat.tear_misses counts them; miss_streak is the
  misses since the last hit (a hit resets it to 0); miss_units adds the streak at every miss
  (1 + 2 + ... over consecutive misses), so a penalty growing linearly per consecutive miss is
  c * delta(miss_units).
  abp-0.2.6 (user decision 2026-09-26): the setup chunk calls AbpSetMissCap(k); with k > 0 a miss adds
  min(miss_streak, k) to miss_units, so the growth stops at the k-th miss in a row (0 = no cap;
  reset turns it off). miss_streak itself is not capped.
  abp-0.2.7 (user decision 2026-09-27, credit at the firing step): every tear the player fires
  records the logic frame it was fired in. The HP a lineage NPC loses is charged to the tears that
  hit it (oldest first, up to each tear's damage), a lineage death to the tear that hit it last
  (within KILL_FRAMES), a miss to the missing tear; anything without a tear is charged to the
  current frame. combat.credits lists, per fire frame, the damage, kills and miss units charged
  since the previous step observation ({fire_frame, damage, kills, miss_units}, sorted by frame);
  a step observation clears it. The cumulative counters are unchanged.
  abp-0.2.8 (user decision 2026-09-28, EXPERIMENTS.md C37): a lineage NPC that dies and lives on as
  the same entity (a Gaper that loses its head becomes a Gusher or Pacer: same Index and InitSeed,
  MC_POST_NPC_DEATH fired) stayed out of the lineage for good, because its key was already gone: hits
  on it earned nothing and tears on it were misses. In lineage modes 1-3 it now joins again from the
  frame after its death, with its HP then as the budget; its own death later is another kill.
  abp-0.2.9 (user request 2026-09-28, EXPERIMENTS.md A6): the duel. The setup chunk spawns one NPC of type
  DUEL_TYPE (a Pacer, 11.1: wanders, never attacks, no death spawns) and calls AbpDuelAttach(npc); from then on the
  step command's duel_move / duel_shoot (the player's move and shoot values) drive it as a second Isaac, calibrated
  to the player (A6): the player's movement law (two physics sub-steps per logic frame), a shot every 11 frames
  (fire delay 10), shots at 10 px/frame plus 1.2 x its velocity, spawned 10 px ahead and 3-5 px to the side
  (alternating), with a tear's range, and a hit radius of the player's (NPC Size 10, shot Size 8.165); 21 HP (six
  3.5 tears; the player dies to six half-heart shots); after damage 29 frames without damage, shots absorbed
  meanwhile (as the player's damage cooldown); a lethal hit is cancelled and the NPC is dead (as the player's);
  tear hits push it (0.3 x the tear's velocity). The NPC plays after its spawn phase only (about 20 frames, the
  engine's FLAG_APPEAR): obs.duel.active. obs.duel also reports, per side (the player, the NPC), shots fired, shots
  that damaged the other side (hits), shots gone without (misses, their run and units, the miss cap applies to
  both), damage dealt (in the other side's units: NPC HP, half hearts) and damage taken (events, amount). A reset
  drops the binding. The binary observation ends with the duel block (a flag byte, 0 without a duel).
  abp-0.2.8-hp (user decision 2026-09-28, EXPERIMENTS.md C39; abp-0.2.8 plus one counter, kept apart from the
  duel's abp-0.2.9 on host 2): combat.monster_damage, the HP every monster in the room lost, the room's own and every
  spawn alike, doors-blocking or not. A monster is an NPC of the engine's active-enemy types
  (Entity::IsActiveEnemy: types 10-999 except shopkeepers 17, fire places 33, poop 245, movable TNT 292).
  Only real HP loss counts: per frame the fall of each monster's HitPoints (floored at 0), so a death without
  damage (an explosion of its own, a Portal that closes) adds nothing and regrown HP can be lost again. A
  monster that is gone from the room in the frame of its death callback is charged the damage
  MC_ENTITY_TAKE_DMG reported for it in that frame, up to the HP it had. The binary combat block grows to 18
  doubles (monster_damage last).
  abp-0.2.10 (2026-09-29): abp-0.2.9 (the duel) with abp-0.2.8-hp's combat.monster_damage; the binary
  combat block has 18 doubles and the duel block stays last.
  move: 0 stop 1 up 2 up-right 3 right 4 down-right 5 down 6 down-left 7 left 8 up-left;
  shoot: 0 none 1 up 2 right 3 down 4 left.
]]

-- abp-0.2.11 (C41, user decisions 2026-09-29): AbpSetStats(speed, damage, shot_speed, tears, range), called by the room
-- setup chunk before the episode reseed, offsets the player's stats for the episode in MC_EVALUATE_CACHE on top of what
-- the game computes. MoveSpeed, Damage and ShotSpeed are added to; tears is the change in shots per second
-- (30 / (MaxFireDelay + 1)), rounded to a whole MaxFireDelay as AB+ keeps it an integer; range is in Repentance units of
-- 40 px: TearHeight is scaled by (6.5 + range) / 6.5, so the reported range 260 * TearHeight / -23.75 moves by 40 px per
-- unit. ShotSpeed stops at 0.6, as the engine does (measured: a lower value reads back as 0.6). Every reset clears the offsets; while they are all zero nothing is re-evaluated, so older runs and the
-- evaluations play exactly as before.
-- abp-0.2.12 (Go-Explore, user request 2026-09-30): batched replay. {"cmd":"play","actions":[code, ...],"repeat":k,
-- "repeats":[k, ...] (optional, per action),"stop_clear":bool} plays the actions back to back, each held for its repeat
-- logic frames exactly as the same step commands would (the next action is applied where a step observation would
-- have been sent), with a single observation at the end: first {"type":"ok","cmd":"play","played":n,"stop":"done"|
-- "dead"|"clear"}, then the step observation (its credits cover the whole batch). The batch stops early once the player
-- is dead (a cancelled lethal hit included) or, with stop_clear, the room is clear. code = move + 9 shoot + 45 bomb +
-- 90 item. Not with the duel NPC (its action comes with each step).
local VERSION = "abp-0.2.12"
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
		enemy_damage = 0, enemy_damage_fraction = 0, blocking_hp = 0, blocking_points = 0, blocking_count = 0,
		lineage_damage = 0, lineage_kills = 0, lineage_count = 0, lineage_hp = 0, tear_hits = 0, blocked_hits = 0,
		tear_misses = 0, miss_units = 0, miss_streak = 0, monster_damage = 0 },
	-- abp-0.2.8-hp: monsters = HP last seen per monster key; monster_hits = damage reported this frame per key.
	health = { players = {}, npcs = {}, deaths = {}, monsters = {}, monster_hits = {} },
	lethal = {},             -- player index -> true once a lethal hit was cancelled (virtual death)
	-- abp-0.2.3 roster lineage, keyed by entity_key: HP budget still countable, HP last seen,
	-- members that died or left, and this frame's lineage deaths and NPC inits (resolved per frame).
	lineage = {}, lineage_hp = {}, lineage_gone = {}, lineage_dying = {}, lineage_born = {},
	-- abp-0.2.4: tears already counted as a hit (entity_key), and the invincible player.
	tear_hit = {}, invincible = false,
	-- abp-0.2.5: the player's tears still in flight (entity_key -> EntityTear), resolved as hit or miss.
	tears_live = {},
	-- abp-0.2.6: the streak at which a miss's share of miss_units stops growing (0 = no cap).
	miss_cap = 0,
	-- abp-0.2.7: fire frame per tear key, hits waiting for their HP loss per NPC key ({fire, left,
	-- frame}), the last hit per NPC key ({fire, frame}), and the credits per fire frame ({d, k, m}).
	tear_fire = {}, pending_hits = {}, last_hit = {}, credits = {},
	-- abp-0.2.12: the play batch being run ({codes, repeats, rep, i, stop_clear}), nil outside one.
	play = nil,
}

-- abp-0.2.9 duel: the NPC's constants, measured on the player (EXPERIMENTS.md A6).
local DUEL_TYPE = 11   -- Pacer (11.1); MC_NPC_UPDATE is registered for this type
local DUEL = {
	-- movement per physics sub-step, two per logic frame: pos += v, then v = (the part along the input direction
	-- x friction, or x friction_against when it points against it; the part across x friction_side) + accel x input;
	-- no input: v x friction. Fitted on the player (MoveSpeed 1): largest error 0.003 px/frame.
	friction = 0.8803, friction_against = 0.845, friction_side = 0.78, accel = 0.528, substeps = 2,
	engine_factor = 0.75,   -- the NPC moves 0.75 x the velocity set in MC_NPC_UPDATE in the same frame
	fire_delay = 10,        -- the player's MaxFireDelay: a shot every 11 frames while shooting
	shot_speed = 10.0, inherit = 1.2, shot_spawn = 10.0, shot_side_min = 3.0, shot_side_max = 5.0,
	shot_height = -23.75, shot_falling = 0.13, shot_size = 8.165,
	iframes = 30,           -- damage again 30 frames after a hit (the player: cooldown 60, -2 per frame)
	hp = 21.0, size = 10.0, knockback = 0.3,
}
local DUEL_MOVES = { [0] = { 0, 0 }, { 0, -1 }, { 0.70710678, -0.70710678 }, { 1, 0 }, { 0.70710678, 0.70710678 },
	{ 0, 1 }, { -0.70710678, 0.70710678 }, { -1, 0 }, { -0.70710678, -0.70710678 } }
local DUEL_SHOTS = { [1] = { 0, -1 }, [2] = { 1, 0 }, [3] = { 0, 1 }, [4] = { -1, 0 } }

local function duel_side()
	return { shots = 0, hits = 0, misses = 0, miss_units = 0, miss_streak = 0, damage = 0, hurt = 0, hurt_amount = 0 }
end

-- key: the NPC's entity_key (nil: no duel); vx, vy: its velocity in the player's engine units; intended: the
-- displacement set last frame; hit_frame: logic_frames at its last damage; eye: side of the next shot; kick: tear
-- pushes waiting for the next frame (pushed: the tears that pushed); side[1] the player, side[2] the NPC; live[s]:
-- shots in flight (key -> entity); hit[s]: shots that damaged the other side.
local duel = {}
local function duel_clear()
	duel = { key = nil, active = false, dead = false, move = 0, shoot = 0, cooldown = 0, vx = 0, vy = 0,
		last = nil, intended = nil, hit_frame = nil, updated = nil, eye = 1, rng = nil, kick_x = 0, kick_y = 0, pushed = {},
		side = { duel_side(), duel_side() }, live = { {}, {} }, hit = { {}, {} } }
end
duel_clear()

local function zero_events()
	duel_clear()   -- abp-0.2.9: a reset drops the duel (a new NPC can reuse the old one's Index and InitSeed)
	state.events = { damage = 0, tears = 0, npc_deaths = 0, clears = 0 }
	state.combat = { player_damage_events = 0, player_damage = 0, enemy_damage_events = 0,
		enemy_damage = 0, enemy_damage_fraction = 0, blocking_hp = 0, blocking_points = 0, blocking_count = 0,
		lineage_damage = 0, lineage_kills = 0, lineage_count = 0, lineage_hp = 0, tear_hits = 0, blocked_hits = 0,
		tear_misses = 0, miss_units = 0, miss_streak = 0, monster_damage = 0 }
	state.health = { players = {}, npcs = {}, deaths = {}, monsters = {}, monster_hits = {} }
	state.lethal = {}
	state.lineage, state.lineage_hp, state.lineage_gone = {}, {}, {}
	state.lineage_dying, state.lineage_born = {}, {}
	state.tear_hit, state.invincible, state.tears_live = {}, false, {}
	state.miss_cap = 0
	state.stat_mods = nil   -- abp-0.2.11: the player's stat offsets (AbpSetStats)
	state.tear_fire, state.pending_hits, state.last_hit, state.credits = {}, {}, {}, {}
end

local function entity_key(e) return tostring(e.Index) .. ':' .. tostring(e.InitSeed) end

-- abp-0.2.8-hp: the engine's active-enemy types (Entity::IsActiveEnemy, RVA 0x13E0E0: 10 <= type < 1000, not a
-- shopkeeper 17, fire place 33, poop 245 or movable TNT 292); e must be an NPC.
local function is_monster(e)
	local t = e.Type
	return t >= 10 and t < 1000 and t ~= 17 and t ~= 33 and t ~= 245 and t ~= 292
end

local function record_enemy_loss(previous, hp)
	local loss = math.max(0, previous.hp - math.max(0, hp))
	if loss > 0 then
		state.combat.enemy_damage_events = state.combat.enemy_damage_events + 1
		state.combat.enemy_damage = state.combat.enemy_damage + loss
		state.combat.enemy_damage_fraction = state.combat.enemy_damage_fraction + loss / previous.max_hp
	end
end

-- abp-0.2.3 roster lineage ------------------------------------------------------------------
-- Entity:CanShutDoors() is a method; the EntityNPC a callback receives has a CanShutDoors field.
local function blocks_doors(e)
	local v = e.CanShutDoors
	if type(v) == "function" then return v(e) == true end
	return v == true
end

local function join_lineage(e)
	local key = entity_key(e)
	if state.lineage[key] == nil then
		local hp = math.max(0, e.HitPoints)
		state.lineage[key] = hp
		state.lineage_hp[key] = hp
	end
end

-- Death successors. In AB+ the NPCs a death leaves (a Nest's Big Spider or Trite, a Mulligan's
-- flies) are initialised in the frame of the death callback with SpawnerEntity nil, so they are
-- found by frame and place: doors-blocking NPCs initialised in that frame within LINEAGE_RADIUS
-- of where the lineage NPC died. mode 0: none join; 1: all of them join; 2: they join only when
-- the death left exactly one (a transformation), not several (flies released at death).
local LINEAGE_RADIUS = 60

-- The setup chunk (abplus.py ROOM_LUA / ARENA_LUA) calls this once the room is ready: the
-- doors-blocking NPCs alive now are the roster. Returns their number.
function AbpRosterMark(mode)
	state.lineage_mode = tonumber(mode) or 0
	state.lineage, state.lineage_hp, state.lineage_gone = {}, {}, {}
	state.lineage_dying, state.lineage_born = {}, {}
	state.combat.lineage_damage, state.combat.lineage_kills = 0, 0
	state.combat.tear_hits, state.tear_hit = 0, {}
	state.combat.tear_misses, state.combat.miss_units, state.combat.miss_streak = 0, 0, 0
	state.tears_live = {}
	state.tear_fire, state.pending_hits, state.last_hit, state.credits = {}, {}, {}, {}
	local n = 0
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		-- Exists(): the arena setup removes the room's own NPCs in the same frame.
		if e:ToNPC() and e:Exists() and e:CanShutDoors() and not e:IsDead() then
			join_lineage(e)
			n = n + 1
		end
	end
	return n
end

local function in_lineage(e)
	local key = entity_key(e)
	return state.lineage[key] ~= nil and not state.lineage_gone[key]
end

-- abp-0.2.4: the setup chunk switches the player's invincibility per episode (reset turns it off).
function AbpSetInvincible(on)
	state.invincible = (on == true)
	return state.invincible
end

-- abp-0.2.6: the setup chunk sets the miss cap per episode (reset turns it off).
function AbpSetMissCap(k)
	state.miss_cap = math.max(0, math.floor(tonumber(k) or 0))
	return state.miss_cap
end

-- abp-0.2.11 (C41): the player's stat offsets for the episode (header). stats_modified: the cached stats carry offsets,
-- so a call with all zero re-evaluates once to bring the base stats back.
local stats_modified = false
function AbpSetStats(speed, damage, shot_speed, tears, range)
	local m = { speed = tonumber(speed) or 0, damage = tonumber(damage) or 0, shot_speed = tonumber(shot_speed) or 0,
		tears = tonumber(tears) or 0, range = tonumber(range) or 0 }
	local any = m.speed ~= 0 or m.damage ~= 0 or m.shot_speed ~= 0 or m.tears ~= 0 or m.range ~= 0
	state.stat_mods = any and m or nil
	local player = Isaac.GetPlayer(0)
	if any or stats_modified then
		player:AddCacheFlags(CacheFlag.CACHE_ALL)
		player:EvaluateItems()
	end
	stats_modified = any
	return string.format("%.4f,%.4f,%.4f,%.4f,%.4f", player.MoveSpeed, player.Damage, player.ShotSpeed,
		player.MaxFireDelay, player.TearHeight)
end

mod:AddCallback(ModCallbacks.MC_EVALUATE_CACHE, function(_, player, flag)
	local m = state.stat_mods
	if not m then return end
	if flag == CacheFlag.CACHE_SPEED then
		player.MoveSpeed = math.max(0.1, player.MoveSpeed + m.speed)
	elseif flag == CacheFlag.CACHE_DAMAGE then
		player.Damage = math.max(0.5, player.Damage + m.damage)
	elseif flag == CacheFlag.CACHE_SHOTSPEED then
		player.ShotSpeed = math.max(0.6, player.ShotSpeed + m.shot_speed)   -- the engine's own floor is 0.6
	elseif flag == CacheFlag.CACHE_FIREDELAY then
		local tears = math.max(0.5, 30 / (player.MaxFireDelay + 1) + m.tears)
		player.MaxFireDelay = math.max(1, math.floor(30 / tears - 1 + 0.5))
	elseif flag == CacheFlag.CACHE_RANGE then
		player.TearHeight = player.TearHeight * (6.5 + m.range) / 6.5
	end
end)

-- abp-0.2.9 duel ------------------------------------------------------------------------------
-- The setup chunk calls this with the NPC it spawned (DUEL_TYPE): hp (default 21), no contact damage, the player's
-- Size and Mass, no engine knockback or status effects (tear pushes go through the duel's own movement law). The NPC starts
-- playing when its spawn phase ends (the first MC_NPC_UPDATE: duel.active).
function AbpDuelAttach(e, hp)
	local npc = e and e:ToNPC()
	if npc == nil or e.Type ~= DUEL_TYPE then return "not a duel NPC" end
	duel_clear()
	duel.key = entity_key(e)
	hp = tonumber(hp) or DUEL.hp
	npc.MaxHitPoints, npc.HitPoints = hp, hp
	npc.CollisionDamage = 0
	e.Size = DUEL.size
	e.Mass = Isaac.GetPlayer(0).Mass   -- bodies push each other alike (Pacer 3, player 5)
	npc:AddEntityFlags(EntityFlag.FLAG_NO_STATUS_EFFECTS | EntityFlag.FLAG_NO_KNOCKBACK)
	local rng = RNG()
	rng:SetSeed(e.InitSeed, 35)   -- the shots' side offsets: a function of the (reseeded) spawn
	duel.rng = rng
	return "attached " .. duel.key
end

-- Damage to the duel NPC (MC_ENTITY_TAKE_DMG): the player's damage cooldown and lethal rule; the player's tear that
-- does the damage is a hit. Returns false to cancel the damage, nil to let it through.
local function duel_take_damage(e, amount, source)
	if duel.dead or not duel.active then return false end
	if duel.hit_frame ~= nil and state.logic_frames - duel.hit_frame < DUEL.iframes then return false end
	duel.hit_frame = state.logic_frames
	local hp = math.max(0, e.HitPoints)
	local dealt = math.min(amount, hp)
	local me = duel.side[2]
	me.hurt, me.hurt_amount = me.hurt + 1, me.hurt_amount + dealt
	local shot = source and source.Entity
	if shot ~= nil and shot.Type == EntityType.ENTITY_TEAR then
		local key = entity_key(shot)
		if duel.live[1][key] ~= nil then
			duel.hit[1][key] = true
			local p = duel.side[1]
			p.hits, p.damage, p.miss_streak = p.hits + 1, p.damage + dealt, 0
		end
	end
	if amount >= hp then
		duel.dead = true   -- as the player's lethal hit: cancelled and counted; the NPC stays where it is
		return false
	end
	return nil
end

-- The engine's grid collision lets the NPC, moving diagonally, into a corner up to the corner point (2.9 px deeper
-- than the player into each wall, A6): at the end of every frame its circle (the player's radius) is pushed out of
-- every blocking grid cell around it, as the player is.
local function duel_unstick()
	if duel.key == nil or not duel.active then return end
	local e = nil
	for _, x in ipairs(Isaac.GetRoomEntities()) do
		if entity_key(x) == duel.key then e = x; break end
	end
	if e == nil then return end
	local room = Game():GetRoom()
	local width, size = room:GetGridWidth(), room:GetGridSize()
	local r = DUEL.size
	local x, y = e.Position.X, e.Position.Y
	local changed = false
	for _ = 1, 4 do
		local moved = false
		local centre = room:GetGridIndex(Vector(x, y))
		local col, row = centre % width, centre // width
		for dr = -1, 1 do
			for dc = -1, 1 do
				local c, rr = col + dc, row + dr
				local index = rr * width + c
				if c >= 0 and c < width and rr >= 0 and index < size
					and room:GetGridCollision(index) ~= GridCollisionClass.COLLISION_NONE then
					local cell = room:GetGridPosition(index)
					local qx = math.max(cell.X - 20, math.min(x, cell.X + 20))
					local qy = math.max(cell.Y - 20, math.min(y, cell.Y + 20))
					local dx, dy = x - qx, y - qy
					local d = math.sqrt(dx * dx + dy * dy)
					if d < r - 1e-6 then
						if d > 1e-6 then
							x, y = qx + dx / d * r, qy + dy / d * r
						else   -- the centre inside the cell: out along the nearer side
							local ox, oy = 20 + r - math.abs(x - cell.X), 20 + r - math.abs(y - cell.Y)
							if ox < oy then x = x + (x >= cell.X and ox or -ox) else y = y + (y >= cell.Y and oy or -oy) end
						end
						moved, changed = true, true
					end
				end
			end
		end
		if not moved then break end
	end
	if changed then e.Position = Vector(x, y) end
end

-- Duel shots that are gone: one that never damaged the other side is a miss (abp-0.2.5's miss rules and abp-0.2.6's
-- cap, for both sides).
local function duel_resolve()
	if duel.key == nil then return end
	for s = 1, 2 do
		local side, live, hit = duel.side[s], duel.live[s], duel.hit[s]
		for key, shot in pairs(live) do
			local ok, alive = pcall(function() return shot:Exists() and not shot:IsDead() end)
			if not ok or not alive then
				live[key] = nil
				if not hit[key] then
					side.misses = side.misses + 1
					side.miss_streak = side.miss_streak + 1
					side.miss_units = side.miss_units + (state.miss_cap > 0 and math.min(side.miss_streak, state.miss_cap)
						or side.miss_streak)
				end
				hit[key] = nil
			end
		end
	end
end

-- abp-0.2.4: a player tear that damages a (vulnerable, living) lineage NPC is one hit; a tear counts
-- once. Called from MC_ENTITY_TAKE_DMG for non-player entities; only source.Entity is read (see
-- the header: EntityRef.Type aborts the game).
local function count_tear_hit(entity, amount, source)
	if amount <= 0 or source == nil then return end
	local tear = source.Entity
	if tear == nil or tear.Type ~= EntityType.ENTITY_TEAR then return end
	if not entity:ToNPC() or entity:IsDead() or not entity:IsVulnerableEnemy() or not in_lineage(entity) then return end
	local key = entity_key(tear)
	if state.tear_hit[key] then return end
	state.tear_hit[key] = true
	state.combat.tear_hits = state.combat.tear_hits + 1
	state.combat.miss_streak = 0      -- abp-0.2.5: a hit ends the run of misses
	-- abp-0.2.7: the HP loss this hit causes (charged a frame later at most) goes to the tear's fire frame.
	local fire, npc = state.tear_fire[key], entity_key(entity)
	local list = state.pending_hits[npc]
	if list == nil then list = {}; state.pending_hits[npc] = list end
	list[#list + 1] = { fire = fire, left = amount, frame = state.logic_frames }
	state.last_hit[npc] = { fire = fire, frame = state.logic_frames }
end

-- abp-0.2.7: charge an amount of damage (d), kills (k) or miss units (m) to the frame a tear was
-- fired in; nil (no tear) charges the current frame.
local function credit(fire, field, amount)
	if amount == 0 then return end
	local f = fire or state.logic_frames
	local c = state.credits[f]
	if c == nil then c = { d = 0, k = 0, m = 0 }; state.credits[f] = c end
	c[field] = c[field] + amount
end

local PENDING_FRAMES = 4   -- a hit whose HP loss has not shown up after this many frames is dropped
local KILL_FRAMES = 6      -- a lineage death this soon after a tear hit it is that tear's kill

local function kill_credit(key)
	local h = state.last_hit[key]
	credit(h and state.logic_frames - h.frame <= KILL_FRAMES and h.fire or nil, "k", 1)
end

local function credit_list()
	local frames = {}
	for f in pairs(state.credits) do frames[#frames + 1] = f end
	table.sort(frames)
	local out = {}
	for _, f in ipairs(frames) do
		local c = state.credits[f]
		out[#out + 1] = { f, c.d, c.k, c.m }
	end
	return out
end

-- abp-0.2.5: tears that are gone. One that never damaged a lineage NPC is a miss; each miss adds the
-- length of the current run of misses to miss_units (so the k-th miss in a row adds k), up to the
-- miss cap (abp-0.2.6).
local function resolve_tears()
	for key, tear in pairs(state.tears_live) do
		local ok, alive = pcall(function() return tear:Exists() and not tear:IsDead() end)
		if not ok or not alive then
			state.tears_live[key] = nil
			if not state.tear_hit[key] then
				local c = state.combat
				c.tear_misses = c.tear_misses + 1
				c.miss_streak = c.miss_streak + 1
				local units = state.miss_cap > 0 and math.min(c.miss_streak, state.miss_cap) or c.miss_streak
				c.miss_units = c.miss_units + units
				credit(state.tear_fire[key], "m", units)   -- abp-0.2.7
			end
			state.tear_fire[key] = nil
		end
	end
end

-- HP a lineage NPC lost since last seen, up to its remaining budget.
local function charge_lineage(key, hp)
	local previous = state.lineage_hp[key]
	if previous and hp < previous then
		local loss = math.min(previous - hp, state.lineage[key])
		state.lineage[key] = state.lineage[key] - loss
		state.combat.lineage_damage = state.combat.lineage_damage + loss
		-- abp-0.2.7: the loss goes to the tears that hit this NPC, oldest first, up to each one's damage.
		local list, left = state.pending_hits[key], loss
		while list and #list > 0 and left > 0 do
			local p = list[1]
			local take = math.min(p.left, left)
			credit(p.fire, "d", take)
			p.left, left = p.left - take, left - take
			if p.left <= 1e-9 then table.remove(list, 1) end
		end
		if left > 0 then credit(nil, "d", left) end
	end
	state.lineage_hp[key] = hp
end

-- Diagnostics of the successor rule (read through the lua command).
AbpLineageStats = { deaths = 0, born = 0, near = 0, joined = 0, errors = 0, last_error = "" }

-- Successors of this frame's lineage deaths join the lineage (see LINEAGE_RADIUS).
local function resolve_lineage()
	local mode = state.lineage_mode or 0
	if mode > 0 and next(state.lineage_dying) ~= nil then
		local stats = AbpLineageStats
		for _ in pairs(state.lineage_dying) do stats.deaths = stats.deaths + 1 end
		stats.born = stats.born + #state.lineage_born
		local left = {}   -- dying key -> NPCs initialised this frame near where it died
		for _, npc in ipairs(state.lineage_born) do
			local ok, err = pcall(function()
				if npc:Exists() and not npc:IsDead() and blocks_doors(npc) then
					for key, pos in pairs(state.lineage_dying) do
						if npc.Position:Distance(pos) <= LINEAGE_RADIUS then
							left[key] = left[key] or {}
							left[key][#left[key] + 1] = npc
							stats.near = stats.near + 1
							break
						end
					end
				end
			end)
			if not ok then stats.errors = stats.errors + 1; stats.last_error = tostring(err) end
		end
		for _, npcs in pairs(left) do
			if mode == 1 or #npcs == 1 then
				for _, npc in ipairs(npcs) do join_lineage(npc); stats.joined = stats.joined + 1 end
			end
		end
	end
	state.lineage_dying, state.lineage_born = {}, {}
end

-- abp-0.2.8: a member that died and lives on as the same entity (a Gaper that loses its head becomes a
-- Gusher or Pacer) joins again with its HP then as the budget. Not in the frame of its death callback:
-- update_lineage counts that death as the kill first (IsDead() can turn true a frame before it).
local function rejoin_lineage(e)
	local key = entity_key(e)
	if state.lineage_gone[key] and not state.health.deaths[key] then
		local hp = math.max(0, e.HitPoints)
		state.lineage[key], state.lineage_hp[key], state.lineage_gone[key] = hp, hp, nil
		AbpLineageStats.rejoined = (AbpLineageStats.rejoined or 0) + 1
	end
end

local function update_lineage()
	local mode = state.lineage_mode or 0
	if mode > 0 then
		for _, e in ipairs(Isaac.GetRoomEntities()) do
			if e:ToNPC() and e:Exists() and not e:IsDead() and blocks_doors(e) then
				-- abp-0.2.5: in mode 3 every doors-blocking NPC joins from the first frame it is seen
				-- (spawns included).
				if mode == 3 then join_lineage(e) end
				rejoin_lineage(e)
			end
		end
	end
	local seen, count, hp_total = {}, 0, 0
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if e:ToNPC() and in_lineage(e) then
			local key = entity_key(e)
			seen[key] = true
			local dead = e:IsDead() or state.health.deaths[key]
			local hp = dead and 0 or math.max(0, e.HitPoints)
			charge_lineage(key, hp)
			if dead then
				state.lineage_gone[key] = true
				state.combat.lineage_kills = state.combat.lineage_kills + 1
				kill_credit(key)   -- abp-0.2.7
			else
				count = count + 1
				hp_total = hp_total + hp
			end
		end
	end
	for key in pairs(state.lineage) do
		if not seen[key] and not state.lineage_gone[key] then
			-- Left the room: a death this frame is a kill, anything else only ends the tracking.
			if state.health.deaths[key] then
				charge_lineage(key, 0)
				state.combat.lineage_kills = state.combat.lineage_kills + 1
				kill_credit(key)   -- abp-0.2.7
			end
			state.lineage_gone[key] = true
		end
	end
	state.combat.lineage_count = count
	state.combat.lineage_hp = hp_total
	-- abp-0.2.7: hits whose HP loss never showed up (a spent budget, a cancelled hit) are dropped.
	for key, list in pairs(state.pending_hits) do
		while #list > 0 and state.logic_frames - list[1].frame > PENDING_FRAMES do table.remove(list, 1) end
		if #list == 0 then state.pending_hits[key] = nil end
	end
end

-- Settled health every logic frame (same definition as 0.2.0).
local function update_combat()
	local game = Game()
	resolve_lineage()
	resolve_tears()
	duel_unstick()   -- abp-0.2.9
	duel_resolve()
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
	local current, monsters = {}, {}
	local blocking_hp, blocking_points, blocking_count = 0, 0, 0
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if e:ToNPC() and e:CanShutDoors() and not e:IsDead() then
			blocking_hp = blocking_hp + math.max(0, e.HitPoints)
			blocking_points = blocking_points + math.ceil(5 * math.max(0, e.MaxHitPoints) ^ 0.2)
			blocking_count = blocking_count + 1
		end
		if e:ToNPC() and e:IsEnemy() and e.MaxHitPoints > 0 then
			local key = entity_key(e)
			local previous = state.health.npcs[key]
			local hp = e:IsDead() and 0 or math.max(0, e.HitPoints)
			if previous then record_enemy_loss(previous, hp) end
			current[key] = { hp = hp, max_hp = previous and previous.max_hp or e.MaxHitPoints }
		end
		-- abp-0.2.8-hp: a monster's real HP loss since the last frame (a dead one keeps its HitPoints).
		if e:ToNPC() and is_monster(e) then
			local key = entity_key(e)
			local hp = math.max(0, e.HitPoints)
			local previous = state.health.monsters[key]
			if previous and hp < previous then state.combat.monster_damage = state.combat.monster_damage + (previous - hp) end
			monsters[key] = hp
		end
	end
	for key, previous in pairs(state.health.npcs) do
		if not current[key] and state.health.deaths[key] then record_enemy_loss(previous, 0) end
	end
	-- abp-0.2.8-hp: a monster gone in the frame of its death: the damage reported for it this frame, up to its HP.
	for key, previous in pairs(state.health.monsters) do
		if not monsters[key] and state.health.deaths[key] then
			state.combat.monster_damage = state.combat.monster_damage + math.min(previous, state.health.monster_hits[key] or 0)
		end
	end
	state.health.monsters, state.health.monster_hits = monsters, {}
	update_lineage()
	state.health.npcs = current
	state.health.deaths = {}
	state.combat.blocking_hp = blocking_hp
	state.combat.blocking_points = blocking_points
	state.combat.blocking_count = blocking_count
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
	state.play = nil
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
			rec.lineage = in_lineage(e); rec.blocking = e:CanShutDoors() and not e:IsDead()
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

-- abp-0.2.9: the duel NPC entity (nil when gone) and the observation's duel record (nil without a duel).
local function duel_npc()
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if entity_key(e) == duel.key then return e end
	end
	return nil
end

local function duel_record()
	if duel.key == nil then return nil end
	local e = duel_npc()
	local p = Isaac.GetPlayer(0)
	return {
		active = duel.active, dead = duel.dead, npc = e and e.Index or -1, pos = e and vec(e.Position) or { 0, 0 },
		size = e and e.Size or 0, hp = (e ~= nil and not duel.dead) and math.max(0, e.HitPoints) or 0,
		max_hp = e and e.MaxHitPoints or 0,
		-- frames of the coming ones in which damage is still blocked (both sides: 29 right after a hit)
		iframes = duel.hit_frame and math.max(0, DUEL.iframes - (state.logic_frames - duel.hit_frame)) or 0,
		player_iframes = math.max(0, p:GetDamageCooldown() // 2 - 1),
		cooldown = duel.cooldown, vel = { duel.vx, duel.vy }, move = duel.move, shoot = duel.shoot,
		player = duel.side[1], npc_side = duel.side[2],
	}
end

-- abp-0.2.9: moves the duel NPC to (x, y) at rest (checks and setups; lua command).
function AbpDuelPlace(x, y)
	local e = duel_npc()
	if e == nil then return "no duel NPC" end
	e.Position, e.Velocity = Vector(x, y), Vector(0, 0)
	duel.vx, duel.vy, duel.kick_x, duel.kick_y, duel.intended = 0, 0, 0, 0, nil
	duel.last = Vector(x, y)
	return "ok"
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
		blocking_hp = state.combat.blocking_hp, blocking_points = state.combat.blocking_points,
		blocking_count = state.combat.blocking_count, lineage_damage = state.combat.lineage_damage,
		lineage_kills = state.combat.lineage_kills, lineage_count = state.combat.lineage_count,
		lineage_hp = state.combat.lineage_hp, tear_hits = state.combat.tear_hits,
		blocked_hits = state.combat.blocked_hits, tear_misses = state.combat.tear_misses,
		miss_units = state.combat.miss_units, miss_streak = state.combat.miss_streak,
		credits = credit_list(),   -- abp-0.2.7
		monster_damage = state.combat.monster_damage }   -- abp-0.2.8-hp
	obs.duel = duel_record()   -- abp-0.2.9
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
			kind, extra = 6, packf("<BBBi8BdBB", B(e:IsEnemy()), B(e:IsVulnerableEnemy()), B(boss),
				I(npc:GetChampionColorIdx()), B(has_hp), has_hp and e.HitPoints / e.MaxHitPoints or 0,
				B(in_lineage(e)), B(e:CanShutDoors() and not e:IsDead()))
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

-- abp-0.2.7: combat.credits as a count and {fire_frame:int64, damage, kills, miss_units:double} records.
local function pack_credits()
	local list = credit_list()
	local parts = { packf("<I2", #list) }
	for _, r in ipairs(list) do parts[#parts + 1] = packf("<i8ddd", I(r[1]), r[2], r[3], r[4]) end
	return table.concat(parts)
end

-- abp-0.2.9: the duel block, last in the payload: 0, or 1 and the NPC's state and both sides' counters
-- (python/isaac_bridge/abplus_obs.py _DUEL, _DUEL_SIDE).
local function pack_duel()
	local r = duel_record()
	if r == nil then return packf("<B", 0) end
	local parts = { packf("<BBBi8ddddd" .. "i8i8i8ddBB", 1, B(r.active), B(r.dead), I(r.npc), r.pos[1], r.pos[2], r.size, r.hp,
		r.max_hp, I(r.iframes), I(r.player_iframes), I(r.cooldown), r.vel[1], r.vel[2], I(r.move), I(r.shoot)) }
	for _, s in ipairs({ r.player, r.npc_side }) do
		parts[#parts + 1] = packf("<dddddddd", s.shots, s.hits, s.misses, s.miss_units, s.miss_streak, s.damage, s.hurt,
			s.hurt_amount)
	end
	return table.concat(parts)
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
		packf("<dddddddddddddddddd", c.player_damage_events, c.player_damage, c.enemy_damage_events, c.enemy_damage,
			c.enemy_damage_fraction, c.blocking_hp, c.blocking_points, c.blocking_count,
			c.lineage_damage, c.lineage_kills, c.lineage_count, c.lineage_hp, c.tear_hits, c.blocked_hits,
			c.tear_misses, c.miss_units, c.miss_streak, c.monster_damage),   -- abp-0.2.8-hp: 18 doubles
		pack_credits(),   -- abp-0.2.7
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
	parts[#parts + 1] = pack_duel()   -- abp-0.2.9
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

local function send_obs_now(event)
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

-- abp-0.2.7: a step observation hands over the credits collected since the previous one.
local function send_obs(event)
	local sent = send_obs_now(event)
	if event == "step" then state.credits = {} end
	return sent
end

------------------------------------------------------------------ command loop

local function do_reset(cmd)
	state.held = {}; state.triggered = {}; state.frames_left = 0; state.play = nil
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

-- abp-0.2.12 play: the batch's next action, applied as the step command applies its own.
local function play_next()
	local p = state.play
	p.i = p.i + 1
	local code = math.floor(tonumber(p.codes[p.i]) or 0)
	apply_action({ move = code % 9, shoot = (code // 9) % 5, bomb = (code // 45) % 2, item = (code // 90) % 2 })
	state.frames_left = math.max(1, math.floor(tonumber(p.repeats and p.repeats[p.i]) or p.rep))
	state.stats.steps = state.stats.steps + 1
end

-- Where a step observation would be sent: true when the batch goes on with its next action; false when it has ended
-- (the ok message is sent, the step observation follows).
local function play_continue()
	local p = state.play
	local player = Isaac.GetPlayer(0)
	local stop = nil
	if player ~= nil and (player:IsDead() or state.lethal[player.Index] == true) then stop = "dead"
	elseif p.stop_clear and Game():GetRoom():IsClear() then stop = "clear"
	elseif p.i >= #p.codes then stop = "done" end
	if stop == nil then
		play_next()
		return true
	end
	state.play = nil
	send({ type = "ok", cmd = "play", played = p.i, stop = stop })
	return false
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
			if duel.key ~= nil then   -- abp-0.2.9: the duel NPC's move and shoot for the frames of this step
				duel.move = math.max(0, math.min(8, math.floor(tonumber(cmd.duel_move) or 0)))
				duel.shoot = math.max(0, math.min(4, math.floor(tonumber(cmd.duel_shoot) or 0)))
			end
			state.frames_left = math.max(1, math.floor(tonumber(cmd["repeat"]) or cfg.default_repeat))
			state.stats.steps = state.stats.steps + 1
			return
		elseif cmd.cmd == "play" then   -- abp-0.2.12
			if type(cmd.actions) ~= "table" or #cmd.actions == 0 then
				send({ type = "error", msg = "play: no actions" })
			elseif cmd.repeats ~= nil and (type(cmd.repeats) ~= "table" or #cmd.repeats ~= #cmd.actions) then
				send({ type = "error", msg = "play: repeats must match actions" })
			elseif duel.key ~= nil then
				send({ type = "error", msg = "play: not with the duel NPC" })
			else
				state.play = { codes = cmd.actions, repeats = cmd.repeats, i = 0, stop_clear = cmd.stop_clear == true,
					rep = math.max(1, math.floor(tonumber(cmd["repeat"]) or cfg.default_repeat)) }
				play_next()
				return
			end
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

-- abp-0.2.9: damage the player takes in a duel (every source; the engine calls MC_ENTITY_TAKE_DMG for the player only
-- outside its damage cooldown); the duel NPC's shot that does it is the NPC's hit.
local function duel_player_damage(p, amount, source)
	local dealt = math.min(amount, health_units(p))
	local me = duel.side[1]
	me.hurt, me.hurt_amount = me.hurt + 1, me.hurt_amount + dealt
	local shot = source and source.Entity
	if shot ~= nil and shot.Type == EntityType.ENTITY_PROJECTILE then
		local key = entity_key(shot)
		if duel.live[2][key] ~= nil then
			duel.hit[2][key] = true
			local n = duel.side[2]
			n.hits, n.damage, n.miss_streak = n.hits + 1, n.damage + dealt, 0
		end
	end
end

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount, flags, source, countdown)
	local p = entity and entity:ToPlayer()
	if not p then
		if entity then
			if duel.key ~= nil and entity_key(entity) == duel.key then   -- abp-0.2.9
				local result = duel_take_damage(entity, amount, source)
				if result ~= nil then return result end
			end
			if entity:ToNPC() and is_monster(entity) then   -- abp-0.2.8-hp: for a monster gone this frame
				local key = entity_key(entity)
				state.health.monster_hits[key] = (state.health.monster_hits[key] or 0) + math.max(0, amount)
			end
			count_tear_hit(entity, amount, source)
		end
		return nil
	end
	-- abp-0.2.4: an invincible player takes no damage (returning false cancels it, and the later
	-- callbacks of the same event are not called).
	if state.invincible and state.client then
		state.combat.blocked_hits = state.combat.blocked_hits + 1
		return false
	end
	state.events.damage = state.events.damage + 1
	if duel.key ~= nil and not state.lethal[p.Index] then duel_player_damage(p, amount, source) end   -- abp-0.2.9
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
	local key = entity_key(tear)
	state.tears_live[key] = tear   -- abp-0.2.5: followed until resolved (resolve_tears)
	-- abp-0.2.7: MC_POST_UPDATE (which counts logic_frames) runs after the entities of the frame.
	state.tear_fire[key] = state.logic_frames + 1
	if duel.key ~= nil then   -- abp-0.2.9: the player's duel shots
		duel.live[1][key] = tear
		duel.side[1].shots = duel.side[1].shots + 1
	end
end)

-- abp-0.2.9: the duel NPC's frame, after the engine's AI for it (which is overridden). The engine then moves it by
-- 0.75 x the velocity set here, in this frame (A6: no lag).
mod:AddCallback(ModCallbacks.MC_NPC_UPDATE, function(_, npc)
	if duel.key == nil or entity_key(npc) ~= duel.key then return end
	local pos = npc.Position
	if not duel.active then
		-- The engine updates a new NPC once when it spawns and then not until its spawn phase is over (about 20
		-- frames): it plays from the second of two updates in consecutive frames.
		local frame = state.logic_frames
		local previous = duel.updated
		duel.updated = frame
		if previous == nil or frame - previous ~= 1 then
			npc.Velocity = Vector(0, 0)   -- not the engine's own walk
			return
		end
		duel.active = true
		duel.cooldown = -1      -- the player's fire delay is below 0 by now: both can shoot at once
		duel.last = Vector(pos.X, pos.Y)
	end
	if duel.dead then
		npc.Velocity = Vector(0, 0)
		return
	end
	-- Walls: the part of last frame's displacement that did not happen stops that part of the velocity, as the
	-- player's grid collision does.
	if duel.intended ~= nil then
		local dx, dy = pos.X - duel.last.X, pos.Y - duel.last.Y
		local ix, iy = duel.intended[1], duel.intended[2]
		if math.abs(ix) > 1e-6 then duel.vx = duel.vx * math.max(0, math.min(1, dx / ix)) end
		if math.abs(iy) > 1e-6 then duel.vy = duel.vy * math.max(0, math.min(1, dy / iy)) end
	end
	local vx, vy = duel.vx + duel.kick_x, duel.vy + duel.kick_y
	duel.kick_x, duel.kick_y = 0, 0
	local start_x, start_y = vx, vy   -- the velocity a shot of this frame inherits (the player's at the frame start)
	local u = DUEL_MOVES[duel.move] or DUEL_MOVES[0]
	local ux, uy = u[1], u[2]
	local sx, sy = 0, 0
	for _ = 1, DUEL.substeps do
		sx, sy = sx + vx, sy + vy
		if ux == 0 and uy == 0 then
			vx, vy = vx * DUEL.friction, vy * DUEL.friction
		else
			local along = vx * ux + vy * uy
			local qx, qy = vx - along * ux, vy - along * uy
			along = along * (along >= 0 and DUEL.friction or DUEL.friction_against)
			vx = along * ux + qx * DUEL.friction_side + DUEL.accel * ux
			vy = along * uy + qy * DUEL.friction_side + DUEL.accel * uy
		end
	end
	duel.vx, duel.vy = vx, vy
	duel.intended = { sx, sy }
	duel.last = Vector(pos.X, pos.Y)
	npc.Velocity = Vector(sx / DUEL.engine_factor, sy / DUEL.engine_factor)
	-- Shooting, as the player's fire delay: down by one every frame, a shot when it is below 0 while shooting.
	duel.cooldown = duel.cooldown - 1
	local d = DUEL_SHOTS[duel.shoot]
	if d ~= nil and duel.cooldown < 0 then
		local side = duel.eye * (DUEL.shot_side_min + (DUEL.shot_side_max - DUEL.shot_side_min) * duel.rng:RandomFloat())
		duel.eye = -duel.eye
		local wx, wy = d[1] * DUEL.shot_speed + DUEL.inherit * start_x, d[2] * DUEL.shot_speed + DUEL.inherit * start_y
		-- A tear starts from where the player ends the frame and moves once in the frame it is fired in; a projectile
		-- spawned here moves from the next frame on, so it starts one move and one fall step ahead.
		local x = pos.X + sx + d[1] * DUEL.shot_spawn - d[2] * side + wx
		local y = pos.Y + sy + d[2] * DUEL.shot_spawn + d[1] * side + wy
		local shot = Isaac.Spawn(EntityType.ENTITY_PROJECTILE, 0, 0, Vector(x, y), Vector(wx, wy), npc):ToProjectile()
		local fall = DUEL.shot_falling + 0.1 * (1 - DUEL.shot_falling)   -- a tear's falling speed after its first frame
		shot.Height, shot.FallingSpeed, shot.FallingAccel = DUEL.shot_height + fall, fall, 0
		shot.Size = DUEL.shot_size
		duel.live[2][entity_key(shot)] = shot
		duel.side[2].shots = duel.side[2].shots + 1
		duel.cooldown = DUEL.fire_delay
	end
end, DUEL_TYPE)

-- abp-0.2.9: a player tear that meets the duel NPC pushes it (the player's push from a projectile: 0.3 x its velocity),
-- also when it does no damage.
mod:AddCallback(ModCallbacks.MC_PRE_TEAR_COLLISION, function(_, tear, other, low)
	if duel.key ~= nil and other ~= nil and not duel.dead and entity_key(other) == duel.key then
		local key = entity_key(tear)
		if not duel.pushed[key] then   -- once per tear
			duel.pushed[key] = true
			duel.kick_x = duel.kick_x + DUEL.knockback * tear.Velocity.X
			duel.kick_y = duel.kick_y + DUEL.knockback * tear.Velocity.Y
		end
	end
	return nil
end)
mod:AddCallback(ModCallbacks.MC_POST_NPC_DEATH, function(_, npc)
	state.events.npc_deaths = state.events.npc_deaths + 1
	local key = entity_key(npc)
	state.health.deaths[key] = true
	-- IsDead() can turn true a frame before this callback, when update_lineage has already counted
	-- the kill: any member (alive or just gone) marks its death place for its successors.
	if state.lineage[key] ~= nil then
		state.lineage_dying[key] = Vector(npc.Position.X, npc.Position.Y)
		AbpLineageStats.marked = (AbpLineageStats.marked or 0) + 1
	end
end)
mod:AddCallback(ModCallbacks.MC_POST_NPC_INIT, function(_, npc)
	state.lineage_born[#state.lineage_born + 1] = npc
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
		if state.play ~= nil and play_continue() then return end   -- abp-0.2.12
		if send_obs("step") then wait_command() end
	end
end)

log("loaded " .. VERSION .. " port " .. cfg.port .. " (binds on the first logic frame)")
