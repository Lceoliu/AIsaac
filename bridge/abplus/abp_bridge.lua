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
-- abp-0.2.13 (goal-conditioned line, EXPERIMENTS.md A9/A10, rl/docs/GOAL_CONDITIONED_DESIGN.md):
--   * An undiscovered secret-room door (GridEntityDoor variant 7, DOOR_HIDDEN) is left out of the doors of every
--     observation and its grid cell is reported as a wall (type 15); a bomb turns it into variant 8 (open), from then on
--     it is listed as usual. The goto rooms of the earlier runs have only their barred entrance, so they are unchanged.
--   * A step ends early, at the first logic frame after the room changed (MC_POST_NEW_ROOM during the step): the held
--     action keeps acting on that frame, as the engine does, and the observation is the new room's first frame. A play
--     batch stopped there too (stop "room") until 2026-10-07; since then (same VERSION) the action under way ends there
--     as a step's does and the batch goes on with its next action, so a play is the same game as the same step lines
--     across room changes (a bomb opening a secret room, a door walked through); "stop_room":true in a play or
--     fork_many command keeps the old stop.
--   * "goal":[x, y, r] in a step command: every logic frame of the step the player's distance to (x, y) is checked; the
--     step ends early at the first frame it is <= r.
--   * Every observation carries nav = {goal_hit, room_changed, leave_door, enter_door, goal_min_dist} for its step
--     (binary v2: a trailing <BBqqd> section).
--   * "stop_clear":true in a step command (the goal line's COMBAT options): the step ends at the logic frame the room
--     becomes clear, so the player cannot walk out through a door that opens within the same step.
-- abp-0.2.14 (EXPERIMENTS.md A19, B9, B10): `fork` (a clone of the process through abp_turbo's ABP_FORK), lean mode and
-- native input; described at "lean mode" and "fork" below.
-- abp-0.2.15 (whole-floor episodes, EXPERIMENTS.md C57): the lean observation's room index is the room's SafeGridIndex,
-- a door's fourth byte says whether the room behind it was visited / is clear, a trapdoor or stairs appearing resends
-- the terrain, and a map block carries the rooms the minimap shows. Everything else as abp-0.2.14.
-- Native obs (2026-10-03, same VERSION: the bytes on the wire are unchanged): with a libabp_turbo.so that has
-- ABP_OBS_INIT, the lean observation's fixed part and its terrain check are built in C (see "native obs" at pack_lean);
-- ISAAC_RL_NATIVE_OBS=0 or the `native_obs` command (mode 0 / 1 / 2 = Lua / native / check) selects the path, and
-- `lean_profile` times the parts.
local VERSION = "abp-0.2.15"
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

local FM_AVAILABLE = getenv("ABP_FM_PROBE") == "1"   -- abp_turbo has fork_many's pipe (see fork_many)

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
	-- abp-0.2.13: the step's goal ({x, y, r}, nil without one) and what happened during the step.
	goal = nil, nav = { hit = false, changed = false, min = -1 },
	stop_clear = false, was_clear = false,
	lean = false,            -- abp-0.2.14: lean mode (pack_lean)
	native_input = false,    -- abp-0.2.14: abp_turbo answers the engine's input queries (lean mode)
	-- 2026-10-06 (items, Phase A; same VERSION, hello says items = true): the lean observation's inventory block (flag 16,
	-- see pack_lean) and the run-mode map rules. Off unless ISAAC_RL_LEAN_ITEMS=1 at launch or the `lean_items` command
	-- (clones inherit it); off, the bytes on the wire are those of abp-0.2.15.
	lean_items = getenv("ISAAC_RL_LEAN_ITEMS") == "1",
}

-- abp-0.2.13: an undiscovered secret-room door (DOOR_HIDDEN) is not part of what the player sees.
local DOOR_HIDDEN = 7
local function visible_door(d)
	return d:GetVariant() ~= DOOR_HIDDEN
end

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
	if state.clone then getenv("ABP_EXIT") end   -- abp-0.2.14: a clone lives as long as its connection
	log("client disconnected: " .. tostring(reason))
	if state.client then state.client:close() end
	state.client = nil
	state.held = {}; state.triggered = {}; state.frames_left = 0; state.pending_reset = false
	state.play = nil
	if state.native_input then   -- abp-0.2.14: without a client the engine reads the real devices again
		getenv("ABP_INPUT_OFF")
		state.native_input = false
	end
end

-- 2026-10-06 (the root's first-build hang, EXPERIMENTS.md): ISAAC_RL_PORT_FILE=<path> (set by the tok workers'
-- instances, abplus_goexplore.Instance) makes the bridge write the port it listens on to that file, and bind a free
-- port (port 0) when the configured one is taken. Before, a taken port (a parked clone's connection that happened to get
-- it as its ephemeral local port, another run's instance) disabled the bridge: the game ran on without a client at a
-- full core and the worker waited for it. Without the variable nothing changes. Networking only: no game state.
local function write_port_file(port)
	local path = getenv("ISAAC_RL_PORT_FILE")
	if not path or path == "" or not io or not io.open then return end
	local ok, f = pcall(io.open, path .. ".tmp", "w")
	if not ok or not f then log("port file " .. path .. " not written"); return end
	f:write(tostring(port) .. "\n")
	f:close()
	local renamed, res = false, nil
	if os and os.rename then renamed, res = pcall(os.rename, path .. ".tmp", path) end
	if not (renamed and res) then log("port file " .. path .. " not renamed") end
end

local function try_bind()
	state.bind_attempted = true
	if not socket_ok then
		log("luasocket unavailable (" .. tostring(socket) .. "); start the game with --luadebug. Bridge disabled.")
		return
	end
	local server, err = socket.bind(cfg.host, cfg.port, 1)
	if not server and (getenv("ISAAC_RL_PORT_FILE") or "") ~= "" then
		log("bind failed on " .. cfg.host .. ":" .. cfg.port .. " (" .. tostring(err) .. "); binding a free port")
		server, err = socket.bind(cfg.host, 0, 1)
		if server then
			local _, p = server:getsockname()
			cfg.port = tonumber(p) or cfg.port
		end
	end
	if not server then
		log("bind failed on " .. cfg.host .. ":" .. cfg.port .. " (" .. tostring(err) .. "). Bridge disabled.")
		return
	end
	server:settimeout(0)
	state.server = server
	write_port_file(cfg.port)
	log("listening on " .. cfg.host .. ":" .. cfg.port .. " (" .. VERSION .. ")")
end

local function try_accept()
	if not state.server or state.clone then return end
	local c = state.server:accept()
	if c then
		c:settimeout(nil)
		c:setoption("tcp-nodelay", true)
		state.client = c
		state.obs_format, state.obs_validate, state.terrain_dirty = 1, false, true
		log("client connected on port " .. cfg.port)
		send({ type = "hello", version = VERSION, engine = "abplus-1.06", port = cfg.port,
			privileged = cfg.privileged, game_frame = Game():GetFrameCount(), logic_frames = state.logic_frames,
			step_line = true, fork_many = FM_AVAILABLE, items = true, lean_items = state.lean_items })
	end
end

------------------------------------------------------------------ actions

-- abp-0.2.14 native input (lean mode; abp_turbo ABP_INPUT): the held and pressed-this-frame actions as bit masks,
-- answered by abp_turbo to the engine's input queries in place of the MC_INPUT_ACTION callback below, with the same
-- values. Every change of state.held / state.triggered is pushed.
local function push_input()
	if not state.native_input then return end
	local h, t = 0, 0
	for a in pairs(state.held) do h = h | (1 << a) end
	for a in pairs(state.triggered) do t = t | (1 << a) end
	getenv("ABP_INPUT:" .. h .. ":" .. t)
end

local function set_native_input(on)
	if on and state.client and state.control then
		local mask = 0
		for a in pairs(CONTROLLED) do mask = mask | (1 << a) end
		state.native_input = getenv("ABP_INPUT_ON:" .. mask) == "1"
		push_input()
	else
		if state.native_input then getenv("ABP_INPUT_OFF") end
		state.native_input = false
	end
end

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
	push_input()
end

mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, action)
	if not (state.client or state.headless) or not state.control then return nil end
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
			if gt == GridEntityType.GRID_DOOR and g:GetVariant() == DOOR_HIDDEN then   -- abp-0.2.13: seen as a wall
				cells[#cells + 1] = { i, GridEntityType.GRID_WALL, 0, 0, g.CollisionClass, g.Position.X, g.Position.Y }
			elseif gt ~= GridEntityType.GRID_NULL and gt ~= GridEntityType.GRID_DECORATION then
				cells[#cells + 1] = { i, gt, g:GetVariant(), g.State, g.CollisionClass, g.Position.X, g.Position.Y }
			end
		end
	end
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		if d and visible_door(d) then
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

-- abp-0.2.13: what happened during the step (python/isaac_bridge/abplus_obs.py _NAV).
local function nav_record()
	local level = Game():GetLevel()
	return { goal_hit = state.nav.hit, room_changed = state.nav.changed, leave_door = level.LeaveDoor,
		enter_door = level.EnterDoor, goal_min_dist = state.nav.min }
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
	obs.nav = nav_record()   -- abp-0.2.13
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
	-- abp-0.2.13: door variants too, so the grid is re-sent on the frame a bomb reveals a secret door (DOOR_HIDDEN ->
	-- open), not a frame later when its collision changes
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		parts[#parts + 1] = d and d:GetVariant() or -1
	end
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

local function pack_nav()
	local r = nav_record()
	return packf("<BBi8i8d", B(r.goal_hit), B(r.room_changed), I(r.leave_door), I(r.enter_door), r.goal_min_dist)
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
		if d and visible_door(d) then
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
	parts[#parts + 1] = pack_nav()    -- abp-0.2.13
	return table.concat(parts)
end

------------------------------------------------------------------ lean mode (abp-0.2.14)
-- For the fork sampler's episode clones (the `lean` command, or lean = true in `fork`): the per-frame bookkeeping of
-- the reward profiles (update_combat, the tear and lineage tables) is skipped, and a step answers with one
-- fixed-layout observation, "L <bytes> <event> <seq>\n" + payload, decoded by isaac_bridge/abplus_lean.py:
--   header   <I4I4I4BBBB  "ABP3", logic frames, game frame, flags (1 room clear, 2 terrain block present, 4 paused,
--            8 map block present), players, doors, lasers
--   room     <i4i4i4i4i4i4i4i4ffff  type, shape, grid width, grid height, room index (the room's SafeGridIndex: the
--            same through whichever door a big room was entered; negative off the grid), stage, alive enemies, room
--            frame, top left x y, bottom right x y
--   totals   <dddd  player 0's total damage taken since the run's start (half hearts), the summed HitPoints of the
--            room's monsters (is_monster), the summed HitPoints and the count of its door-blocking NPCs. For rewards
--            and the episode's end, not for the policy: a monster's HP is not something the player sees.
--   players  x 38 doubles (LEAN_PLAYER; abplus_lean.PLAYER_FIELDS). 2026-10-07 (charge, same VERSION): the last two
--            are the charge counter of a charged weapon (Entity_Player +0x2634, int: what the charge bar shows for
--            Brimstone, Monstro's Lung, Mom's Knife, Chocolate Milk, Cursed Eye, Tech X, the Forgotten's bone; 0 while
--            nothing charges; abp_turbo's abp_player_charge, no Lua API reads it) and the weapon types as a bit mask
--            (bit w set when HasWeaponType(w), w = 0 .. 10: 1 tears, 2 Brimstone, 3 Technology, 4 Mom's Knife, 5 Dr.
--            Fetus, 6 Epic Fetus, 7 Monstro's Lung, 8 Ludovico, 9 Tech X, 10 bone)
--   doors    x <BBBBffi4  slot, open, locked, flags (1 the room behind was visited, 2 it is clear: what the
--            minimap shows), x, y, target room type (visible doors only)
--   <I2 entity count, entities x LEAN_ENTITY (108 bytes; the rules of pack_entity: no player, only whitelisted
--            effects, only visible entities; hp is HitPoints / MaxHitPoints for bosses only, as in format 2)
--   lasers   x <i8 entity index + format 2's laser block
--   terrain  (flag 2) <I4 version, s4 terrain JSON, s4 grid JSON, as format 2
--   map      (flag 8) <B count, rooms x <BBBBB  GridIndex, shape, type (0 unless its icon is shown or it was visited),
--            DisplayFlags, flags (1 visited, 2 visited and clear, 4 the room the player is in). Only rooms the minimap
--            shows (DisplayFlags ~= 0): what the player sees of the floor. Sent with a connection's first observation
--            and when it changes (checked at a room change, a clear and every LEAN_FULL_EVERY logic frames).
-- The engine's velocity is always sent. The terrain is checked every step on the cells that held something at the
-- last full pass (a destroyed rock or poop shows at once) and in full every LEAN_FULL_EVERY logic frames; since
-- 2026-10-07 also every step on everything the block's grid records hold (terrain_content_sig: a grid entity's type,
-- variant, state, collision class), so a step's terrain block is never behind the game's.
local LEAN_MAGIC = 0x33504241
local LEAN_ENTITY = "<i8i4i4i4ddddfffi4i4fi4i4BBBBi4ffff"
local LEAN_PLAYER = "<" .. string.rep("d", 38)
local LEAN_FULL_EVERY = 30
local lean = { cells = nil, full_at = -100000, full_sig = nil, part_sig = nil, map_key = nil, map_at = -100000,
	map_sig = nil }

local function lean_room_index(level)
	local idx = level:GetCurrentRoomIndex()
	if idx >= 0 then
		local ok, safe = pcall(function() return level:GetCurrentRoomDesc().SafeGridIndex end)
		if ok and math.tointeger(safe) then idx = math.tointeger(safe) end
	end
	return idx
end

local function lean_map(level, here)
	-- items / run mode (state.lean_items): under Curse of the Lost the game shows no map
	if state.lean_items and (level:GetCurses() & 4) ~= 0 then return spack("<B", 0) end
	local rooms = level:GetRooms()
	local recs = {}
	for i = 0, rooms.Size - 1 do
		local r = rooms:Get(i)
		local display, gi = r.DisplayFlags, r.GridIndex
		if display ~= 0 and gi >= 0 and gi < 169 then
			local d = r.Data
			local visited = r.VisitedCount > 0
			local known = visited or (display & 4) ~= 0
			recs[#recs + 1] = spack("<BBBBB", gi, d.Shape & 255, known and (d.Type & 255) or 0, display & 255,
				(visited and 1 or 0) + ((visited and r.Clear) and 2 or 0) + (r.SafeGridIndex == here and 4 or 0))
		end
	end
	return spack("<B", #recs) .. table.concat(recs)
end

-- Inventory block (2026-10-06, items, Phase A; flag 16; only with state.lean_items, built here in Lua in every mode, so
-- the native and the Lua paths send the same bytes):
--   <BBHHHHHHI4  n (held collectible ids that follow, at most INV_CAP), layout version 1, the active item's MaxCharges
--                (0 without one), trinket slots 0 and 1, pill color in pocket slot 0, its PillEffect + 1 when the pill
--                is identified (what the game names) else 0, card / rune in slot 0, the level's curses
--   n x <HB      collectible id (1 .. 1023), how many are held (GetCollectibleNum, at most 255); newest first: an id
--                whose count went up moves to the front, ids gained together in id order, an id held at the first scan
--                of the run in id order
-- The held set comes from GetCollectibleNum over every collectible id of the item config (AB+: 1 .. 552). The scan runs
-- when a cheap key changes (GetCollectibleCount, the active item, pill, card, trinkets, curses, room, stage), at a
-- connection's first observation and every LEAN_FULL_EVERY logic frames; the block is sent with a connection's first
-- observation and when its bytes change. ISAAC_RL_INV_CHECK=1 (diagnostic): every decision in between also compares
-- the cached block with a full scan (ABP_INV.checks / misses) without changing what is sent.
local INV_CAP = 64
local INV = { n = nil, order = {}, counts = {}, key = nil, at = -100000, block = nil, sent = nil, scans = 0,
	checks = 0, misses = 0, first_miss = nil }
ABP_INV = INV
local INV_CHECK = getenv("ISAAC_RL_INV_CHECK") == "1"

local function inv_ids()
	if INV.n == nil then
		local ok, size = pcall(function() return Isaac.GetItemConfig():GetCollectibles().Size end)
		size = ok and math.tointeger(size) or 553
		INV.n = math.min(1023, size - 1)
	end
	return INV.n
end

local function inv_held(p)
	local held = {}
	for id = 1, inv_ids() do
		local c = p:GetCollectibleNum(id)
		if c > 0 then held[id] = c end
	end
	return held
end

local function inv_order(held)
	local new, isnew = {}, {}
	for id, c in pairs(held) do
		if (INV.counts[id] or 0) < c then new[#new + 1] = id; isnew[id] = true end
	end
	table.sort(new)
	local order = new
	for _, id in ipairs(INV.order) do
		if held[id] and not isnew[id] then order[#order + 1] = id end
	end
	INV.order, INV.counts = order, held
end

local function inv_header(p, level, n)
	local active = p:GetActiveItem()
	local maxc = 0
	if active ~= 0 then
		pcall(function() maxc = math.tointeger(Isaac.GetItemConfig():GetCollectible(active).MaxCharges) or 0 end)
	end
	local pill = p:GetPill(0)
	local effect = 0
	if pill ~= 0 then
		pcall(function()
			local pool = Game():GetItemPool()
			if pool:IsPillIdentified(pill) then effect = (math.tointeger(pool:GetPillEffect(pill)) or -1) + 1 end
		end)
	end
	return spack("<BBHHHHHHI4", n, 1, maxc, p:GetTrinket(0) & 0xFFFF, p:GetTrinket(1) & 0xFFFF, pill & 0xFFFF, effect,
		p:GetCard(0) & 0xFFFF, level:GetCurses())
end

local function inv_block(p, level)
	local n = math.min(#INV.order, INV_CAP)
	local parts = {}
	for k = 1, n do
		local id = INV.order[k]
		parts[k] = spack("<HB", id, math.min(INV.counts[id], 255))
	end
	return inv_header(p, level, n) .. table.concat(parts)
end

local function lean_inventory(event, here)
	local p = Isaac.GetPlayer(0)
	local level = Game():GetLevel()
	local key = table.concat({ p:GetCollectibleCount(), p:GetActiveItem(), p:GetPill(0), p:GetCard(0), p:GetTrinket(0),
		p:GetTrinket(1), level:GetCurses(), here, level:GetStage() }, ",")
	local due = event ~= "step" or INV.block == nil or key ~= INV.key or state.logic_frames - INV.at >= LEAN_FULL_EVERY
		or state.logic_frames < INV.at
	if INV_CHECK and not due then   -- diagnostic: the cached block against a full scan
		INV.checks = INV.checks + 1
		local held, same = inv_held(p), true
		for id, c in pairs(held) do if INV.counts[id] ~= c then same = false end end
		for id, c in pairs(INV.counts) do if held[id] ~= c then same = false end end
		if inv_header(p, level, math.min(#INV.order, INV_CAP)) ~= INV.block:sub(1, 18) then same = false end
		if not same then
			INV.misses = INV.misses + 1
			if INV.first_miss == nil then INV.first_miss = state.logic_frames end
		end
	end
	if due then
		INV.key, INV.at = key, state.logic_frames
		inv_order(inv_held(p))
		INV.scans = INV.scans + 1
		INV.block = inv_block(p, level)
	end
	if event ~= "step" or INV.block ~= INV.sent then
		INV.sent = INV.block
		return INV.block
	end
	return ""
end

-- Per cell the room's collision class and, with a grid entity, its type, variant, State and CollisionClass, else -1
-- (abp_turbo's obs_terrain_content reads the same values).
local function terrain_content_sig(room)
	local parts, k = {}, 0
	for i = 0, room:GetGridSize() - 1 do
		k = k + 1
		parts[k] = room:GetGridCollision(i)
		local g = room:GetGridEntity(i)
		if g then
			parts[k + 1], parts[k + 2], parts[k + 3], parts[k + 4] = g:GetType(), g:GetVariant(), g.State, g.CollisionClass
			k = k + 4
		else
			k = k + 1
			parts[k] = -1
		end
	end
	return table.concat(parts, ",")
end

local function lean_terrain_changed(room, player, force)
	local changed = false
	if force or lean.cells == nil or state.logic_frames - lean.full_at >= LEAN_FULL_EVERY
		or state.logic_frames < lean.full_at then
		local sig = terrain_signature(room, player)
		-- a trapdoor or stairs appearing (the floor's exit after the boss) changes no collision class
		for i = 0, room:GetGridSize() - 1 do
			local g = room:GetGridEntity(i)
			if g then
				local gt = g:GetType()
				if gt == 17 or gt == 18 then sig = sig .. ";" .. gt .. "@" .. i end
			end
		end
		lean.full_at = state.logic_frames
		if force or sig ~= lean.full_sig then
			lean.full_sig = sig
			local cells = {}
			for i = 0, room:GetGridSize() - 1 do
				local c = room:GetGridCollision(i)
				if c ~= 0 and c ~= 4 then cells[#cells + 1] = i end   -- not empty, not a wall
			end
			lean.cells, lean.part_sig = cells, nil
			changed = true
		end
	end
	local parts, cells = {}, lean.cells
	for k = 1, #cells do parts[k] = room:GetGridCollision(cells[k]) end
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		parts[#parts + 1] = d and d:GetVariant() or -1
	end
	local sig = table.concat(parts, ",")
	if lean.part_sig ~= nil and sig ~= lean.part_sig then changed = true end
	lean.part_sig = sig
	-- 2026-10-07: every decision, what the terrain block's grid records hold (terrain_content_sig): a grid entity's
	-- state or variant changing without a collision change (a poop or a cobweb hit by a bomb or a tear, a trapdoor
	-- appearing) is resent at once, not at the next full pass, so a step's terrain is always the current one (as a
	-- restored clone's, whose first observation carries it anew)
	local content = terrain_content_sig(room)
	if lean.content_sig ~= nil and content ~= lean.content_sig then changed = true end
	lean.content_sig = content
	if changed then lean.full_sig = nil end   -- the next full pass re-reads the cells to watch
	return changed
end

-- Native obs (2026-10-03, on top of abp-0.2.15; VERSION is unchanged because the Python client checks it). abp_turbo
-- (getenv "ABP_OBS_INIT") registers two Lua C functions that read the engine through the same getters luabridge calls
-- for the Lua code below: abp_native_lean(logic_frames, extra_flags), the fixed part of the observation (header, room,
-- totals, players, doors, entity records; nil, reason when a laser is in the room: its samples are built in Lua), and
-- abp_native_terrain(logic_frames, force) -> resend, room index, clear: lean_terrain_changed with its own state, plus
-- what the map block needs. The terrain and map blocks themselves are still built here (rare). Mode (env
-- ISAAC_RL_NATIVE_OBS, or the `native_obs` command): 0 the Lua code only, 1 native when available (default), 2 check:
-- both, the Lua result is sent, ABP_NATIVE_OBS counts the comparisons and keeps the first difference. Changing the
-- mode forces a full terrain pass (each path keeps its own terrain state). While a player has a cancelled lethal hit
-- (state.lethal) the Lua code builds the observation.
local native_lean, native_terrain = nil, nil
if getenv("ABP_OBS_INIT") == "1" and type(abp_native_lean) == "function" and type(abp_native_terrain) == "function" then
	native_lean, native_terrain = abp_native_lean, abp_native_terrain
end
-- Post-update skip (2026-10-04, frame-cost work, same VERSION: nothing on the wire changes). abp_turbo puts a gate on the
-- engine's one call of LuaEngine::PostUpdate (Game::Update); abp_pu_arm(n) makes it answer the next n calls itself, with
-- what this file's MC_POST_UPDATE callback does on a step's inner frames in lean mode with native input: count the frame,
-- count the step down, and on its first frame clear the pressed-this-frame actions (input_triggered = 0). A step line
-- arms rep - 1 frames (begin_step_line); MC_POST_UPDATE first takes the count (abp_pu_take) and adds it to
-- state.logic_frames and subtracts it from state.frames_left; a room change disarms it (MC_POST_NEW_ROOM ends the step).
-- Only with lean mode, native input and a client, without a duel. Off unless ISAAC_RL_PU_SKIP=1 or ABP_PU.enabled = true.
local pu_arm, pu_take = nil, nil
if native_lean ~= nil and type(abp_pu_arm) == "function" and type(abp_pu_take) == "function" then
	pu_arm, pu_take = abp_pu_arm, abp_pu_take
end
ABP_PU = { available = pu_arm ~= nil, enabled = pu_arm ~= nil and getenv("ISAAC_RL_PU_SKIP") == "1", armed = 0 }
-- 2026-10-05 (the teacher's cost): ISAAC_RL_PU_PLAY=1 (or ABP_PU.play = true) arms the same skip for each action of a
-- play batch (the teacher's restore replay, fork_many's clones, the clone that plays the last batch itself): its inner
-- frames are those of a step line (no goal; stop_clear, the room-clear stop of play, is checked at an action's end
-- only), so the callback does the same on them. Needs ABP_PU.enabled; not when a stale step stop_clear is set.
ABP_PU.play = ABP_PU.enabled and getenv("ISAAC_RL_PU_PLAY") == "1"
ABP_PU.play_armed = 0
state.native_obs = math.tointeger(tonumber(getenv("ISAAC_RL_NATIVE_OBS") or "1")) or 1
ABP_NATIVE_OBS = { available = native_lean ~= nil, native = 0, fallbacks = 0, last_fallback = nil, checks = 0,
	mismatches = 0, first = nil }
-- Terrain block cache (2026-10-04, same VERSION: the bytes on the wire are unchanged). The lean terrain check resends
-- the terrain block every LEAN_FULL_EVERY logic frames once anything in the room changed (a full pass after a change
-- clears the full signature), each time built anew in Lua: the cells' tables and two json.encode, the costliest thing
-- left per decision on average. abp_turbo's abp_native_terrain_key(mark) compares everything the two JSON strings are
-- built from (the room's shape, size, index and stage, player 0's Size and CanFly, every cell's collision and its grid
-- entity's type, variant, state, collision class and position) with the key of the last block built (mark: and makes
-- this the new one); when equal, the last block's JSON strings are reused, with the new version number. Mode
-- (ISAAC_RL_TERRAIN_CACHE, or "terrain_cache" in the native_step command): 0 off, 1 on (default), 2 check (built anew
-- and compared with the cached strings: ABP_TERRAIN_CACHE.mismatches).
local native_tkey = native_lean ~= nil and type(abp_native_terrain_key) == "function" and abp_native_terrain_key or nil
state.terrain_cache = math.tointeger(tonumber(getenv("ISAAC_RL_TERRAIN_CACHE") or "1")) or 1
ABP_TERRAIN_CACHE = { available = native_tkey ~= nil, hits = 0, builds = 0, checks = 0, mismatches = 0 }

-- The entity part of the lean observation in Lua: (<I2 count .. records, laser blocks, monsters' HP, door-blocking HP,
-- door-blocking count).
local function lean_entities()
	local ents, lasers = {}, {}
	local monsters_hp, blocking_hp, blocking_count = 0, 0, 0
	local whitelist = cfg.effect_whitelist
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		local t = e.Type
		local npc = t >= 10 and t < 1000 and e:ToNPC() or nil
		if npc then
			local hp = e.HitPoints
			if hp > 0 then
				if is_monster(e) then monsters_hp = monsters_hp + hp end
				if e:CanShutDoors() and not e:IsDead() then
					blocking_hp = blocking_hp + hp
					blocking_count = blocking_count + 1
				end
			elseif e:CanShutDoors() and not e:IsDead() then
				blocking_count = blocking_count + 1
			end
		end
		if t ~= 1 and (t ~= 1000 or whitelist[e.Variant]) and e.Visible then
			local kind, flags, hpf, a, b, c, champion = 0, 0, 0.0, 0.0, 0.0, 0.0, -1
			if t == 2 then
				local x = e:ToTear()
				kind, a, b, c = 1, x.Height, x.FallingSpeed, x.Scale
			elseif t == 9 then
				local x = e:ToProjectile()
				kind, a, b, c = 2, x.Height, x.FallingSpeed, x.Scale
			elseif t == 7 then
				kind = 3
				lasers[#lasers + 1] = spack("<i8", e.Index) .. pack_laser(e)
			elseif t == TYPE_BOMB then
				kind = 4
			elseif t == 5 then
				kind = 5
			elseif npc then
				kind = 6
				local boss = e:IsBoss()
				if e:IsEnemy() then flags = flags + 1 end
				if e:IsVulnerableEnemy() then flags = flags + 2 end
				if boss then
					flags = flags + 4
					if e.MaxHitPoints > 0 then hpf = e.HitPoints / e.MaxHitPoints end
				end
				if e:CanShutDoors() and not e:IsDead() then flags = flags + 8 end
				champion = npc:GetChampionColorIdx()
			end
			local spr = e:GetSprite()
			local anim, names = 0, ANIMATIONS[t]
			if names then
				for k = 1, #names do
					if spr:IsPlaying(names[k]) or spr:IsFinished(names[k]) then anim = k; break end
				end
			end
			local pos, vel, sm = e.Position, e.Velocity, e.SizeMulti
			ents[#ents + 1] = spack(LEAN_ENTITY, e.Index, t, e.Variant, e.SubType, pos.X, pos.Y, vel.X, vel.Y, e.Size,
				sm.X, sm.Y, e.EntityCollisionClass, e.GridCollisionClass, e.CollisionDamage, spr:GetFrame(),
				e.FrameCount, kind, flags, anim, B(e.FlipX), champion, hpf, a, b, c)
		end
	end
	return spack("<I2", #ents) .. table.concat(ents), lasers, monsters_hp, blocking_hp, blocking_count
end

-- 2026-10-07 (charge): player i's charge counter (abp_turbo's abp_player_charge, registered with the native obs; 0
-- without it) and its weapon types (HasWeaponType 0 .. 10 as bits).
local function lean_charge(i)
	if type(abp_player_charge) ~= "function" then return 0 end
	return abp_player_charge(i) or 0
end

local function lean_weapons(p)
	local mask = 0
	for w = 0, 10 do
		if p:HasWeaponType(w) then mask = mask | (1 << w) end
	end
	return mask
end

local function lean_players(n_players)
	local players = {}
	for i = 0, n_players - 1 do
		local p = Isaac.GetPlayer(i)
		local pos, vel = p.Position, p.Velocity
		players[#players + 1] = spack(LEAN_PLAYER, p.Index, pos.X, pos.Y, vel.X, vel.Y, p.Size,
			p:GetHearts(), p:GetMaxHearts(), p:GetSoulHearts(), p:GetBlackHearts(), p:GetBoneHearts(),
			p:GetEternalHearts(), p:GetGoldenHearts(), p:GetExtraLives(), p:GetNumCoins(), p:GetNumBombs(),
			p:GetNumKeys(), p.Damage, p.MaxFireDelay, p.ShotSpeed, tear_range(p), p.MoveSpeed, p.Luck, B(p.CanFly),
			p:GetActiveItem(), p:GetActiveCharge(), B(p:GetActiveItem() ~= 0 and not p:NeedsCharge()),
			B(p:GetDamageCooldown() > 0), B(p.ControlsEnabled), p:GetHeadDirection(), p:GetFireDirection(),
			p:GetMovementDirection(), p:GetSprite():GetFrame(), B(p:IsDead() or state.lethal[p.Index] == true),
			p.FireDelay, p:GetDamageCooldown(), lean_charge(i), lean_weapons(p))
	end
	return players
end

local function lean_doors(room, level)
	local doors = {}
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		if d and visible_door(d) then
			local pos = d.Position
			local seen = 0
			pcall(function()
				local behind = level:GetRoomByIdx(d.TargetRoomIndex)
				if behind.VisitedCount > 0 then
					seen = behind.Clear and 3 or 1
				end
			end)
			doors[#doors + 1] = spack("<BBBBffi4", slot, B(d:IsOpen()), B(d:IsLocked()), seen, pos.X, pos.Y,
				math.tointeger(d.TargetRoomType) or 0)
		end
	end
	return doors
end

local function hex16(s, at)
	return (s:sub(at, at + 15):gsub(".", function(ch) return string.format("%02x", ch:byte()) end))
end

-- mode 2: one comparison of a native result with the Lua one.
local function native_check(what, equal, native, lua)
	local s = ABP_NATIVE_OBS
	s.checks = s.checks + 1
	if equal then return end
	s.mismatches = s.mismatches + 1
	if s.first ~= nil then return end
	if what == "fixed" then
		local at = 1
		while at <= #native and native:byte(at) == lua:byte(at) do at = at + 1 end
		s.first = string.format("%s frame=%d byte=%d len=%d/%d native=%s lua=%s", what, state.logic_frames, at - 1,
			#native, #lua, hex16(native, at), hex16(lua, at))
	else
		s.first = string.format("%s frame=%d native=%s lua=%s", what, state.logic_frames, native, lua)
	end
end

-- The fixed part in Lua: header .. room .. totals .. players .. doors .. entities .. lasers.
local function lean_fixed_lua(game, room, level, player0, here, extra_flags)
	local n_players = game:GetNumPlayers()
	local tl, br = room:GetTopLeftPos(), room:GetBottomRightPos()
	local players = lean_players(n_players)
	local doors = lean_doors(room, level)
	local ents_blob, lasers, monsters_hp, blocking_hp, blocking_count = lean_entities()
	local flags = (room:IsClear() and 1 or 0) + extra_flags + (game:IsPaused() and 4 or 0)
	return table.concat({
		spack("<I4I4I4BBBB", LEAN_MAGIC, state.logic_frames, game:GetFrameCount(), flags, n_players, #doors, #lasers),
		spack("<i4i4i4i4i4i4i4i4ffff", room:GetType(), room:GetRoomShape(), room:GetGridWidth(), room:GetGridHeight(),
			here, level:GetStage(), room:GetAliveEnemiesCount(), room:GetFrameCount(),
			tl.X, tl.Y, br.X, br.Y),
		spack("<dddd", player0:GetTotalDamageTaken(), monsters_hp, blocking_hp, blocking_count),
		table.concat(players), table.concat(doors), ents_blob, table.concat(lasers) })
end

local pack_lean, lean_blocks, lean_fixed   -- defined below; lean_profile times pack_lean

-- Diagnostic (the `lean_profile` command; use a throw-away clone: it moves the terrain checks' state): os.clock
-- milliseconds of each part of pack_lean in the current state, n times each.
local function lean_profile(n)
	local game = Game()
	local room, level = game:GetRoom(), game:GetLevel()
	local player0 = Isaac.GetPlayer(0)
	local ms, t = {}, 0
	local function timed(name, f)
		t = os.clock()
		for _ = 1, n do f() end
		ms[name] = 1000 * (os.clock() - t) / n
	end
	timed("players", function() lean_players(game:GetNumPlayers()) end)
	timed("doors", function() lean_doors(room, level) end)
	timed("entities_lua", lean_entities)
	timed("fixed_lua", function() lean_fixed_lua(game, room, level, player0, lean_room_index(level), 0) end)
	timed("terrain_step", function() lean_terrain_changed(room, player0, false) end)
	timed("terrain_full", function() lean.full_at = -100000; lean_terrain_changed(room, player0, false) end)
	timed("room_index", function() lean_room_index(level) end)
	timed("map", function() lean_map(level, lean_room_index(level)) end)
	local mode = state.native_obs
	if native_lean then
		timed("fixed_native", function() native_lean(state.logic_frames, 0) end)
		timed("terrain_native_step", function() native_terrain(state.logic_frames, false) end)
		local k = 0
		timed("terrain_native_full", function() k = k + 100; native_terrain(state.logic_frames + k, false) end)
	end
	for m = 0, (native_lean and 1 or 0) do
		state.native_obs = m
		timed("pack_lean_mode" .. m, function() pack_lean("step") end)
	end
	state.native_obs = mode
	-- the per-step command path (2026-10-04 breakdown): the JSON step command's parse, apply_action (with the native
	-- input push), the step branch's prelude, the "L" frame's header line
	local line = '{"cmd":"step","repeat":4,"move":3,"shoot":2,"bomb":0,"item":0}'
	timed("cmd_json_decode", function() json.decode(line) end)
	local cmd = json.decode(line)
	local held0, trig0 = state.held, state.triggered
	timed("apply_action", function() apply_action(cmd) end)
	timed("push_input", push_input)
	timed("step_prelude", function()
		state.nav = { hit = false, changed = false, min = -1 }
		state.was_clear = Game():GetRoom():IsClear()
	end)
	local payload = pack_lean("step")
	timed("frame_header", function() return "L " .. #payload .. " step " .. state.seq .. "\n" .. payload end)
	-- the terrain block (resent every LEAN_FULL_EVERY logic frames once anything changed): Lua tables, JSON
	timed("terrain_block_tables", function() terrain_record(room, player0, true); grid_records(room) end)
	timed("terrain_block_build", function()
		return spack("<I4s4s4", 1, json.encode(terrain_record(room, player0, true)), json.encode(grid_records(room)))
	end)
	if native_tkey then timed("terrain_key_native", function() native_tkey(false) end) end
	state.held, state.triggered = held0, trig0
	push_input()
	state.terrain_dirty = true
	return ms
end

local function lean_handles()
	local game = Game()
	return game, game:GetRoom(), game:GetLevel(), Isaac.GetPlayer(0)
end

function pack_lean(event)
	local mode = native_lean ~= nil and state.native_obs or 0
	local force = event ~= "step" or state.terrain_dirty
	local game, room, level, player0   -- mode 1 needs them only for the rare Lua-built parts
	if mode ~= 1 then game, room, level, player0 = lean_handles() end
	local resend, here, clear
	if mode ~= 0 then resend, here, clear = native_terrain(state.logic_frames, force) end
	if resend == nil or mode == 2 then
		if game == nil then game, room, level, player0 = lean_handles() end
		local lr = lean_terrain_changed(room, player0, force)
		local lh, lc = lean_room_index(level), room:IsClear()
		if resend ~= nil then
			native_check("terrain", resend == lr and here == lh and clear == lc,
				tostring(resend) .. "," .. tostring(here) .. "," .. tostring(clear),
				tostring(lr) .. "," .. tostring(lh) .. "," .. tostring(lc))
		end
		resend, here, clear = lr, lh, lc
	end
	state.terrain_dirty = false
	local terrain, map, inv = lean_blocks(event, resend, here, clear)
	return lean_fixed(mode, here, (resend and 2 or 0) + (map ~= "" and 8 or 0) + (inv ~= "" and 16 or 0)) .. terrain
		.. map .. inv
end

-- The terrain block (when the terrain check says resend) and the map block (when it is due and changed) of a lean
-- observation; "" for a block that is not sent.
function lean_blocks(event, resend, here, clear)
	local terrain = ""
	if resend then
		local _, room, _, player0 = lean_handles()
		terrain_cache.version = terrain_cache.version + 1
		terrain_cache.sig = nil
		local tc, mode = ABP_TERRAIN_CACHE, native_tkey ~= nil and state.terrain_cache or 0
		local same = false
		if mode ~= 0 then same = native_tkey(true) and lean.tblock ~= nil end
		local tj, gj
		if same and mode == 1 then
			tj, gj = lean.tblock[1], lean.tblock[2]
			tc.hits = tc.hits + 1
		else
			tj, gj = json.encode(terrain_record(room, player0, true)), json.encode(grid_records(room))
			tc.builds = tc.builds + 1
			if same then
				tc.checks = tc.checks + 1
				if tj ~= lean.tblock[1] or gj ~= lean.tblock[2] then tc.mismatches = tc.mismatches + 1 end
			end
			lean.tblock = mode ~= 0 and { tj, gj } or nil
		end
		terrain = spack("<I4s4s4", terrain_cache.version, tj, gj)
	end
	local map = ""
	local map_key = here * 2 + (clear and 1 or 0)
	-- items / run mode: a new floor's start room may have the index of the room the trapdoor was in
	if state.lean_items then map_key = map_key + 1000 * Game():GetLevel():GetStage() end
	if event ~= "step" or map_key ~= lean.map_key or state.logic_frames - lean.map_at >= LEAN_FULL_EVERY
		or state.logic_frames < lean.map_at then
		lean.map_key, lean.map_at = map_key, state.logic_frames
		local ok, block = pcall(lean_map, Game():GetLevel(), here)
		if not ok then block = spack("<B", 0) end
		if event ~= "step" or block ~= lean.map_sig then
			lean.map_sig = block
			map = block
		end
	end
	local inv = state.lean_items and lean_inventory(event, here) or ""
	return terrain, map, inv
end

-- The fixed part of a lean observation (header .. entities .. lasers): native in mode 1 / 2 when it can, else Lua.
function lean_fixed(mode, here, extra)
	if mode ~= 0 and next(state.lethal) == nil then
		local s = ABP_NATIVE_OBS
		local fixed, why = native_lean(state.logic_frames, extra)
		if fixed ~= nil then
			if mode == 1 then
				s.native = s.native + 1
				return fixed
			end
			local ref = lean_fixed_lua(Game(), Game():GetRoom(), Game():GetLevel(), Isaac.GetPlayer(0), here, extra)
			native_check("fixed", fixed == ref, fixed, ref)
			return ref
		end
		s.fallbacks = s.fallbacks + 1
		s.last_fallback = why
	end
	local game, room, level, player0 = lean_handles()
	return lean_fixed_lua(game, room, level, player0, here, extra)
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
	if state.lean then
		local ok, payload = pcall(pack_lean, event)
		if not ok then
			log("lean pack failed: " .. tostring(payload))
			return send({ type = "error", msg = "pack: " .. tostring(payload) })
		end
		return send_raw("L " .. #payload .. " " .. event .. " " .. state.seq .. "\n" .. payload)
	end
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
	push_input()
	state.goal, state.nav = nil, { hit = false, changed = false, min = -1 }   -- abp-0.2.13
	state.stop_clear = false
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
	-- 2026-10-06 (items): + 180 x the pill / card action (codes below 180 as before)
	apply_action({ move = code % 9, shoot = (code // 9) % 5, bomb = (code // 45) % 2, item = (code // 90) % 2,
		pill = (code // 180) % 2 })
	state.frames_left = math.max(1, math.floor(tonumber(p.repeats and p.repeats[p.i]) or p.rep))
	state.stats.steps = state.stats.steps + 1
	if ABP_PU.play and state.frames_left > 1 and state.lean and state.native_input and not state.stop_clear
		and (state.client ~= nil or state.headless) and duel.key == nil and state.goal == nil then
		pu_arm(state.frames_left - 1)   -- see ABP_PU.play
		ABP_PU.play_armed = ABP_PU.play_armed + 1
	end
end

-- Where a step observation would be sent: true when the batch goes on with its next action; false when it has ended
-- (the ok message is sent, the step observation follows).
local function play_continue()
	local p = state.play
	local player = Isaac.GetPlayer(0)
	local stop = nil
	if player ~= nil and (player:IsDead() or state.lethal[player.Index] == true) then stop = "dead"
	elseif state.nav.changed and p.stop_room then stop = "room"   -- abp-0.2.13 (2026-10-07: only with stop_room)
	elseif p.stop_clear and Game():GetRoom():IsClear() then stop = "clear"
	elseif p.i >= #p.codes then stop = "done" end
	if stop == nil then
		-- 2026-10-07: a room change ended the action under way (as it ends a step); the next action starts with a fresh
		-- record, as begin_step_line gives each step one
		if state.nav.changed then state.nav = { hit = false, changed = false, min = -1 } end
		play_next()
		return true
	end
	state.play = nil
	if state.headless then   -- fork_many: kept for the report, no message
		state.headless.played, state.headless.stop = p.i, stop
	else
		send({ type = "ok", cmd = "play", played = p.i, stop = stop })
	end
	return false
end

-- abp-0.2.14 fork, in the clone (abp_turbo ABP_FORK answered "0"): the parent's sockets are private dummies here and are
-- dropped; the clone connects to the address in the command, says hello there and takes its commands from that
-- connection, in the state the parent was in when it read the fork command. Without a connection it ends.
local function fork_clone(cmd)
	state.clone = true
	if tonumber(cmd.alarm) then getenv("ABP_ALARM:" .. math.floor(tonumber(cmd.alarm))) end   -- real seconds it may live
	-- reseed: the clone's global MT starts anew, so clones of one state differ in the game's later random draws
	if tonumber(cmd.reseed) then getenv("ABP_RESEED:" .. math.floor(tonumber(cmd.reseed))) end
	if cmd.lean ~= nil then state.lean = cmd.lean == true end
	-- 2026-10-04 (frame-cost work, off by default): ISAAC_RL_LUA_GCPAUSE=<n> sets the Lua collector's pause in a lean
	-- clone (collectgarbage "setpause"; Lua's default 200 = a cycle each time the heap doubles). A clone lives about an
	-- episode and makes ~8 KB of Lua garbage per decision on a ~2 MB heap; a larger pause means fewer marking passes over
	-- the heap it shares copy-on-write with its parent. Lua memory only: nothing the game reads.
	local gcpause = state.lean and math.tointeger(tonumber(getenv("ISAAC_RL_LUA_GCPAUSE") or "")) or nil
	if gcpause and gcpause > 0 then collectgarbage("setpause", gcpause) end
	state.terrain_dirty = true   -- the clone's first observation carries the terrain: its client may be a new one
	if state.lean_items then   -- items / run mode: and the map and inventory blocks (a restore clone's first observation
		lean.map_key, lean.map_sig, INV.sent = nil, nil, nil   -- is a play's step observation)
	end
	pcall(function() state.client:close() end)
	pcall(function() state.server:close() end)
	state.server = true   -- never accepts: MC_POST_UPDATE only checks that it is set
	state.client = nil
	local c = socket.connect(tostring(cmd.host or cfg.host), math.floor(tonumber(cmd.port) or 0))
	if not c then getenv("ABP_EXIT"); return end
	c:settimeout(nil)
	c:setoption("tcp-nodelay", true)
	state.client = c
	set_native_input(state.lean)   -- needs the connection: without a client the engine reads the real devices
	send({ type = "hello", version = VERSION, engine = "abplus-1.06", clone = true, tag = cmd.tag,
		game_frame = Game():GetFrameCount(), logic_frames = state.logic_frames, turbo = getenv("ABP_FORK_COUNTERS"),
		step_line = true, fork_many = FM_AVAILABLE, items = true, lean_items = state.lean_items })
end

-- fork_many (2026-10-04, same VERSION; hello says fork_many = true when abp_turbo has ABP_FM_*): the hindsight teacher's
-- children in one command. {"cmd":"fork_many","batches":[[code, ...], ...],"repeat":r,"alarm":s,"stop_clear":b} in lean
-- mode: one clone per batch, made one after the other from this state. A clone gets no connection: it plays its batch
-- exactly as a `play` command (same repeat and stop_clear) sent to a `fork` clone of this state would (state.headless
-- stands in for the client where the game's course depends on one: lethal-hit blocking, invincibility, input; native
-- input stays on as the parent had it), and at the frame the play's observation would be built it reports "<batch>
-- <player 0's total damage taken> <dead 0/1> <played> <stop>" through abp_turbo's pipe and ends. The parent answers
-- {"type":"ok","cmd":"fork_many","forked":n,"results":"<report>;<report>;..."} once every clone has reported or ended.
local function fork_headless(cmd, i, codes)
	state.clone = true
	if tonumber(cmd.alarm) then getenv("ABP_ALARM:" .. math.floor(tonumber(cmd.alarm))) end   -- real seconds it may live
	pcall(function() state.client:close() end)
	pcall(function() state.server:close() end)
	state.server = true   -- never accepts
	state.client = nil
	state.headless = { batch = i, played = 0, stop = "?", stat = cmd.stat == true }
	state.goal, state.nav = nil, { hit = false, changed = false, min = -1 }
	state.play = { codes = codes, repeats = nil, i = 0, stop_clear = cmd.stop_clear == true,
		stop_room = cmd.stop_room == true, rep = math.max(1, math.floor(tonumber(cmd["repeat"]) or cfg.default_repeat)) }
	play_next()
end

local function fork_many(cmd)
	local batches = cmd.batches
	local ok = FM_AVAILABLE and state.lean and state.native_input and duel.key == nil and type(batches) == "table"
		and #batches > 0
	for _, codes in ipairs(ok and batches or {}) do
		if type(codes) ~= "table" or #codes == 0 then ok = false end
	end
	if not ok then
		send({ type = "error", msg = "fork_many: needs lean mode, native input, abp_turbo ABP_FM and non-empty batches" })
		return false
	end
	if getenv("ABP_FM_OPEN") ~= "1" then
		send({ type = "error", msg = "fork_many: no pipe" })
		return false
	end
	-- self_last: this process plays the last batch itself (no fork for it) and answers at its end; it is no longer in
	-- the state it was cloned in afterwards (the teacher closes it)
	local own = cmd.self_last == true and #batches or nil
	local n = 0
	for i, codes in ipairs(batches) do
		if i == own then break end
		local answer = getenv("ABP_FORK")
		if answer == "0" then
			fork_headless(cmd, i, codes)
			return true   -- the clone: its frames run
		elseif answer == nil or answer == "-1" then
			own = nil
			break
		end
		n = n + 1
	end
	if own ~= nil then
		state.headless = { batch = own, played = 0, stop = "?", parent = true, forked = n, stat = cmd.stat == true }
		state.goal, state.nav = nil, { hit = false, changed = false, min = -1 }
		state.play = { codes = batches[own], repeats = nil, i = 0, stop_clear = cmd.stop_clear == true,
			stop_room = cmd.stop_room == true, rep = math.max(1, math.floor(tonumber(cmd["repeat"]) or cfg.default_repeat)) }
		play_next()
		return true   -- its frames run; headless_report answers
	end
	send({ type = "ok", cmd = "fork_many", forked = n, results = getenv("ABP_FM_COLLECT:" .. n) or "" })
	return false
end


-- pickup_block (2026-10-07, same VERSION; the counterfactual branches of tok_branch.py): {"cmd":"pickup_block",
-- "subtype":id,"x":..,"y":..} makes the collectible pedestal (pickup variant 100) holding that collectible nearest to
-- (x, y) (the player when absent) untakeable for the rest of this process and of its clones: the engine's
-- MC_PRE_PICKUP_COLLISION is answered false for player contacts with it, which in AB+ v1.06 keeps the collision (the
-- player bumps into it as into a shop item it cannot afford) and skips the pickup code (checked 2026-10-07: nil takes,
-- false keeps the pedestal and the player stands at it, true lets the player through and the pedestal vanishes).
-- Nothing else is written. The pedestal is known by its InitSeed and by (room index, position) (a room left and
-- entered again rebuilds it). The callback is registered at the first such command only: a process that never gets
-- one runs exactly as before. Answers {"type":"ok","cmd":"pickup_block","found":0/1,"seed":..,"subtype":..,"x":..,"y":..}.
local PICKUP_BLOCK = nil   -- { seeds = {InitSeed = true}, places = {{room, x, y}}, contacts = n }
local function pickup_blocked(pickup)
	local b = PICKUP_BLOCK
	if b.seeds[pickup.InitSeed] then return true end
	if #b.places == 0 then return false end
	local here = lean_room_index(Game():GetLevel())
	local pos = pickup.Position
	for _, pl in ipairs(b.places) do
		if pl[1] == here and math.abs(pos.X - pl[2]) < 1 and math.abs(pos.Y - pl[3]) < 1 then return true end
	end
	return false
end
local function pickup_block(cmd)
	local x, y = tonumber(cmd.x), tonumber(cmd.y)
	local p = Isaac.GetPlayer(0)
	if x == nil or y == nil then x, y = p.Position.X, p.Position.Y end
	local want = math.tointeger(tonumber(cmd.subtype))
	local best, best_d = nil, nil
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if e.Type == EntityType.ENTITY_PICKUP and e.Variant == 100 and (want == nil or e.SubType == want) then
			local d = (e.Position.X - x) ^ 2 + (e.Position.Y - y) ^ 2
			if best_d == nil or d < best_d then best, best_d = e, d end
		end
	end
	if best == nil then return { type = "ok", cmd = "pickup_block", found = 0 } end
	if PICKUP_BLOCK == nil then
		PICKUP_BLOCK = { seeds = {}, places = {}, contacts = 0 }
		mod:AddCallback(ModCallbacks.MC_PRE_PICKUP_COLLISION, function(_, pickup, other, low)
			if PICKUP_BLOCK == nil or other == nil or other.Type ~= EntityType.ENTITY_PLAYER then return nil end
			if pickup.Variant ~= 100 or not pickup_blocked(pickup) then return nil end
			PICKUP_BLOCK.contacts = PICKUP_BLOCK.contacts + 1
			return false
		end)
	end
	PICKUP_BLOCK.seeds[best.InitSeed] = true
	PICKUP_BLOCK.places[#PICKUP_BLOCK.places + 1] = { lean_room_index(Game():GetLevel()), best.Position.X, best.Position.Y }
	return { type = "ok", cmd = "pickup_block", found = 1, seed = best.InitSeed, subtype = best.SubType,
		x = best.Position.X, y = best.Position.Y, contacts = PICKUP_BLOCK.contacts }
end

-- Step line (2026-10-04, same VERSION; hello says step_line = true): "S <repeat> <move> <shoot> <bomb> <item>\n" is the
-- step command {"cmd":"step","repeat":..,"move":..,"shoot":..,"bomb":..,"item":..} without goal, stop_clear or duel
-- fields, without its JSON parse (json.decode of a step command cost 0.07-0.12 ms, more than the native observation).
-- 2026-10-06 (items): an optional sixth number is the pill / card action ("S <repeat> <move> <shoot> <bomb> <item>
-- <pill>"); without it 0.
local step_cmd = { move = 0, shoot = 0, bomb = 0, item = 0, pill = 0 }
local function begin_step_line(rep, move, shoot, bomb, item, pill)
	step_cmd.move, step_cmd.shoot, step_cmd.bomb, step_cmd.item, step_cmd.pill = move, shoot, bomb, item, pill or 0
	apply_action(step_cmd)
	state.goal = nil
	state.nav = { hit = false, changed = false, min = -1 }
	state.stop_clear = false
	state.was_clear = Game():GetRoom():IsClear()
	if duel.key ~= nil then duel.move, duel.shoot = 0, 0 end
	state.frames_left = math.max(1, rep)
	state.stats.steps = state.stats.steps + 1
	if ABP_PU.enabled and state.frames_left > 1 and state.lean and state.native_input and state.client ~= nil
		and not state.headless and duel.key == nil and state.play == nil then
		pu_arm(state.frames_left - 1)
		ABP_PU.armed = ABP_PU.armed + 1
	end
end

-- One command line; true when the frame loop goes on (a step, play, reset or close), false to wait for the next.
local function handle_command(line)
		if line:byte(1) == 83 then   -- "S": a step line
			local rep, move, shoot, bomb, item, pill = line:match("^S (%d+) (%d+) (%d+) (%d+) (%d+) (%d+)$")
			if rep == nil then
				rep, move, shoot, bomb, item = line:match("^S (%d+) (%d+) (%d+) (%d+) (%d+)$")
				pill = "0"
			end
			if rep == nil then
				send({ type = "error", msg = "bad step line" })
				return false
			end
			begin_step_line(math.tointeger(tonumber(rep)), math.tointeger(tonumber(move)),
				math.tointeger(tonumber(shoot)), math.tointeger(tonumber(bomb)), math.tointeger(tonumber(item)),
				math.tointeger(tonumber(pill)))
			return true
		end
		local ok, cmd = pcall(json.decode, line)
		if not ok or type(cmd) ~= "table" then
			send({ type = "error", msg = "bad json" })
		elseif cmd.cmd == "step" then
			apply_action(cmd)
			-- abp-0.2.13: the step's goal and a fresh record of what happens during it
			local g = cmd.goal
			state.goal = (type(g) == "table" and #g == 3) and { tonumber(g[1]), tonumber(g[2]), tonumber(g[3]) } or nil
			state.nav = { hit = false, changed = false, min = -1 }
			state.stop_clear = cmd.stop_clear == true
			state.was_clear = Game():GetRoom():IsClear()
			if duel.key ~= nil then   -- abp-0.2.9: the duel NPC's move and shoot for the frames of this step
				duel.move = math.max(0, math.min(8, math.floor(tonumber(cmd.duel_move) or 0)))
				duel.shoot = math.max(0, math.min(4, math.floor(tonumber(cmd.duel_shoot) or 0)))
			end
			state.frames_left = math.max(1, math.floor(tonumber(cmd["repeat"]) or cfg.default_repeat))
			state.stats.steps = state.stats.steps + 1
			return true
		elseif cmd.cmd == "play" then   -- abp-0.2.12
			if type(cmd.actions) ~= "table" or #cmd.actions == 0 then
				send({ type = "error", msg = "play: no actions" })
			elseif cmd.repeats ~= nil and (type(cmd.repeats) ~= "table" or #cmd.repeats ~= #cmd.actions) then
				send({ type = "error", msg = "play: repeats must match actions" })
			elseif duel.key ~= nil then
				send({ type = "error", msg = "play: not with the duel NPC" })
			else
				state.goal, state.nav = nil, { hit = false, changed = false, min = -1 }   -- abp-0.2.13
				state.play = { codes = cmd.actions, repeats = cmd.repeats, i = 0, stop_clear = cmd.stop_clear == true,
					stop_room = cmd.stop_room == true,
					rep = math.max(1, math.floor(tonumber(cmd["repeat"]) or cfg.default_repeat)) }
				play_next()
				return true
			end
		elseif cmd.cmd == "reset" then
			do_reset(cmd)
			return true
		elseif cmd.cmd == "lean" then   -- abp-0.2.14
			state.lean = cmd.enabled ~= false
			set_native_input(state.lean)
			state.terrain_dirty = true
			send({ type = "ok", cmd = "lean", enabled = state.lean })
		elseif cmd.cmd == "lean_items" then   -- 2026-10-06 (items, Phase A): the inventory block and run-mode map rules
			if cmd.enabled ~= nil then
				state.lean_items = cmd.enabled == true
				lean.map_key, lean.map_sig, INV.sent = nil, nil, nil
			end
			send({ type = "ok", cmd = "lean_items", enabled = state.lean_items, scans = INV.scans, checks = INV.checks,
				misses = INV.misses, first_miss = INV.first_miss, ids = inv_ids() })
		elseif cmd.cmd == "lean_profile" then   -- native obs: cost of each part of pack_lean (diagnostic)
			local ok, ms = pcall(lean_profile, math.max(1, math.floor(tonumber(cmd.n) or 100)))
			send(ok and { type = "ok", cmd = "lean_profile", ms = ms } or { type = "error", msg = tostring(ms) })
		elseif cmd.cmd == "native_obs" then   -- native obs: 0 Lua, 1 native when available, 2 check (pack_lean)
			if cmd.mode ~= nil then
				state.native_obs = math.tointeger(tonumber(cmd.mode)) or 1
				state.terrain_dirty = true   -- the paths keep their own terrain state: a full pass resyncs it
			end
			local s = ABP_NATIVE_OBS
			send({ type = "ok", cmd = "native_obs", mode = state.native_obs, available = s.available, checks = s.checks,
				mismatches = s.mismatches, fallbacks = s.fallbacks, last_fallback = s.last_fallback, native = s.native,
				first = s.first, turbo = getenv("ABP_OBS_STATUS") })
		elseif cmd.cmd == "fork_many" then   -- 2026-10-04 (see fork_many)
			if fork_many(cmd) then return true end
		elseif cmd.cmd == "pickup_block" then   -- 2026-10-07 (see pickup_block)
			local ok, answer = pcall(pickup_block, cmd)
			send(ok and answer or { type = "error", msg = "pickup_block: " .. tostring(answer) })
		elseif cmd.cmd == "native_step" then   -- native step: 0 off, 1 on when available (see native_step_obs)
			if cmd.mode ~= nil then state.native_step = math.tointeger(tonumber(cmd.mode)) or 1 end
			if cmd.terrain_cache ~= nil then state.terrain_cache = math.tointeger(tonumber(cmd.terrain_cache)) or 1 end
			local s, tc = ABP_NATIVE_STEP, ABP_TERRAIN_CACHE
			send({ type = "ok", cmd = "native_step", mode = state.native_step, available = s.available, steps = s.steps,
				lines = s.lines, other = s.other, fallbacks = s.fallbacks, blocks = s.blocks, last_fallback = s.last_fallback,
				turbo = getenv("ABP_OBS_STATUS"), prof = getenv("ABP_STEP_PROF"), terrain_cache = state.terrain_cache,
				tc_available = tc.available, tc_hits = tc.hits, tc_builds = tc.builds, tc_checks = tc.checks,
				tc_mismatches = tc.mismatches })
		elseif cmd.cmd == "fork" then   -- abp-0.2.14
			local answer = getenv("ABP_FORK")
			if answer == "0" then
				fork_clone(cmd)
			elseif answer == nil or answer == "-1" then
				send({ type = "error", msg = "fork: " .. (answer == nil and "abp_turbo has no ABP_FORK" or "fork failed") })
			else
				send({ type = "ok", cmd = "fork", pid = tonumber(answer), tag = cmd.tag })
			end
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
			set_native_input(state.lean)
			send({ type = "ok", cmd = "control", enabled = state.control })
		elseif cmd.cmd == "close" then
			disconnect("close requested")
			return true
		else
			send({ type = "error", msg = "unknown cmd" })
		end
		return false
end

local function wait_command()
	while state.client do
		local line, err = state.client:receive("*l")
		if not line then disconnect(err); return end
		if handle_command(line) then return end
	end
end

-- In a fork_many clone, where its play's observation would be built: the report, then the clone ends. In the parent
-- that played the last batch itself (self_last): the answer with every report, then the next command.
local function headless_report()
	local p = Isaac.GetPlayer(0)
	local dead = p:IsDead() or state.lethal[p.Index] == true
	local h = state.headless
	local line = string.format("%d %.17g %d %d %s", h.batch, p:GetTotalDamageTaken(), dead and 1 or 0, h.played, h.stop)
	-- 2026-10-05 (diagnostic, "stat":true in the command): abp_turbo's ABP_FORK_TIMES of this process after the report's
	-- fields (the teacher's cost breakdown; a client that does not ask gets the five fields as before). The parent
	-- collects first (and then waits for its clones to exit, so the times include their exits).
	if h.parent then
		state.headless = nil
		local res = getenv("ABP_FM_COLLECT:" .. h.forked .. (h.stat and ":w" or "")) or ""
		if h.stat then line = line .. " " .. (getenv("ABP_FORK_TIMES") or "") end
		send({ type = "ok", cmd = "fork_many", forked = h.forked, results = res .. line .. ";" })
		wait_command()
		return
	end
	if h.stat then line = line .. " " .. (getenv("ABP_FORK_TIMES") or "") end
	getenv("ABP_FM_RESULT:" .. line)
	getenv("ABP_EXIT")
end

-- Native step (2026-10-04, same VERSION: the bytes on the wire are unchanged). In lean mode with the native observation
-- (mode 1) and abp_turbo's abp_native_step, a step observation's fixed part is built and written to the client's socket
-- by abp_turbo, which then waits for the next command and takes it when it is a step line: no Lua string of the
-- payload, no luasocket send / receive, no JSON. The terrain check and the terrain / map blocks are pack_lean's (here);
-- when a block is sent, or the native fixed part hands over (a laser), the observation is built and sent as before.
-- Any other command is left in the socket for wait_command. ISAAC_RL_NATIVE_STEP=0 or the `native_step` command (mode
-- 0 / 1) switches it off / on; a client gets the step line (hello: step_line) either way.
local native_step = nil
if native_lean ~= nil and type(abp_native_step) == "function" then native_step = abp_native_step end
state.native_step = math.tointeger(tonumber(getenv("ISAAC_RL_NATIVE_STEP") or "1")) or 1
ABP_NATIVE_STEP = { available = native_step ~= nil, steps = 0, lines = 0, other = 0, fallbacks = 0, blocks = 0,
	last_fallback = nil }

local function send_lean_payload(build)
	local ok, payload = pcall(build)
	if not ok then
		log("lean pack failed: " .. tostring(payload))
		return send({ type = "error", msg = "pack: " .. tostring(payload) })
	end
	return send_raw("L " .. #payload .. " step " .. state.seq .. "\n" .. payload)
end

-- In place of send_obs("step") and wait_command(): false when the native step does not apply.
local function native_step_obs()
	local c = state.client
	if native_step == nil or state.native_step ~= 1 or not state.lean or state.native_obs ~= 1 or c == nil
		or next(state.lethal) ~= nil or c:dirty() then
		return false
	end
	local resend, here, clear = native_terrain(state.logic_frames, state.terrain_dirty)
	if resend == nil then return false end
	state.terrain_dirty = false
	state.seq = state.seq + 1
	state.credits = {}
	local s = ABP_NATIVE_STEP
	local consecutive = s.consecutive
	s.consecutive = false
	local ok, terrain, map, inv = pcall(lean_blocks, "step", resend, here, clear)
	if not ok or terrain ~= "" or map ~= "" or inv ~= "" then   -- a block goes with it: built and sent as pack_lean does
		s.blocks = s.blocks + 1
		local sent = send_lean_payload(function()
			if not ok then error(terrain) end
			return lean_fixed(1, here, (resend and 2 or 0) + (map ~= "" and 8 or 0) + (inv ~= "" and 16 or 0)) ..
				terrain .. map .. inv
		end)
		if sent then wait_command() end
		return true
	end
	local code, a, b, cc, d, e, f = native_step(c:getfd(), state.logic_frames, state.seq, consecutive)
	if code == 1 then   -- a step line: (repeat, move, shoot, bomb, item[, pill]: a library without it gives nil = 0)
		s.steps, s.lines, s.consecutive = s.steps + 1, s.lines + 1, true
		ABP_NATIVE_OBS.native = ABP_NATIVE_OBS.native + 1
		begin_step_line(a, b, cc, d, e, f)
	elseif code == 2 or code == 3 then   -- sent; another kind of command waits in the socket, or a bad step line
		s.steps, s.other = s.steps + 1, s.other + 1
		ABP_NATIVE_OBS.native = ABP_NATIVE_OBS.native + 1
		if code == 3 then send({ type = "error", msg = "bad step line" }) end
		wait_command()
	elseif code == 0 then   -- nothing sent: the native fixed part handed over (a laser)
		s.fallbacks, s.last_fallback = s.fallbacks + 1, a
		if send_lean_payload(function() return lean_fixed(1, here, 0) end) then wait_command() end
	else   -- the connection failed
		disconnect(a)
	end
	return true
end

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
	if pu_arm ~= nil then pu_arm(0) end   -- a room change ends the step at its next MC_POST_UPDATE (post-update skip)
	state.room_ready = true
	state.terrain_dirty = true
	-- abp-0.2.13: a room change during a step or play batch ends it at the new room's first logic frame
	if state.frames_left > 0 or state.play ~= nil then
		state.nav.changed = true
		state.goal = nil
	end
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
	if state.invincible and (state.client or state.headless) then
		state.combat.blocked_hits = state.combat.blocked_hits + 1
		return false
	end
	state.events.damage = state.events.damage + 1
	if duel.key ~= nil and not state.lethal[p.Index] then duel_player_damage(p, amount, source) end   -- abp-0.2.9
	if cfg.block_lethal and (state.client or state.headless) and p:GetExtraLives() == 0 and amount >= health_units(p) then
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
	if state.lean then return end   -- nothing resolves the tables below in lean mode
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
	if state.lean then return end
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
	if state.lean then return end
	state.lineage_born[#state.lineage_born + 1] = npc
end)
mod:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD, function(_, rng, pos)
	state.events.clears = state.events.clears + 1
	return nil
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
	if pu_take ~= nil then   -- the frames abp_turbo answered for this callback (post-update skip)
		local k = pu_take()
		if k > 0 then state.logic_frames, state.frames_left = state.logic_frames + k, state.frames_left - k end
	end
	state.logic_frames = state.logic_frames + 1
	if not state.lean then update_combat() end
	if not state.bind_attempted then try_bind() end
	if not state.server then return end
	if state.clone and not state.client and not state.headless then getenv("ABP_EXIT") end   -- abp-0.2.14
	if not state.client and not state.headless then
		try_accept()
		if not state.client then return end
	end
	if next(state.triggered) ~= nil then   -- (2026-10-04: no new table per frame when there is nothing to clear)
		state.triggered = {}
		if state.native_input then push_input() end
	end
	if state.pending_reset then
		if not state.room_ready then return end
		state.settle = state.settle - 1
		if state.settle > 0 then return end
		state.pending_reset = false
		if send_obs("reset") then wait_command() end
		return
	end
	if state.frames_left > 0 then
		-- abp-0.2.13: the goal check of this logic frame; a reached goal or a room change ends the step here
		local goal = state.goal
		local player = goal ~= nil and Isaac.GetPlayer(0) or nil   -- (2026-10-04: the player only with a goal)
		if goal ~= nil and player ~= nil then
			local d = player.Position:Distance(Vector(goal[1], goal[2]))
			if state.nav.min < 0 or d < state.nav.min then state.nav.min = d end
			if d <= goal[3] then state.nav.hit = true end
		end
		if state.nav.hit or state.nav.changed then state.frames_left = 1 end
		if state.stop_clear and not state.was_clear and Game():GetRoom():IsClear() then state.frames_left = 1 end
		state.frames_left = state.frames_left - 1
	end
	if state.frames_left == 0 then
		if state.play ~= nil and play_continue() then return end   -- abp-0.2.12
		if state.headless then   -- 2026-10-04 fork_many: a clone reports and ends, the parent answers
			headless_report()
			return
		end
		if native_step_obs() then return end   -- 2026-10-04
		if send_obs("step") then wait_command() end
	end
end)

log("loaded " .. VERSION .. " port " .. cfg.port .. " (binds on the first logic frame)")
