-- 状态快照（只读）：帧行、进房行、楼层行、道具行、开局头里的元数据，以及三层 digest。
-- 约定：位置/速度/HP/属性一律 ×100 定点整数（0.01 px）；不使用 string.format("%a")；
-- digest 用 string.pack("<f") 的精确位（引擎内部是 float32）。所有 API 调用都不推进游戏 RNG。
local M = {}

local game = Game()
local floor, tostring, concat = math.floor, tostring, table.concat
local spack, sunpack, srep = string.pack, string.unpack, string.rep

------------------------------------------------------------------ 序列化小工具
local ESC = {['"'] = '\\"', ['\\'] = '\\\\', ['\b'] = '\\b', ['\f'] = '\\f', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t'}
local function jstr(s)
	if s == nil then return "null" end
	s = tostring(s)
	return '"' .. s:gsub('[%c"\\]', function(c) return ESC[c] or ("\\u00" .. ("%02x"):format(c:byte())) end) .. '"'
end
M.jstr = jstr

-- 定点 ×100；NaN/Inf 写 null
local function fx(v)
	if v == nil or v ~= v or v > 1e13 or v < -1e13 then return "null" end
	return tostring(floor(v * 100 + 0.5))
end
M.fx = fx

local function jint(v)
	if v == nil then return "null" end
	if v == true then return "1" elseif v == false then return "0" end
	if math.type(v) == "float" then
		if v ~= v or v > 1e15 or v < -1e15 then return "null" end
		if v == floor(v) then return tostring(floor(v)) end
		return ("%.9g"):format(v)
	end
	return tostring(v)
end
M.jint = jint

-- 通用 JSON（只用于头/低频行，不在每帧热路径上）
local function jval(v)
	local t = type(v)
	if t == "nil" then return "null"
	elseif t == "boolean" then return v and "true" or "false"
	elseif t == "number" then return jint(v)
	elseif t == "string" then return jstr(v)
	elseif t == "table" then
		if v[1] ~= nil or next(v) == nil then
			local b = {}
			for i = 1, #v do b[i] = jval(v[i]) end
			return "[" .. concat(b, ",") .. "]"
		end
		local keys = {}
		for k in pairs(v) do keys[#keys + 1] = tostring(k) end
		table.sort(keys)
		local b = {}
		for _, k in ipairs(keys) do b[#b + 1] = jstr(k) .. ":" .. jval(v[k] == nil and v[tonumber(k)] or v[k]) end
		return "{" .. concat(b, ",") .. "}"
	end
	return jstr(tostring(v))
end
M.jval = jval

------------------------------------------------------------------ digest
local W16 = "<" .. srep("I4", 16)
local function hash(s, h)
	h = h or 2166136261
	local pad = (-#s) % 64
	if pad > 0 then s = s .. srep("\0", pad) end
	local i, n = 1, #s
	while i <= n do
		local w = {sunpack(W16, s, i)}
		for k = 1, 16 do h = ((h ~ w[k]) * 16777619) & 0xffffffff end
		i = w[17]
	end
	return h
end
M.hash = hash

------------------------------------------------------------------ API 探测（一次性；缺失的 getter 写 null）
local function probe(fn, obj)
	local ok, v = pcall(fn, obj)
	return ok and v ~= nil
end

-- 玩家行字段（顺序即文件里的列顺序，写进 header.fields.player）
local PLAYER_FIELDS = {
	{"ptype", function(p) return p:GetPlayerType() end},
	{"x", nil}, {"y", nil}, {"vx", nil}, {"vy", nil},
	{"hearts", function(p) return p:GetHearts() end},
	{"max_hearts", function(p) return p:GetMaxHearts() end},
	{"soul", function(p) return p:GetSoulHearts() end},
	{"black", function(p) return p:GetBlackHearts() end},
	{"bone", function(p) return p:GetBoneHearts() end},
	{"eternal", function(p) return p:GetEternalHearts() end},
	{"golden", function(p) return p:GetGoldenHearts() end},
	{"broken", function(p) return p:GetBrokenHearts() end},
	{"rotten", function(p) return p:GetRottenHearts() end},
	{"extra_lives", function(p) return p:GetExtraLives() end},
	{"coins", function(p) return p:GetNumCoins() end},
	{"bombs", function(p) return p:GetNumBombs() end},
	{"giga_bombs", function(p) return p:GetNumGigaBombs() end},
	{"keys", function(p) return p:GetNumKeys() end},
	{"golden_key", function(p) return p:HasGoldenKey() end},
	{"golden_bomb", function(p) return p:HasGoldenBomb() end},
	{"poop_mana", function(p) return p:GetPoopMana() end},
	{"damage", function(p) return p.Damage end, true},
	{"fire_delay", function(p) return p.FireDelay end, true},
	{"max_fire_delay", function(p) return p.MaxFireDelay end, true},
	{"shot_speed", function(p) return p.ShotSpeed end, true},
	{"tear_range", function(p) return p.TearRange end, true},
	{"move_speed", function(p) return p.MoveSpeed end, true},
	{"luck", function(p) return p.Luck end, true},
	{"tear_height", function(p) return p.TearHeight end, true},
	{"tear_falling_speed", function(p) return p.TearFallingSpeed end, true},
	{"can_fly", function(p) return p.CanFly end},
	{"active0", function(p) return p:GetActiveItem(0) end},
	{"charge0", function(p) return p:GetActiveCharge(0) end},
	{"battery0", function(p) return p:GetBatteryCharge(0) end},
	{"subcharge0", function(p) return p:GetActiveSubCharge(0) end},
	{"active1", function(p) return p:GetActiveItem(1) end},
	{"charge1", function(p) return p:GetActiveCharge(1) end},
	{"active2", function(p) return p:GetActiveItem(2) end},
	{"charge2", function(p) return p:GetActiveCharge(2) end},
	{"active3", function(p) return p:GetActiveItem(3) end},
	{"charge3", function(p) return p:GetActiveCharge(3) end},
	{"card0", function(p) return p:GetCard(0) end},
	{"card1", function(p) return p:GetCard(1) end},
	{"pill0", function(p) return p:GetPill(0) end},
	{"pill1", function(p) return p:GetPill(1) end},
	{"trinket0", function(p) return p:GetTrinket(0) end},
	{"trinket1", function(p) return p:GetTrinket(1) end},
	{"collectibles", function(p) return p:GetCollectibleCount() end},
	{"damage_cooldown", function(p) return p:GetDamageCooldown() end},
	{"total_damage_taken", function(p) return p:GetTotalDamageTaken() end},
	{"head_dir", function(p) return p:GetHeadDirection() end},
	{"fire_dir", function(p) return p:GetFireDirection() end},
	{"move_dir", function(p) return p:GetMovementDirection() end},
	{"aim_x", function(p) return p:GetAimDirection().X end, true},
	{"aim_y", function(p) return p:GetAimDirection().Y end, true},
	{"dead", function(p) return p:IsDead() end},
	{"controls", function(p) return p.ControlsEnabled end},
	{"visible", function(p) return p.Visible end},
	{"size", function(p) return p.Size end, true},
	{"controller", function(p) return p.ControllerIndex end},
	{"init_seed", function(p) return p.InitSeed end},
	{"drop_seed", function(p) return p.DropSeed end},
	{"tear_flags_l", function(p) return p.TearFlags.l end},
	{"tear_flags_h", function(p) return p.TearFlags.h end},
	{"anim", function(p) return p:GetSprite():GetAnimation() end, "s"},
	{"aframe", function(p) return p:GetSprite():GetFrame() end},
	{"overlay", function(p) return p:GetSprite():GetOverlayAnimation() end, "s"},
}
M.PLAYER_FIELD_NAMES = {"num"}
M.FIXED_FIELDS = {player = {"x", "y", "vx", "vy"}, entity = {"x", "y", "vx", "vy", "hp", "max_hp"}, entity_extra = {}}
for _, f in ipairs(PLAYER_FIELDS) do
	M.PLAYER_FIELD_NAMES[#M.PLAYER_FIELD_NAMES + 1] = f[1]
	if f[3] == true then M.FIXED_FIELDS.player[#M.FIXED_FIELDS.player + 1] = f[1] end
end

-- 实体行公共字段；按类型附加的字段见 ENTITY_EXTRA_NAMES
M.ENTITY_FIELD_NAMES = {"idx", "type", "variant", "subtype", "x", "y", "vx", "vy", "hp", "max_hp", "frame", "init_seed",
	"drop_seed", "spawner_type", "parent_idx", "visible", "coll", "anim", "aframe"}
-- 实体行 = 上面 19 个公共字段 + 该类型的附加字段（ENTITY_EXTRA_NAMES[类别]，按顺序平铺在行尾）
M.GRID_FIELD_NAMES = {"index", "type", "variant", "state", "collision", "var_data"}
M.DOOR_FIELD_NAMES = {"slot", "target_room_index", "target_room_type", "open", "locked", "state", "variant"}
M.ROOMDESC_FIELD_NAMES = {"list_index", "grid_index", "safe_grid_index", "spawn_seed", "decoration_seed", "award_seed",
	"type", "variant", "subtype", "shape", "name", "difficulty", "stage_id", "flags", "visited_count", "clear",
	"clear_count", "display_flags"}
M.ROOM_FRAME_FIELD_NAMES = {"room_frame", "clear", "alive_enemies", "room_index"}

local live_player_fields = nil -- {name, getter or false, kind} 通过探测
local function detect_player_fields(p)
	live_player_fields = {}
	for _, f in ipairs(PLAYER_FIELDS) do
		local g = f[2]
		if g and not probe(g, p) then g = false end
		live_player_fields[#live_player_fields + 1] = {f[1], g, f[3]}
	end
end
function M.missing_player_fields()
	local out = {}
	if live_player_fields then
		for _, f in ipairs(live_player_fields) do if f[2] == false then out[#out + 1] = f[1] end end
	end
	return out
end

------------------------------------------------------------------ 每局缓存（进房/开局重置）
local C = {}
function M.reset_run()
	C = {grid = {}, grid_keys = {}, doors = {}, rescan_at = 0, grid_sig = 0, door_sig = 0, items_count = {}, prev_p = {}, prev_e = {},
		ptr2num = {}, nplayers = -1,
		-- v2 外观：字符串字典（每局从 0 编号；新条目在第一次被写出的那一行的 "sd" 里给出）、上一帧外观行、文件名缓存、
		-- 网格/门精灵行
		sdict = {}, snext = 0, snew = {}, prev_pa = {}, prev_ea = {}, fcache = {}, pfcache = {}, gspr = {}, dspr = {}, dpre = {}}
	live_player_fields = nil
end
M.reset_run()

------------------------------------------------------------------ v2：外观字段（动画/精灵状态；不进 digest）
-- 格式版本：header.format = 2（schema 字符串仍是 isaac-run-v1，旧读取器可照常读旧字段，见 README 4.10）
M.FORMAT = 2
-- 外观行（玩家行 "pa" 与实体行 "ea"，key = 玩家编号 / 实体 Index）。×100 定点：sx..poy；字符串列用字典 id。
M.APPEARANCE_FIELD_NAMES = {"key", "file", "ov_anim", "ov_frame", "flip", "sx", "sy", "rot",
	"r", "g", "b", "a", "ro", "go", "bo", "sr", "sg", "sb", "sa", "sro", "sgo", "sbo", "ox", "oy", "pox", "poy"}
M.PLAYER_APPEARANCE_EXTRA = {"body_color", "head_color"}
M.APPEARANCE_FIXED = {"sx", "sy", "rot", "r", "g", "b", "a", "ro", "go", "bo", "sr", "sg", "sb", "sa", "sro", "sgo", "sbo",
	"ox", "oy", "pox", "poy"}
M.APPEARANCE_STRING_COLUMNS = {"file", "ov_anim"}
M.GRID_SPRITE_FIELD_NAMES = {"index", "file", "anim", "frame"}
M.DOOR_SPRITE_FIELD_NAMES = {"slot", "file", "anim", "frame", "rot", "ox", "oy"}

-- 字符串 -> 字典 id（新字符串记进 C.snew，由写这一行的函数输出为 "sd":[[id,"串"],...]）
local function sid(s)
	if s == nil then return nil end
	local id = C.sdict[s]
	if id then return id end
	id = C.snext
	C.snext = id + 1
	C.sdict[s] = id
	C.snew[#C.snew + 1] = "[" .. id .. "," .. jstr(s) .. "]"
	return id
end
local function take_snew()
	if #C.snew == 0 then return nil end
	local s = concat(C.snew, ",")
	C.snew = {}
	return s
end

local function q100(v)
	if v == nil or v ~= v or v > 1e13 or v < -1e13 then return nil end
	return floor(v * 100 + 0.5)
end

------------------------------------------------------------------ 玩家编号（MC_INPUT_ACTION 用）
function M.refresh_players()
	local n = game:GetNumPlayers()
	if n == C.nplayers and n == 1 then return end
	C.nplayers = n
	C.ptr2num = {}
	for i = 0, n - 1 do C.ptr2num[GetPtrHash(Isaac.GetPlayer(i))] = i end
end
function M.player_num(p)
	if C.nplayers == 1 then return 0 end
	return C.ptr2num[GetPtrHash(p)] or -2
end

------------------------------------------------------------------ 网格与门
local GRID_WALL = GridEntityType.GRID_WALL
local GRID_DOOR = GridEntityType.GRID_DOOR

local function grid_row(i, ge)
	local vd = 0
	local ok, v = pcall(function() return ge.VarData end)
	if ok and v then vd = v end
	return concat({i, ge:GetType(), ge:GetVariant(), ge.State, ge.CollisionClass, vd}, ",")
end

-- 全量扫描：返回 {index -> row 字符串}
local function grid_scan(room)
	local rows = {}
	for i = 0, room:GetGridSize() - 1 do
		local ge = room:GetGridEntity(i)
		if ge then rows[i] = grid_row(i, ge) end
	end
	return rows
end

local function door_rows(room)
	local rows = {}
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		if d then
			rows[slot] = concat({slot, d.TargetRoomIndex, d.TargetRoomType, d:IsOpen() and 1 or 0, d:IsLocked() and 1 or 0,
				d.State, d:GetVariant()}, ",")
		end
	end
	return rows
end

-- v2：网格精灵行 [index, file, anim, frame]（墙与门格不记；门的精灵见 door_spr_row）
local function grid_spr_row(i, ge)
	local ok, s = pcall(ge.GetSprite, ge)
	if not ok or not s then return nil end
	return i .. "," .. sid(s:GetFilename()) .. "," .. sid(s:GetAnimation()) .. "," .. s:GetFrame()
end
-- v2：门精灵行 [slot, file, anim, frame, rot, ox, oy]（rot/ox/oy ×100）。
-- 每帧只读动画名与帧号；文件/旋转/偏移在进房、动画名变化时重读（C.dpre[slot] = {动画名, 行前缀, 行后缀}）
local function door_spr_changes(room, force)
	local changes = nil
	for slot = 0, 7 do
		local d = room:GetDoor(slot)
		local old = C.dspr[slot]
		local r = nil
		if d then
			local ok, s = pcall(d.GetSprite, d)
			if ok and s then
				local a, f = s:GetAnimation(), s:GetFrame()
				local pre = C.dpre[slot]
				if force or not pre or pre[1] ~= a then
					local o = s.Offset
					pre = {a, slot .. "," .. sid(s:GetFilename()) .. "," .. sid(a) .. ",",
						"," .. (q100(s.Rotation) or "null") .. "," .. (q100(o.X) or "null") .. "," .. (q100(o.Y) or "null")}
					C.dpre[slot] = pre
				end
				r = pre[2] .. f .. pre[3]
			end
		end
		if not r then C.dpre[slot] = nil end
		if r ~= old then
			changes = changes or {}
			changes[#changes + 1] = r and ("[" .. r .. "]") or ("[" .. slot .. "]")
			C.dspr[slot] = r
		end
	end
	return changes
end
local function grid_spr_scan(room, indices)
	local changes = nil
	for _, i in ipairs(indices) do
		local ge = room:GetGridEntity(i)
		local r = nil
		if ge then
			local t = ge:GetType()
			if t ~= GRID_WALL and t ~= GRID_DOOR then r = grid_spr_row(i, ge) end
		end
		if r ~= C.gspr[i] then
			changes = changes or {}
			changes[#changes + 1] = r and ("[" .. r .. "]") or ("[" .. i .. "]")
			C.gspr[i] = r
		end
	end
	return changes
end

local function sorted_rows(t)
	local keys = {}
	for k in pairs(t) do keys[#keys + 1] = k end
	table.sort(keys)
	local out = {}
	for _, k in ipairs(keys) do out[#out + 1] = "[" .. t[k] .. "]" end
	return out
end

local function sig_of(t)
	local rows = sorted_rows(t)
	return hash(concat(rows, ";"))
end

-- 每帧网格/门变化检测：已知非墙非门格每帧比；每 15 帧全量重扫（新出现的格、墙变门等）
local function grid_changes(room, n)
	local changes = nil
	local rows = C.grid
	if n >= C.rescan_at then
		C.rescan_at = n + 15
		local cur = grid_scan(room)
		-- v2：全扫时全部格（含消失的格）都重查精灵
		local all = {}
		for i in pairs(cur) do all[#all + 1] = i end
		for i in pairs(rows) do if not cur[i] then all[#all + 1] = i end end
		C.gchanged = all
		for i, r in pairs(cur) do
			if rows[i] ~= r then changes = changes or {}; changes[#changes + 1] = "[" .. r .. "]" end
		end
		for i in pairs(rows) do
			if not cur[i] then changes = changes or {}; changes[#changes + 1] = "[" .. i .. "]" end
		end
		C.grid = cur
		C.grid_keys = {}
		for i, r in pairs(cur) do
			local t = tonumber(r:match("^%d+,(%d+)"))
			if t ~= GRID_WALL and t ~= GRID_DOOR then C.grid_keys[#C.grid_keys + 1] = i end
		end
	else
		for _, i in ipairs(C.grid_keys) do
			local ge = room:GetGridEntity(i)
			local r = ge and grid_row(i, ge) or nil
			if r ~= rows[i] then
				changes = changes or {}
				changes[#changes + 1] = r and ("[" .. r .. "]") or ("[" .. i .. "]")
				rows[i] = r
				C.gchanged = C.gchanged or {}
				C.gchanged[#C.gchanged + 1] = i
			end
		end
	end
	if changes then C.grid_sig = sig_of(C.grid) end
	return changes
end

local function door_changes(room)
	local cur = door_rows(room)
	local changes = nil
	for s, r in pairs(cur) do
		if C.doors[s] ~= r then changes = changes or {}; changes[#changes + 1] = "[" .. r .. "]" end
	end
	for s in pairs(C.doors) do
		if not cur[s] then changes = changes or {}; changes[#changes + 1] = "[" .. s .. "]" end
	end
	if changes then C.doors = cur; C.door_sig = sig_of(cur) end
	return changes
end

------------------------------------------------------------------ 房间 / 楼层 行
local function desc_row(d)
	local data = d.Data
	local function g(fn) local ok, v = pcall(fn); if ok then return v end return nil end
	local cells = {
		jint(d.ListIndex), jint(d.GridIndex), jint(d.SafeGridIndex), jint(d.SpawnSeed), jint(d.DecorationSeed), jint(d.AwardSeed),
		jint(data and data.Type), jint(data and data.Variant), jint(data and data.Subtype), jint(data and data.Shape),
		jstr(data and data.Name), jint(data and g(function() return data.Difficulty end)),
		jint(data and g(function() return data.StageID end)),
		jint(g(function() return d.Flags end)), jint(g(function() return d.VisitedCount end)),
		jint(g(function() return d.Clear end)), jint(g(function() return d.ClearCount end)),
		jint(g(function() return d.DisplayFlags end)),
	}
	return "[" .. concat(cells, ",") .. "]"
end

function M.level_line(n)
	local level = game:GetLevel()
	local rooms = level:GetRooms()
	local rows = {}
	for i = 0, rooms.Size - 1 do rows[#rows + 1] = desc_row(rooms:Get(i)) end
	local function g(fn) local ok, v = pcall(fn); if ok then return v end return nil end
	return concat({'{"k":"level","n":', n, ',"gf":', game:GetFrameCount(),
		',"stage":', level:GetStage(), ',"stage_type":', level:GetStageType(), ',"curses":', level:GetCurses(),
		',"name":', jstr(g(function() return level:GetName() end)),
		',"dungeon_seed":', jint(g(function() return level:GetDungeonPlacementSeed() end)),
		',"start_room":', jint(g(function() return level:GetStartingRoomIndex() end)),
		',"is_alt":', jint(g(function() return level:IsAltStage() end)),
		',"greed_wave":', jint(g(function() return level.GreedModeWave end)),
		',"rooms":[', concat(rows, ","), ']}'})
end

function M.room_line(n)
	local room, level = game:GetRoom(), game:GetLevel()
	local function g(fn) local ok, v = pcall(fn); if ok then return v end return nil end
	local desc = g(function() return level:GetCurrentRoomDesc() end)
	C.grid = grid_scan(room)
	C.grid_keys = {}
	for i, r in pairs(C.grid) do
		local t = tonumber(r:match("^%d+,(%d+)"))
		if t ~= GRID_WALL and t ~= GRID_DOOR then C.grid_keys[#C.grid_keys + 1] = i end
	end
	C.rescan_at = n + 15
	C.doors = door_rows(room)
	C.grid_sig = sig_of(C.grid)
	C.door_sig = sig_of(C.doors)
	local tl, br = room:GetTopLeftPos(), room:GetBottomRightPos()
	-- v2：网格精灵与门精灵（进房时全量；之后只写变化，见 frame_parts 的 gs/ds）
	C.gspr, C.dspr, C.dpre = {}, {}, {}
	local gidx = {}
	for i in pairs(C.grid) do gidx[#gidx + 1] = i end
	table.sort(gidx)
	grid_spr_scan(room, gidx)
	door_spr_changes(room, true)
	local gs_rows, ds_rows = sorted_rows(C.gspr), sorted_rows(C.dspr)
	local sd = take_snew()
	return concat({'{"k":"room","n":', n, ',"gf":', game:GetFrameCount(),
		',"room_index":', level:GetCurrentRoomIndex(),
		',"desc":', desc and desc_row(desc) or "null",
		',"seeds":[', room:GetSpawnSeed(), ',', room:GetDecorationSeed(), ',', room:GetAwardSeed(), ']',
		',"type":', room:GetType(), ',"shape":', room:GetRoomShape(),
		',"gw":', room:GetGridWidth(), ',"gh":', room:GetGridHeight(),
		',"top_left":[', fx(tl.X), ',', fx(tl.Y), '],"bottom_right":[', fx(br.X), ',', fx(br.Y), ']',
		',"clear":', jint(room:IsClear()), ',"room_frame":', room:GetFrameCount(),
		',"backdrop":', jint(g(function() return room:GetBackdropType() end)),
		',"stage":', level:GetStage(), ',"stage_type":', level:GetStageType(), ',"curses":', level:GetCurses(),
		',"grid":[', concat(sorted_rows(C.grid), ","), ']',
		',"doors":[', concat(sorted_rows(C.doors), ","), ']',
		sd and (',"sd":[' .. sd .. ']') or '',
		',"grid_spr":[', concat(gs_rows, ","), '],"door_spr":[', concat(ds_rows, ","), ']}'})
end

------------------------------------------------------------------ 道具清单
local max_item = nil
local function item_max()
	if max_item then return max_item end
	local ok, n = pcall(function() return Isaac.GetItemConfig():GetCollectibles().Size - 1 end)
	max_item = ok and n or 732
	return max_item
end

-- 道具清单行。重扫时机：开局（force，总是写）、关键帧、道具数或主动槽 0 变化；清单与上次不同才写。
function M.items_lines(n, force, kf)
	local out = nil
	C.items_list = C.items_list or {}
	C.items_sig = C.items_sig or {}
	for i = 0, game:GetNumPlayers() - 1 do
		local p = Isaac.GetPlayer(i)
		local cnt = p:GetCollectibleCount()
		local sig = cnt .. ":" .. p:GetActiveItem(0)
		if force or kf or C.items_sig[i] ~= sig then
			C.items_sig[i] = sig
			local items = {}
			for id = 1, item_max() do
				local k = p:GetCollectibleNum(id, true)
				if k and k > 0 then items[#items + 1] = "[" .. id .. "," .. k .. "]" end
			end
			local list = concat(items, ",")
			if force or C.items_list[i] ~= list then
				C.items_list[i] = list
				out = out or {}
				out[#out + 1] = concat({'{"k":"items","n":', n, ',"gf":', game:GetFrameCount(), ',"p":', i,
					',"count":', cnt, ',"items":[', list, ']}'})
			end
		end
	end
	return out
end

------------------------------------------------------------------ 帧状态
local ETYPE_PLAYER, ETYPE_TEAR, ETYPE_FAMILIAR, ETYPE_BOMB, ETYPE_PICKUP = 1, 2, 3, 4, 5
local ETYPE_LASER, ETYPE_KNIFE, ETYPE_PROJECTILE, ETYPE_EFFECT = 7, 8, 9, 1000

local function anim_of(spr)
	local ok, a = pcall(spr.GetAnimation, spr)
	if ok then return a end
	return nil
end

-- 按类型的附加字段（逐字段探测；读不到的字段写 null）。kind: "f" = ×100 定点浮点，"i" = 整数/布尔
local EXTRA = {
	npc = {cast = function(e) return e:ToNPC() end, fields = {
		{"state", function(x) return x.State end, "i"}, {"state_frame", function(x) return x.StateFrame end, "i"},
		{"i1", function(x) return x.I1 end, "i"}, {"i2", function(x) return x.I2 end, "i"},
		{"champion", function(x) return x:GetChampionColorIdx() end, "i"}, {"boss", function(x) return x:IsBoss() end, "i"},
		{"v1x", function(x) return x.V1.X end, "f"}, {"v1y", function(x) return x.V1.Y end, "f"},
		{"target_x", function(x) return x.TargetPosition.X end, "f"}, {"target_y", function(x) return x.TargetPosition.Y end, "f"},
		{"projectile_cooldown", function(x) return x.ProjectileCooldown end, "i"}}},
	tear = {cast = function(e) return e:ToTear() end, fields = {
		{"height", function(x) return x.Height end, "f"}, {"falling_speed", function(x) return x.FallingSpeed end, "f"},
		{"falling_accel", function(x) return x.FallingAcceleration end, "f"}, {"scale", function(x) return x.Scale end, "f"},
		{"tear_flags_l", function(x) return x.TearFlags.l end, "i"}, {"tear_flags_h", function(x) return x.TearFlags.h end, "i"}}},
	projectile = {cast = function(e) return e:ToProjectile() end, fields = {
		{"height", function(x) return x.Height end, "f"}, {"falling_speed", function(x) return x.FallingSpeed end, "f"},
		{"falling_accel", function(x) return x.FallingAccel end, "f"}, {"scale", function(x) return x.Scale end, "f"}}},
	pickup = {cast = function(e) return e:ToPickup() end, fields = {
		{"state", function(x) return x.State end, "i"}, {"price", function(x) return x.Price end, "i"},
		{"timeout", function(x) return x.Timeout end, "i"}, {"touched", function(x) return x.Touched end, "i"},
		{"wait", function(x) return x.Wait end, "i"}, {"charge", function(x) return x.Charge end, "i"}}},
	familiar = {cast = function(e) return e:ToFamiliar() end, fields = {
		{"state", function(x) return x.State end, "i"}, {"orbit_layer", function(x) return x.OrbitLayer end, "i"},
		{"coins", function(x) return x.Coins end, "i"}, {"keys", function(x) return x.Keys end, "i"},
		{"hearts", function(x) return x.Hearts end, "i"}}},
	effect = {cast = function(e) return e:ToEffect() end, fields = {
		{"state", function(x) return x.State end, "i"}, {"timeout", function(x) return x.Timeout end, "i"},
		{"life_span", function(x) return x.LifeSpan end, "i"}}},
	bomb = {cast = function(e) return e:ToBomb() end, fields = {
		{"explosion_damage", function(x) return x.ExplosionDamage end, "f"},
		{"radius_multiplier", function(x) return x.RadiusMultiplier end, "f"}, {"is_fetus", function(x) return x.IsFetus end, "i"}}},
	laser = {cast = function(e) return e:ToLaser() end, fields = {
		{"timeout", function(x) return x.Timeout end, "i"}, {"angle", function(x) return x.AngleDegrees end, "f"},
		{"radius", function(x) return x.Radius end, "f"}, {"max_distance", function(x) return x.MaxDistance end, "f"},
		{"shrink", function(x) return x.Shrink end, "i"}}},
	knife = {cast = function(e) return e:ToKnife() end, fields = {
		{"charge", function(x) return x.Charge end, "f"}, {"max_distance", function(x) return x.MaxDistance end, "f"},
		{"is_flying", function(x) return x:IsFlying() end, "i"}}},
}
local CAT_BY_TYPE = {[2] = "tear", [3] = "familiar", [4] = "bomb", [5] = "pickup", [7] = "laser", [8] = "knife",
	[9] = "projectile", [1000] = "effect"}
M.ENTITY_EXTRA_NAMES = {}
for cat, def in pairs(EXTRA) do
	local names = {}
	for _, f in ipairs(def.fields) do names[#names + 1] = f[1] end
	M.ENTITY_EXTRA_NAMES[cat] = names
	local fixed = {}
	for _, f in ipairs(def.fields) do if f[3] == "f" then fixed[#fixed + 1] = f[1] end end
	M.FIXED_FIELDS.entity_extra[cat] = fixed
end
M.ENTITY_EXTRA_TYPES = {npc = "10..999 中 ToNPC() 非空者", tear = 2, familiar = 3, bomb = 4, pickup = 5, laser = 7,
	knife = 8, projectile = 9, effect = 1000}

-- 行内值：整数、字符串或 NULL（JSON null）。浮点一律先 ×100 定点化。
local NULL = setmetatable({}, {__tostring = function() return "null" end})
M.NULL = NULL
local function fxv(v)
	if v == nil or v ~= v or v > 1e13 or v < -1e13 then return NULL end
	return floor(v * 100 + 0.5)
end
local function iv(v)
	if v == nil then return NULL end
	if v == true then return 1 elseif v == false then return 0 end
	if math.type(v) == "float" then
		if v ~= v or v > 1e15 or v < -1e15 then return NULL end
		return floor(v)
	end
	return v
end
local function cell(v)
	if v == NULL then return "null" end
	if type(v) == "string" then return jstr(v) end
	return tostring(v)
end

-- 实体附加字段：追加进 vals；浮点/整数原值追加进 fl/il（digest 用）
local probed = {} -- cat -> {getter or false, ...}
local function append_extra(e, t, vals, fl, il)
	local cat = CAT_BY_TYPE[t] or ((t >= 10 and t < 1000) and "npc" or nil)
	if not cat then return end
	local def = EXTRA[cat]
	local x = def.cast(e)
	if not x then return end
	local live = probed[cat]
	if not live then
		live = {}
		for i, f in ipairs(def.fields) do
			local ok, v = pcall(f[2], x)
			live[i] = (ok and v ~= nil) and f[2] or false
		end
		probed[cat] = live
	end
	for i, f in ipairs(def.fields) do
		local g = live[i]
		if not g then vals[#vals + 1] = NULL
		else
			local v = g(x)
			if f[3] == "f" then vals[#vals + 1] = fxv(v); fl[#fl + 1] = v
			else
				local w = iv(v)
				vals[#vals + 1] = w
				if w ~= NULL then il[#il + 1] = w end
			end
		end
	end
end

-- 行编码：关键帧或新出现的对象写完整行；否则写差分行 [key, [[列, 差], ...]]（两边都是整数时写差值，
-- 否则写新值）；完全没变只写 key。key = 实体 Index / 玩家编号（列 0）。
local function row_full(vals)
	local cells = {}
	for i = 1, #vals do cells[i] = cell(vals[i]) end
	return "[" .. concat(cells, ",") .. "]"
end
-- pc = x 列（1 起；y 在 pc+1，vx/vy 在 pc+2/pc+3）。位置列用预测差：上一帧 x、本帧 x、本帧 vx 都是整数时
-- 写 残差 = x − (上一帧 x + 本帧 vx)，残差为 0 不写（引擎每帧 位置 += 速度，残差多为 0 或 ±1）；否则按普通规则。
M.POSITION_COLUMNS = {player = 2, entity = 4} -- 0 起的 x 列号（写进 header.fields.position_prediction）
local function row_delta(prev, vals, pc)
	local d = nil
	for i = 2, #vals do
		local a, v = prev[i], vals[i]
		if (i == pc or i == pc + 1) and math.type(a) == "integer" and math.type(v) == "integer"
			and math.type(vals[i + 2]) == "integer" then
			local r = v - (a + vals[i + 2])
			if r ~= 0 then
				d = d or {}
				d[#d + 1] = "[" .. (i - 1) .. "," .. r .. "]"
			end
		elseif a ~= v then
			d = d or {}
			if math.type(a) == "integer" and math.type(v) == "integer" then
				d[#d + 1] = "[" .. (i - 1) .. "," .. (v - a) .. "]"
			else
				d[#d + 1] = "[" .. (i - 1) .. "," .. cell(v) .. "]"
			end
		end
	end
	if not d then return tostring(vals[1]) end
	return "[" .. vals[1] .. ",[" .. concat(d, ",") .. "]]"
end

-- v2 外观行：[key, file, ov_anim, ov_frame, flip, sx, sy, rot, 实体色 r g b a ro go bo, 精灵色 r g b a ro go bo,
-- ox, oy, pox, poy]（+ 玩家：body_color, head_color）。flip = FlipX + 2*FlipY。颜色/缩放/旋转/偏移 ×100。
-- 开销控制（每次 API 访问约 0.3–0.4 µs）：文件名（GetFilename）、FlipY、精灵偏移（Sprite.Offset）只在对象新出现、
-- 主动画名变化、关键帧时重读（fc = {init_seed, 动画名, 文件 id, FlipY 位, ox, oy} 缓存）；精灵色（Sprite.Color）只对
-- 特效（1000.x）与玩家每帧读，其余实体写 null（= 未读，查看器按白色处理）；叠加帧只在有叠加动画时读。
local function qn(v)
	if v == nil or v ~= v or v > 1e13 or v < -1e13 then return NULL end
	return floor(v * 100 + 0.5)
end
local function appearance(e, spr, key, anim, init, kf, fc, scolor)
	local c = fc[key]
	if kf or not c or c[1] ~= init or c[2] ~= anim then
		local o = spr.Offset
		c = {init, anim, sid(spr:GetFilename()), spr.FlipY and 2 or 0, qn(o.X), qn(o.Y)}
		fc[key] = c
	end
	local ova = spr:GetOverlayAnimation()
	local ovid, ovf = NULL, -1
	if ova ~= nil and ova ~= "" then ovid = sid(ova); ovf = spr:GetOverlayFrame() end
	local sc, col, po = spr.Scale, e.Color, e.PositionOffset
	local sr, sg, sb, sa, sro, sgo, sbo = NULL, NULL, NULL, NULL, NULL, NULL, NULL
	if scolor then
		local s = spr.Color
		sr, sg, sb, sa, sro, sgo, sbo = qn(s.R), qn(s.G), qn(s.B), qn(s.A), qn(s.RO), qn(s.GO), qn(s.BO)
	end
	return {key, c[3], ovid, ovf, (spr.FlipX and 1 or 0) + c[4], qn(sc.X), qn(sc.Y), qn(spr.Rotation),
		qn(col.R), qn(col.G), qn(col.B), qn(col.A), qn(col.RO), qn(col.GO), qn(col.BO),
		sr, sg, sb, sa, sro, sgo, sbo, c[5], c[6], qn(po.X), qn(po.Y)}
end
-- 稀疏差分：关键帧或新对象写完整行；否则只写变化的对象（差分行同 p/e 规则，无位置预测）；未变的对象不写
local function sparse_row(prev, vals, full)
	if full or not prev or #prev ~= #vals then return row_full(vals) end
	local r = row_delta(prev, vals, -10)
	if r == tostring(vals[1]) then return nil end
	return r
end

-- 构建一帧（追加 JSON 片段到 buf）。kf = 关键帧（开局、进房后第一帧、每 300 帧）。
-- digest 三层：1 = 玩家与房间；2 = + 非外观实体；3 = 全量（+ 特效、实体种子/Index、附加字段、网格、门）
-- omit_e = true：不写实体行（IRR_OBS=keyframe 的非关键帧；digest 照常计算）
-- dry = true：开销基准（harness bench）用，v2 的字典/外观缓存在结束时还原（否则基准里分配的字典条目不会写进文件）
local function copy_tbl(t) local o = {}; for k, v in pairs(t) do o[k] = v end; return o end
function M.frame_parts(n, buf, kf, omit_e, dry)
	local saved = nil
	if dry then
		saved = {sdict = copy_tbl(C.sdict), snext = C.snext, snew = copy_tbl(C.snew), prev_pa = C.prev_pa, prev_ea = C.prev_ea,
			fcache = copy_tbl(C.fcache), pfcache = copy_tbl(C.pfcache), gspr = copy_tbl(C.gspr), dspr = copy_tbl(C.dspr), dpre = copy_tbl(C.dpre)}
	end
	M.frame_parts_inner(n, buf, kf, omit_e)
	if saved then for k, v in pairs(saved) do C[k] = v end end
end
function M.frame_parts_inner(n, buf, kf, omit_e)
	local room, level = game:GetRoom(), game:GetLevel()
	-- v2：外观行（不进 digest）
	local prev_pa = kf and {} or C.prev_pa
	local prev_ea = (kf or omit_e) and {} or C.prev_ea
	local cur_pa, cur_ea, parow, earow = {}, {}, {}, {}
	local fc = C.fcache
	local APP = M.appearance_enabled ~= false   -- 只给开销基准对照用（false = v1 等价的工作量）
	local L1, L2, L3 = {}, {}, {}
	local rf, clear, alive, ridx = room:GetFrameCount(), room:IsClear(), room:GetAliveEnemiesCount(), level:GetCurrentRoomIndex()
	buf[#buf + 1] = ',"r":[' .. rf .. "," .. (clear and 1 or 0) .. "," .. alive .. "," .. ridx .. "]"
	-- 第 1 层与第一阶段的"玩家与房间"口径一致：帧计数、层、房间号、三种房间种子、清理状态 + 玩家
	L1[#L1 + 1] = spack("<jjjjjjjjj", game:GetFrameCount(), rf, clear and 1 or 0, ridx, level:GetStage(), level:GetStageType(),
		room:GetSpawnSeed(), room:GetDecorationSeed(), room:GetAwardSeed())

	-- 玩家
	local prev_p = kf and {} or C.prev_p
	local cur_p = {}
	local prow = {}
	for i = 0, game:GetNumPlayers() - 1 do
		local p = Isaac.GetPlayer(i)
		if not live_player_fields then detect_player_fields(p) end
		local pos, vel = p.Position, p.Velocity
		local vals = {i}
		local fl, il = {pos.X, pos.Y, vel.X, vel.Y}, {i}
		for _, f in ipairs(live_player_fields) do
			local name, g, kind = f[1], f[2], f[3]
			if name == "x" then vals[#vals + 1] = fxv(pos.X)
			elseif name == "y" then vals[#vals + 1] = fxv(pos.Y)
			elseif name == "vx" then vals[#vals + 1] = fxv(vel.X)
			elseif name == "vy" then vals[#vals + 1] = fxv(vel.Y)
			elseif g == false then vals[#vals + 1] = NULL
			else
				local v = g(p)
				if kind == true then vals[#vals + 1] = fxv(v); fl[#fl + 1] = v
				elseif kind == "s" then vals[#vals + 1] = (v == nil) and NULL or tostring(v)
				else
					local w = iv(v)
					vals[#vals + 1] = w
					if w ~= NULL then il[#il + 1] = w end
				end
			end
		end
		local pv = prev_p[i]
		prow[#prow + 1] = (pv and #pv == #vals) and row_delta(pv, vals, 3) or row_full(vals)
		cur_p[i] = vals
		if APP then -- v2 玩家外观（key = 玩家编号）
			local spr = p:GetSprite()
			local av = appearance(p, spr, i, spr:GetAnimation(), p.InitSeed, kf, C.pfcache, true)
			av[#av + 1] = p:GetBodyColor()
			av[#av + 1] = p:GetHeadColor()
			local r = sparse_row(prev_pa[i], av, kf)
			if r then parow[#parow + 1] = r end
			cur_pa[i] = av
		end
		L1[#L1 + 1] = spack("<" .. srep("f", #fl), table.unpack(fl))
		L1[#L1 + 1] = spack("<" .. srep("j", #il), table.unpack(il))
	end
	C.prev_p = cur_p
	buf[#buf + 1] = ',"p":[' .. concat(prow, ",") .. "]"

	-- 实体
	local prev_e = kf and {} or C.prev_e
	local cur_e = {}
	local erow = {}
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		local t = e.Type
		if t ~= ETYPE_PLAYER then
			local pos, vel = e.Position, e.Velocity
			local spr = e:GetSprite()
			local parent = e.Parent
			local hp, mhp = e.HitPoints, e.MaxHitPoints
			local idx, init = e.Index, e.InitSeed
			local anim = anim_of(spr)
			local vals = {idx, t, e.Variant, e.SubType, fxv(pos.X), fxv(pos.Y), fxv(vel.X), fxv(vel.Y), fxv(hp), fxv(mhp),
				e.FrameCount, init, e.DropSeed, e.SpawnerType, parent and parent.Index or -1, e.Visible and 1 or 0,
				e.EntityCollisionClass, anim == nil and NULL or anim, spr:GetFrame()}
			local fl, il = {}, {}
			local ok, err = pcall(append_extra, e, t, vals, fl, il)
			if not ok then vals[#vals + 1] = NULL end
			local pv = prev_e[idx]
			local same = pv and pv[12] == init and #pv == #vals
			erow[#erow + 1] = same and row_delta(pv, vals, 5) or row_full(vals)
			cur_e[idx] = vals
			if APP and not omit_e then -- v2 实体外观（key = Index；与 e 行同时换成完整行）
				local av = appearance(e, spr, idx, anim, init, kf, fc, t == ETYPE_EFFECT)
				local pa = prev_ea[idx]
				local r = sparse_row(pa, av, kf or not same or not pa)
				if r then earow[#earow + 1] = r end
				cur_ea[idx] = av
			end
			-- 第 2 层与第一阶段的"非外观实体"口径一致：类型/位置/速度/HP/NPC State/StateFrame/FrameCount；
			-- Index、种子和其余附加字段只进第 3 层（特效生成次数不同会让后续实体的 Index 整体错位）
			local st, sf = -1, -1
			if t >= 10 and t < 1000 and #vals >= 21 then st, sf = vals[20], vals[21] end
			if st == NULL then st = -1 end
			if sf == NULL then sf = -1 end
			local core = spack("<ffffffjjjjjj", pos.X, pos.Y, vel.X, vel.Y, hp, mhp, t, e.Variant, e.SubType, e.FrameCount, st, sf)
			if t == ETYPE_EFFECT then L3[#L3 + 1] = core else L2[#L2 + 1] = core end
			L3[#L3 + 1] = spack("<jjj", init, e.DropSeed, idx)
			if #fl > 0 then L3[#L3 + 1] = spack("<" .. srep("f", #fl), table.unpack(fl)) end
			if #il > 0 then L3[#L3 + 1] = spack("<" .. srep("j", #il), table.unpack(il)) end
		end
	end
	C.prev_e = cur_e
	if not omit_e then buf[#buf + 1] = ',"e":[' .. concat(erow, ",") .. "]" end
	C.prev_pa = cur_pa
	C.prev_ea = cur_ea
	-- 文件名缓存只留本帧还在的对象
	if kf then
		local keep = {}
		for k in pairs(cur_ea) do keep[k] = fc[k] end
		C.fcache = keep
	end

	-- 网格/门：只写变化
	C.gchanged = nil
	local gch = grid_changes(room, n)
	local dch = door_changes(room)
	-- v2：网格精灵（网格行变化的格、每 15 帧全扫时的全部格）与门精灵（每帧）
	local gsc = nil
	local dsc = nil
	if APP then
		if C.gchanged then gsc = grid_spr_scan(room, C.gchanged) end
		dsc = door_spr_changes(room, false)
	end
	local sd = take_snew()
	if sd then buf[#buf + 1] = ',"sd":[' .. sd .. "]" end
	if #parow > 0 then buf[#buf + 1] = ',"pa":[' .. concat(parow, ",") .. "]" end
	if #earow > 0 then buf[#buf + 1] = ',"ea":[' .. concat(earow, ",") .. "]" end
	if gch then buf[#buf + 1] = ',"g":[' .. concat(gch, ",") .. "]" end
	if dch then buf[#buf + 1] = ',"dr":[' .. concat(dch, ",") .. "]" end
	if gsc then buf[#buf + 1] = ',"gs":[' .. concat(gsc, ",") .. "]" end
	if dsc then buf[#buf + 1] = ',"ds":[' .. concat(dsc, ",") .. "]" end
	L3[#L3 + 1] = spack("<jj", C.grid_sig, C.door_sig)

	local d1 = hash(concat(L1))
	local d2 = hash(concat(L2), d1)
	local d3 = hash(concat(L3), d2)
	buf[#buf + 1] = ',"dg":[' .. d1 .. "," .. d2 .. "," .. d3 .. "]"
end

------------------------------------------------------------------ 开局元数据（D10 中 Lua 能读的部分）
local OPTION_KEYS = {"MouseControl", "ChargeBars", "Language", "Fullscreen", "MusicVolume", "SFXVolume", "MapOpacity",
	"Filter", "Gamma", "Exposure", "PauseOnFocusLost", "CameraStyle", "HUDOffset", "FoundHUD", "BulletVisibility",
	"ConsoleFont", "FadedConsoleDisplay", "SaveCommandHistory", "MaxScale", "MaxRenderScale", "VSync", "RumbleEnabled",
	"AnnouncerVoiceMode", "ExtraHUDStyle", "DisplayPopups", "EnableDebugConsole", "JacobEsauControls", "MusicEnabled"}

function M.options()
	local o = {}
	if type(Options) ~= "table" and type(Options) ~= "userdata" then return o end
	for _, k in ipairs(OPTION_KEYS) do
		local ok, v = pcall(function() return Options[k] end)
		if ok and v ~= nil then o[k] = v end
	end
	return o
end

function M.meta()
	local seeds = game:GetSeeds()
	local function g(fn) local ok, v = pcall(fn); if ok then return v end return nil end
	local eff = {}
	if SeedEffect then
		local names = {}
		for name in pairs(SeedEffect) do names[#names + 1] = name end
		table.sort(names)
		for _, name in ipairs(names) do
			if g(function() return seeds:HasSeedEffect(SeedEffect[name]) end) == true then eff[#eff + 1] = name end
		end
	end
	local m = {
		REPENTANCE = REPENTANCE, REPENTANCE_PLUS = REPENTANCE_PLUS, lua_version = _VERSION,
		-- 全局是否定义（nil 在表里存不住，这里写成字符串 "nil"）
		globals = {REPENTANCE = tostring(REPENTANCE), REPENTANCE_PLUS = tostring(REPENTANCE_PLUS)},
		seeds = {start_seed = g(function() return seeds:GetStartSeed() end),
			start_seed_string = g(function() return seeds:GetStartSeedString() end),
			player_init_seed = g(function() return seeds:GetPlayerInitSeed() end),
			is_custom_run = g(function() return seeds:IsCustomRun() end),
			count_seed_effects = g(function() return seeds:CountSeedEffects() end),
			seed_effects = eff},
		difficulty = g(function() return game.Difficulty end),
		challenge = g(function() return game.Challenge end),
		victory_lap = g(function() return game:GetVictoryLap() end),
		greed_mode = g(function() return game:IsGreedMode() end),
		time_counter = g(function() return game.TimeCounter end),
		blue_womb_par_time = g(function() return game.BlueWombParTime end),
		num_players = game:GetNumPlayers(),
		options = M.options(),
		players = {},
	}
	for i = 0, game:GetNumPlayers() - 1 do
		local p = Isaac.GetPlayer(i)
		m.players[#m.players + 1] = {num = i, type = p:GetPlayerType(), name = g(function() return p:GetName() end),
			controller_index = g(function() return p.ControllerIndex end), init_seed = p.InitSeed, drop_seed = p.DropSeed,
			index = p.Index}
	end
	return m
end

return M
