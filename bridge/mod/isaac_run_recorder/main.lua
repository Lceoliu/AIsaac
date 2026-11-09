-- Isaac Run Recorder：Repentance v1.7.8a.J746 整局被动录制（DESIGN_BRIEF §6 D1–D12）。
-- 必须以 --luadebug 启动（需要 io/os）。输出目录 = 环境变量 IRR_DIR，否则 <mod 目录>/recordings。
-- 模式：
--   record（默认）：被动。MC_INPUT_ACTION 只读并返回 nil；MC_PRE_ENTITY_SPAWN 只记录不改种子；不执行任何命令。
--   harness：<IRR_DIR>/irr_control.json 存在时加载测试驱动（harness.lua），只用于自动化测试。
--   replay：另有 <IRR_DIR>/irr_replay.lua 时，harness 开出的局按记录注入输入并钉回生成种子（回归测试）。
-- 每局一个文件 run-<proc>-<NNNN>.jsonl：MC_POST_GAME_STARTED 开、MC_PRE_GAME_EXIT / 死亡 / 下一局开始时关。
-- 格式见 README.md。
local mod = RegisterMod("Isaac Run Recorder", 1)
local VERSION = "0.4.0"
local SCHEMA = "isaac-run-v1"
local game = Game()
local concat = table.concat

------------------------------------------------------------------ 加载子模块
local SRC = nil
pcall(function()
	local s = debug.getinfo(1, "S").source
	if s:sub(1, 1) == "@" then s = s:sub(2) end
	SRC = s:match("^(.*)[/\\][^/\\]*$")
end)
local function load_module(name)
	if SRC then
		local ok, m = pcall(dofile, SRC .. "/" .. name .. ".lua")
		if ok then return m end
	end
	return include(name)
end
local ST = load_module("state")
local IN = load_module("inputs")
IN.player_num = ST.player_num
-- 0.4.0（phase10）：第三方 mod 的"效果"调用（api 行，api.lua）。IRR_API=0 不装包装
local API = load_module("api")

-- 读状态时引擎内部可能再触发 MC_INPUT_ACTION（例如某些 getter 会查询输入），IN.reading 让这些查询被忽略
local function quiet(fn, ...)
	IN.reading = true
	local ok, a = pcall(fn, ...)
	IN.reading = false
	if not ok then error(a, 0) end
	return a
end


------------------------------------------------------------------ 输出目录与模式
local function getenv(k) local ok, v = pcall(os.getenv, k); if ok then return v end return nil end
local OUT = (getenv("IRR_DIR") or ((SRC or ".") .. "/recordings")):gsub("\\", "/")
local PROC = getenv("IRR_PROC") or tostring(os.time())
local HUD = getenv("IRR_HUD") ~= "0"
-- E5：观测模式。full（默认）= 每帧完整观测；keyframe = 实体行只在关键帧写，其余帧只写 mt/玩家行/输入/生成种子/
-- 房间头/网格与门的变化/digest（实体状态由回放重建）
local OBS = (getenv("IRR_OBS") == "keyframe") and "keyframe" or "full"
-- E3/E4/E7：全局 MT reseed 与每帧 MT 校验（需要注入的 rr_mt.dll；没有就 available=false，照常录制）
local MT = load_module("mt")
pcall(MT.init, getenv, ST.hash)
local function read_file(path)
	local fh = io.open(path, "r")
	if not fh then return nil end
	local s = fh:read("*a"); fh:close()
	return s
end

local S = {runs = 0, renders = 0, paused = false, last_status = -1e9, err = nil, errors = 0, prespawn = {}, writable = false}
local R = nil -- 当前局
local H = nil -- harness
local RP = nil -- replay 对象
local W = nil -- 观战/实录导出（watch.lua；只在回放且有 irr_watch.json 时加载）

local function append_file(name, text)
	local fh = io.open(OUT .. "/" .. name, "a")
	if fh then fh:write(text, "\n"); fh:close() end
end

local function fail(where, e)
	S.err = where .. ": " .. tostring(e)
	S.errors = S.errors + 1
	Isaac.DebugString("[IRR] " .. S.err)
	if S.errors <= 200 then
		append_file("errors.jsonl", '{"where":' .. ST.jstr(where) .. ',"err":' .. ST.jstr(tostring(e)) .. ',"gf":'
			.. game:GetFrameCount() .. ',"t":' .. Isaac.GetTime() .. '}')
	end
	if R then R.errors = R.errors + 1 end
end

local function guarded(where, fn)
	return function(...)
		local ok, r = pcall(fn, ...)
		if not ok then fail(where, r); return nil end
		return r
	end
end

-- 检查输出目录可写
do
	local fh = io.open(OUT .. "/irr_boot.json", "w")
	if fh then
		fh:write('{"v":"', VERSION, '","t":', tostring(Isaac.GetTime()), ',"proc":', ST.jstr(PROC), ',"src":', ST.jstr(SRC),
			',"obs":', ST.jstr(OBS), ',"mt":', ST.jval(MT.header_base()), '}')
		fh:close()
		S.writable = true
	else
		Isaac.DebugString("[IRR] output dir not writable: " .. OUT .. " -- recorder disabled")
	end
end

------------------------------------------------------------------ 写行
-- 这个 Lua 构建的数字转字符串偶发损坏：整数前多出 "0x"（MT_PROBE §5.3 的 `0x2869294168`、`0x13:15:0:0`；第三阶段 eC3 录制里
-- 网格行 `[55,2,1003,0x1,3,0]`），会让整行 JSON 解析失败。我们写的行里数字从不是十六进制、字符串里也不会出现 "0x"+数字，
-- 所以写出前把 "0x" 后紧跟（可带负号的）数字的地方去掉 "0x"，并计数（end 行 fmt_repairs / fmt_examples）。
local function repair(r, s)
	local fixed, n = s:gsub("0x(%-?%d)", "%1")
	if n > 0 then
		r.fmt_repairs = r.fmt_repairs + n
		if #r.fmt_examples < 5 then
			local i = s:find("0x", 1, true)
			r.fmt_examples[#r.fmt_examples + 1] = (s:sub(math.max(1, i - 40), i + 20):gsub("0x", "0 x")) -- 写成 "0 x"，否则写 end 行时又被修掉
		end
	end
	return fixed
end

local function line(s)
	if R then
		if s:find("0x", 1, true) then s = repair(R, s) end
		R.fh:write(s, "\n")
	end
end

local function write_status()
	if not S.writable then return end
	local fh = io.open(OUT .. "/status.json", "w")
	if not fh then return end
	fh:write('{"v":"', VERSION, '","t":', tostring(Isaac.GetTime()), ',"proc":', ST.jstr(PROC),
		',"mode":', ST.jstr(RP and "replay" or (H and "harness" or "record")),
		',"in_game":', tostring(R ~= nil), ',"run":', ST.jstr(R and R.name), ',"n":', tostring(R and R.n or 0),
		',"gf":', tostring(game:GetFrameCount()), ',"paused":', tostring(game:IsPaused()),
		',"runs":', tostring(S.runs), ',"ack":', tostring(H and H.ack or 0), ',"errors":', tostring(S.errors),
		',"err":', ST.jstr(S.err), ',"dead":', tostring(R ~= nil and Isaac.GetPlayer(0):IsDead()),
		',"seed":', ST.jstr(R and R.seed), ',"replay":', RP and RP.stats_json() or "null",
		-- 0.4.0：当前房间号与是否已清（测试驱动用；只读）
		',"room":', tostring(R and game:GetLevel():GetCurrentRoomIndex() or "null"),
		',"clear":', tostring(R ~= nil and game:GetRoom():IsClear()),
		'}')
	fh:close()
	S.last_status = Isaac.GetTime()
end

local function signal(name, kind, extra)
	local fh = io.open(OUT .. "/signal-" .. name .. "." .. kind .. ".json", "w")
	if fh then fh:write(extra); fh:close() end
end

-- 第三方 mod 的效果调用（api.lua）：替换 Game/Level/EntityPlayer/SFXManager 实例元表的 __index（Level 另有 __newindex）、
-- Isaac.ExecuteCommand；别的 mod 调用被监视的方法时写 api 行。本 mod 自己的命令（harness）经 API.exec_raw，不记录
local API_ON = getenv("IRR_API") ~= "0"
-- 控制台命令（phase10）：RR_MT_CONSOLE=1（录制会话 --allow-console）时 DLL 挂 Console::SubmitInput，这里每次渲染取走
pcall(MT.console_init)
if API_ON and S.writable then
	local ok, e = pcall(API.install, {run = function() return R end, line = line, jstr = ST.jstr,
		recording = function() return S.writable end, self_dir = SRC and SRC:match("([^/\\]+)$") or "isaac_run_recorder"})
	if not ok then fail("api_install", e) end
end

------------------------------------------------------------------ 开销统计
local HIST_N = 32
local function new_cost() local h = {}; for i = 1, HIST_N do h[i] = 0 end; return {hist = h, sum = 0, max = 0, frames = 0} end
local function add_cost(c, ms)
	local b = ms + 1; if b > HIST_N then b = HIST_N end
	c.hist[b] = c.hist[b] + 1; c.sum = c.sum + ms; c.frames = c.frames + 1
	if ms > c.max then c.max = ms end
end
local function cost_json(c)
	return '{"frames":' .. c.frames .. ',"sum_ms":' .. c.sum .. ',"max_ms":' .. c.max .. ',"hist_ms":[' .. concat(c.hist, ",") .. ']}'
end

------------------------------------------------------------------ 局的开关
local function close_run(reason)
	if not R then return end
	local r = R
	local fields = ST.missing_player_fields()
	line('{"k":"end","n":' .. r.n .. ',"gf":' .. game:GetFrameCount() .. ',"t":' .. Isaac.GetTime() .. ',"reason":' .. ST.jstr(reason)
		.. ',"frames":' .. r.n .. ',"renders":' .. r.renders .. ',"errors":' .. r.errors .. ',"cost":' .. cost_json(r.cost)
		.. ',"missing_player_fields":' .. ST.jval(fields) .. ',"mt_unstable_reads":' .. ST.jval(MT.unstable_reads())
		.. ',"fmt_repairs":' .. r.fmt_repairs .. ',"fmt_examples":' .. ST.jval(r.fmt_examples)
		.. ',"api":' .. ST.jval(API.counts)
		.. ',"console_cmds":' .. tostring(S.console_cmds or 0)
		.. ',"replay":' .. (r.replay and RP.stats_json() or "null") .. '}')
	r.fh:close()
	if r.replay and H and RP.pinlog_json then
		local fh = io.open(OUT .. "/irr_pinlog-" .. r.name .. ".json", "w")
		if fh then fh:write(RP.pinlog_json()); fh:close() end
	end
	R = nil
	IN.replay = nil
	signal(r.name, "end", '{"run":' .. ST.jstr(r.name) .. ',"reason":' .. ST.jstr(reason) .. ',"frames":' .. r.n .. ',"t":'
		.. Isaac.GetTime() .. ',"os_time":' .. os.time() .. '}')
	write_status()
end

local function open_run(continued, mt_header)
	close_run("restart")
	if not S.writable then return end
	S.runs = S.runs + 1
	local name = "run-" .. PROC .. "-" .. ("%04d"):format(S.runs)
	local fh = io.open(OUT .. "/" .. name .. ".jsonl", "wb")
	if not fh then fail("open_run", "cannot open " .. name); return end
	fh:setvbuf("full", 1 << 18)
	ST.reset_run()
	ST.refresh_players()
	IN.reset(game:GetFrameCount())
	IN.n_next = 1
	local hinfo = H and H.on_game_started(continued) or nil
	local is_replay = hinfo and hinfo.replay and RP ~= nil
	R = {fh = fh, name = name, n = 0, renders = 0, errors = 0, spawns = {}, cost = new_cost(), perf = new_cost(),
		perf_renders = 0, perf_t = Isaac.GetTime(), gf_last = game:GetFrameCount(), perf_ifc = Isaac.GetFrameCount(), hud = HUD,
		replay = is_replay, seed = game:GetSeeds():GetStartSeedString(), force_kf = true, mouse = {}, mouse_r = 0,
		mouse_last = nil, fmt_repairs = 0, fmt_examples = {}}
	if is_replay then
		-- 同一进程里的第二个回放局（观战的进程内倒带）：先清掉上一局的回放状态
		if RP.active and RP.reset then RP.reset() end
		IN.replay = RP; RP.active = true
	end
	local meta = quiet(ST.meta)
	pcall(API.verify)
	local session = read_file(OUT .. "/irr_session.json")
	local pre = S.prespawn
	S.prespawn = {}
	line(concat({'{"k":"header","schema":"', SCHEMA, '","recorder":"', VERSION, '","mode":', ST.jstr(is_replay and "replay" or "record"),
		',"run":', ST.jstr(name), ',"proc":', ST.jstr(PROC), ',"run_index":', S.runs,
		',"format":', ST.FORMAT, ',"continued":', tostring(continued == true), ',"obs":', ST.jstr(OBS), ',"mt":', ST.jval(mt_header or MT.header_base()),
		',"gf":', game:GetFrameCount(), ',"t":', Isaac.GetTime(),
		',"ifc":', Isaac.GetFrameCount(), ',"os_time":', os.time(), ',"os_date":', ST.jstr(os.date("!%Y-%m-%dT%H:%M:%SZ")),
		',"meta":', ST.jval(meta),
		',"api":', ST.jval(API.header()),
		',"console":', ST.jval({hook = MT.console_ok == true, reason = MT.console_reason}),
		',"harness":', hinfo and ST.jval({start = hinfo.start, bot = hinfo.bot, bot_seed = hinfo.bot_seed, frames = hinfo.frames,
			cmds = hinfo.cmds, replay = hinfo.replay}) or "null",
		',"replay_of":', is_replay and ST.jstr(RP.tbl.source) or "null",
		',"session":', (session and session:match("^%s*{") and session) or "null",
		',"fields":', ST.jval({player = ST.PLAYER_FIELD_NAMES, entity = ST.ENTITY_FIELD_NAMES, entity_extra = ST.ENTITY_EXTRA_NAMES,
			grid = ST.GRID_FIELD_NAMES, door = ST.DOOR_FIELD_NAMES, room_desc = ST.ROOMDESC_FIELD_NAMES,
			room_frame = ST.ROOM_FRAME_FIELD_NAMES, tick_input = {"value", "pressed", "triggered"},
			spawn = {"type", "variant", "subtype", "seed", "gf"}, extra_input = {"hook", "action", "who", "value"},
			mouse = {"render", "screen_x", "screen_y", "game_x", "game_y", "buttons"}, mt = {"cursor", "hash"},
			fixed_x100 = ST.FIXED_FIELDS, entity_extra_types = ST.ENTITY_EXTRA_TYPES,
			position_prediction = ST.POSITION_COLUMNS,
			-- v2（format 2）：外观行 pa/ea、字符串字典 sd、网格/门精灵 gs/ds（README 4.10）
			appearance = ST.APPEARANCE_FIELD_NAMES, player_appearance_extra = ST.PLAYER_APPEARANCE_EXTRA,
			appearance_fixed_x100 = ST.APPEARANCE_FIXED, appearance_string_columns = ST.APPEARANCE_STRING_COLUMNS,
			grid_sprite = ST.GRID_SPRITE_FIELD_NAMES, door_sprite = ST.DOOR_SPRITE_FIELD_NAMES}),
		',"prestart_spawns":[', concat(pre, ","), ']}'}))
	line(quiet(ST.level_line, 1))
	line(quiet(ST.room_line, 1))
	local il = quiet(ST.items_lines, 1, true)
	if il then for _, s in ipairs(il) do line(s) end end
	for _, s in ipairs(API.on_run_open()) do line(s) end
	S.console_cmds = 0
	signal(name, "start", '{"run":' .. ST.jstr(name) .. ',"continued":' .. tostring(continued == true) .. ',"seed":'
		.. ST.jstr(R.seed) .. ',"t":' .. Isaac.GetTime() .. ',"os_time":' .. os.time() .. '}')
	S.paused = game:IsPaused()
	write_status()
end

local function ev(name, args)
	if not R then return end
	line('{"k":"ev","n":' .. (R.n + 1) .. ',"gf":' .. game:GetFrameCount() .. ',"ev":"' .. name .. '","a":' .. args .. '}')
end

-- 玩家在调试控制台提交的命令（phase10；DLL 钩子排队，MT.console_take 取走）。写成 ev cmd：a = [命令, 参数]（与 MC_EXECUTE_CMD
-- 的 cmd 事件相同的拆法），src = "console"，line = 原文，r = 本帧窗口里暂停开始的那次渲染序号（控制台打开 = 暂停；
-- 命令在暂停期间执行，回放在这次渲染之后执行），paused = 取走时是否暂停中。局外（菜单里）提交的写进 errors.jsonl。
local function take_console()
	if not MT.console_ok then return end
	for _ = 1, 16 do
		local okc, text, k, left = pcall(MT.console_take)
		if not okc or text == nil then return end
		local cmd, params = text:match("^%s*(%S+)%s*(.-)%s*$")
		if R then
			local paused = game:IsPaused()
			local r = (paused and S.pause_rs) or IN.renders_since_update()
			line('{"k":"ev","n":' .. (R.n + 1) .. ',"gf":' .. game:GetFrameCount() .. ',"ev":"cmd","a":[' .. ST.jstr(cmd or "")
				.. "," .. ST.jstr(params or "") .. '],"src":"console","line":' .. ST.jstr(text) .. ',"r":' .. r
				.. ',"paused":' .. tostring(paused) .. ',"k_console":' .. tostring(k) .. "}")
			S.console_cmds = (S.console_cmds or 0) + 1
		else
			fail("console_outside_run", text)
		end
	end
end

------------------------------------------------------------------ 每逻辑帧
local function build_frame(n, gf)
	-- 关键帧：本局第 1 帧、进房后第一帧、每 300 帧（玩家/实体行写完整行；其余帧写差分行）
	local kf = n == 1 or n % 300 == 1 or R.force_kf
	R.force_kf = false
	local buf = {'{"k":"f","n":', tostring(n), ',"gf":', tostring(gf), ',"t":', tostring(Isaac.GetTime()),
		kf and ',"kf":1' or '', ',"tk":[', IN.take_ticks(), ']'}
	if #R.spawns > 0 then
		buf[#buf + 1] = ',"sp":[' .. concat(R.spawns, ",") .. "]"
		R.spawns = {}
	end
	if #R.mouse > 0 then
		buf[#buf + 1] = ',"ms":[' .. concat(R.mouse, ",") .. "]"
		R.mouse = {}
	end
	-- E4：全局 MT 的 (cursor, hash)，只读不推进
	if MT.available then
		local c, h, reads = MT.state()
		if c then
			-- 第三个数只在 DLL 读了不止一次才稳定时出现（两份拷贝不一致，见 rr_mt.cpp stableSnapshot）
			if reads and reads > 1 then buf[#buf + 1] = ',"mt":[' .. c .. "," .. h .. "," .. reads .. "]"
			else buf[#buf + 1] = ',"mt":[' .. c .. "," .. h .. "]" end
		end
	end
	ST.refresh_players()
	ST.frame_parts(n, buf, kf, OBS == "keyframe" and not kf)
	buf[#buf + 1] = "}"
	local il = ST.items_lines(n, false, kf)
	if il then for _, s in ipairs(il) do line(s) end end
	line(concat(buf))
	R.mouse_r = 0
end

-- 鼠标流（D6：实测 Input.IsMouseBtnPressed / GetMousePosition 能读到真实鼠标）：每个渲染帧读一次，
-- 与上次不同才记 [本逻辑帧窗口内第几次渲染, 屏幕x, 屏幕y, 游戏x, 游戏y, 按键位(左1 右2 中4)]（坐标 ×100）
local function sample_mouse()
	R.mouse_r = R.mouse_r + 1
	local s, g = Input.GetMousePosition(false), Input.GetMousePosition(true)
	local b = (Input.IsMouseBtnPressed(0) and 1 or 0) + (Input.IsMouseBtnPressed(1) and 2 or 0) + (Input.IsMouseBtnPressed(2) and 4 or 0)
	local key = ST.fx(s.X) .. "," .. ST.fx(s.Y) .. "," .. ST.fx(g.X) .. "," .. ST.fx(g.Y) .. "," .. b
	if key ~= R.mouse_last then
		R.mouse_last = key
		R.mouse[#R.mouse + 1] = "[" .. R.mouse_r .. "," .. key .. "]"
	end
end

local function frame()
	local t0 = os.clock()
	local n = R.n + 1
	R.n = n
	local gf = game:GetFrameCount()
	R.gf_last = gf
	local ok, err = pcall(quiet, build_frame, n, gf)
	-- tick 分类状态无论如何都要推进，否则后续 tick 全部错类
	IN.on_update(gf)
	IN.n_next = n + 1
	API.mark("U", 0)
	R.room_last = game:GetLevel():GetCurrentRoomIndex()
	if not ok then fail("frame", err) end
	if n % 300 == 0 then R.fh:flush() end
	local ms = math.floor((os.clock() - t0) * 1000 + 0.5)
	add_cost(R.cost, ms)
	add_cost(R.perf, ms)
	if R.perf.frames >= 150 then
		local now, ifc = Isaac.GetTime(), Isaac.GetFrameCount()
		line('{"k":"perf","n":' .. n .. ',"t":' .. now .. ',"ifc":' .. ifc .. ',"dt_ms":' .. (now - R.perf_t) .. ',"renders":'
			.. R.perf_renders .. ',"cost":' .. cost_json(R.perf) .. '}')
		R.perf = new_cost(); R.perf_t = now; R.perf_renders = 0
	end
end

------------------------------------------------------------------ 回调
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, action)
	if not R or IN.reading then return nil end
	if API.mk ~= "Q" then API.mark("Q", IN.renders_since_update()) end
	local ok, ret = pcall(IN.on_query, entity, hook, action)
	if ok then return ret end
	IN.reading = false
	fail("input", ret)
	return nil
end)

mod:AddCallback(ModCallbacks.MC_PRE_ENTITY_SPAWN, function(_, t, v, s, pos, vel, spawner, seed)
	local ok, r = pcall(function()
		local gf = game:GetFrameCount()
		local ret = nil
		if R and R.replay then ret = RP.pin(gf, t, v, s, seed) end
		local eff = ret and ret[4] or seed
		local entry = "[" .. t .. "," .. v .. "," .. s .. "," .. eff .. "," .. gf .. "]"
		if R then R.spawns[#R.spawns + 1] = entry
		elseif #S.prespawn < 4096 then S.prespawn[#S.prespawn + 1] = entry end
		return ret
	end)
	if ok then return r end
	fail("spawn", r)
	return nil
end)

mod:AddCallback(ModCallbacks.MC_POST_UPDATE, guarded("post_update", function()
	-- 回放：录制里第 n 帧逻辑更新期间（本 mod 的 MC_POST_UPDATE 之前）别的 mod 的调用、以及没赶上渲染的调用（api 行）
	if R and R.replay and RP.api_update_start then RP.api_update_start(R.n + 1) end
	if R and R.replay and RP.console_update_start then RP.console_update_start(R.n + 1) end
	if R then frame() end
	-- 回放：录制里排在本 mod 的 MC_POST_UPDATE 之后（同一次迭代）的调用
	if R and R.replay and RP.api_update_end then RP.api_update_end(R.n) end
	-- 回放：录制里"下一帧第 1 次渲染"读到的鼠标在这里还原（phase8-mouse；replay.lua 的鼠标一节）
	if R and R.replay and RP.mouse_on_update then
		local okm, em = pcall(RP.mouse_on_update, R.n)
		if not okm then fail("mouse_replay", em) end
	end
	-- 观战/导出：在 harness 的帧上限关局之前告诉 watch 本帧号（导出要在下一次更新时写出"本帧最后一次渲染"）
	if W then W.on_update(R and R.replay and R.n or nil) end
	if H then
		if R then H.on_run_update(R.n) end
		H.on_any_update()
	end
end))

mod:AddCallback(ModCallbacks.MC_POST_RENDER, guarded("post_render", function()
	-- 观战挂起/慢放的空迭代（DLL 整次跳过了 Manager::Update）：这次渲染不属于逻辑 tick 结构，录制侧的渲染计数、
	-- tick 分类、暂停检测、鼠标采样全部跳过，只画观战 GUI、读热键
	if W and W.skipped_render() then W.on_render(); return end
	S.renders = S.renders + 1
	if R then
		-- 回放：录制里本次渲染（排在本 mod 的渲染回调之前）别的 mod 的调用（api 行）。在本 mod 自己的渲染计数之前执行：
		-- 与录制时一样，调用期间引擎若查询输入（例如 ChangeRoom 里），tick 序号仍是"上一次渲染"的
		if R.replay and RP.api_render then RP.api_render(R.n + 1, IN.renders_since_update() + 1, false) end
		IN.on_render()
		API.mark("R", IN.renders_since_update())
		R.renders = R.renders + 1
		R.perf_renders = R.perf_renders + 1
		local paused = game:IsPaused()
		if paused ~= S.paused then
			S.paused = paused
			ev(paused and "pause" or "unpause", "[" .. Isaac.GetFrameCount() .. "," .. Isaac.GetTime() .. "," .. IN.renders_since_update() .. "]")
			if paused then S.pause_rs = IN.renders_since_update() end
			if paused then R.fh:flush() end
			if R.replay then RP.on_pause_flip(IN.n_next, paused, IN.renders_since_update()) end
		end
		-- 回放的菜单暂停：Lua 关不掉菜单，向 Python（replay_run）请求发一次真实 Esc（pausectl.txt = 请求标识，空 = 无请求）
		if R.replay then
			local req = RP.close_request(IN.n_next, IN.renders_since_update()) or ""
			if req ~= S.pausectl then
				S.pausectl = req
				local fh = io.open(OUT .. "/pausectl.txt", "w")
				if fh then fh:write(req); fh:close() end
			end
		end
		take_console()
		-- 回放：录制里玩家在控制台提交的命令（ev cmd src=console），在同一帧窗口、暂停开始的那次渲染之后执行
		if R.replay and RP.console_render then RP.console_render(R.n + 1, IN.renders_since_update()) end
		local okm, em = pcall(quiet, sample_mouse)
		if not okm and not R.mouse_err then R.mouse_err = true; fail("mouse", em) end
		-- 回放：采样之后设置录制里"下一次渲染"读到的鼠标（phase8-mouse）
		if R.replay and RP.mouse_on_render then
			local okr, er = pcall(RP.mouse_on_render, R.n + 1, R.mouse_r)
			if not okr then fail("mouse_replay", er) end
		end
		-- 常驻 "REC"（左上角、心之上方的空白处、红色），让玩家随时知道正在录；前 2 秒更亮（IRR_HUD=0 关闭）
		if R.hud then
			local a = (R.renders < 120) and 1.0 or 0.75
			pcall(Isaac.RenderScaledText, "REC", 4, 2, 1.0, 1.0, 1.0, 0.15, 0.15, a)
		end
		-- 回放：录制里排在本 mod 渲染回调之后的调用（af = 1）
		if R and R.replay and RP.api_render then RP.api_render(R.n + 1, IN.renders_since_update(), true) end
	end
	if H then H.on_render(S.renders) end
	if W then W.on_render() end
	if Isaac.GetTime() - S.last_status > (H and 250 or 2000) then write_status() end
end))

-- E3：开局 MT reseed。新局：第一个 MC_POST_PLAYER_INIT（GetFrameCount()==0，开局房实体初始化之前）与
-- MC_POST_GAME_STARTED 各一次，值 = StartSeed xor salt（同一个 salt，每局随机）；续局（recorder 0.3.1 起）：读档后第一个
-- MC_POST_PLAYER_INIT（还没有打开的局、gf>0，进开局房与选出生位置之前）与 MC_POST_GAME_STARTED 各一次（同一个 salt；
-- 0.3.0 及以前续局只在 MC_POST_GAME_STARTED reseed 一次，出生位置不受控，回放靠 start_players 钉回）。回放局（harness 的下一个局是 replay）改用录制 header.mt.reseeds 里同一时机的 salt，录制里没有的时机不 reseed。
local function replay_salts()
	if H and RP and H.next_is_replay() then return RP.tbl.mt or {} end
	return nil
end

-- 诊断（只在 harness 下）：开局/续局各回调的时机、玩家位置、MT 状态，写 <IRR_DIR>/irr_diag.jsonl（phase7-continue 出生门实验用）
local function diag(what, arg)
	local parts = {}
	for i = 0, game:GetNumPlayers() - 1 do
		local p = Isaac.GetPlayer(i)
		if p then parts[#parts + 1] = "[" .. ST.fx(p.Position.X) .. "," .. ST.fx(p.Position.Y) .. "]" end
	end
	local c, h = nil, nil
	if MT.available then c, h = MT.state() end
	append_file("irr_diag.jsonl", '{"what":' .. ST.jstr(what) .. ',"arg":' .. ST.jval(arg) .. ',"gf":' .. game:GetFrameCount()
		.. ',"ifc":' .. Isaac.GetFrameCount() .. ',"in_run":' .. tostring(R ~= nil) .. ',"room":' .. game:GetLevel():GetCurrentRoomIndex()
		.. ',"start_seed":' .. tostring(game:GetSeeds():GetStartSeed())
		.. ',"players":[' .. concat(parts, ",") .. '],"mt":[' .. tostring(c) .. ',' .. tostring(h) .. ']}')
end

-- 续局的早期 reseed（phase7-continue）：续局时第一个 MC_POST_PLAYER_INIT 发生在读档之后、进开局房与引擎给玩家选出生位置之前
-- （此时还没有打开的局：R == nil，gf = 存档里的帧数 > 0）。IRR_MT_EARLY_CONTINUE=0 关闭（录制侧回到 0.3.0 的规则）；
-- 回放局只看录制 header 里有没有这个时机。
local EARLY_CONTINUE = getenv("IRR_MT_EARLY_CONTINUE") ~= "0"

mod:AddCallback(ModCallbacks.MC_POST_PLAYER_INIT, guarded("player_init", function()
	if H then diag("player_init", nil) end
	if MT.early_done or not MT.available then return end
	local gf = game:GetFrameCount()
	local continuing = gf ~= 0 and R == nil
	if gf ~= 0 and not continuing then return end
	local rs = replay_salts()
	if continuing and not rs and not EARLY_CONTINUE then return end
	MT.early_done = true
	local salt
	if rs then salt = rs.POST_PLAYER_INIT else salt = MT.new_salt() end
	if not salt then return end
	MT.pending = {rec = MT.reseed("POST_PLAYER_INIT", salt, rs ~= nil)}
	MT.pending.rec.continuing = continuing
end))

local function mt_on_game_started(continued)
	local hdr = MT.header_base()
	hdr.continued = continued == true
	hdr.rule = "seeded_with = StartSeed xor salt; new run: first POST_PLAYER_INIT (gf=0) + POST_GAME_STARTED (same salt); "
		.. "continued: first POST_PLAYER_INIT while no run is open (gf>0, after the save is loaded, before the start room and "
		.. "the spawn position) + POST_GAME_STARTED (same salt)" .. (EARLY_CONTINUE and "" or " [early continue reseed disabled: POST_GAME_STARTED only]")
	hdr.early_continue = EARLY_CONTINUE
	-- CPU 亲和（phase8-mouse §H4）：进程启动时 DLL 读到的 RR_MT_AFFINITY 掩码，写成十进制数（环境变量是 "0x1" 这类
	-- 十六进制串；行里不能出现 "0x"+数字，见第 106 行的数字格式修复）；没设则 nil。录制时没限核的局，回放不保证精确。
	hdr.affinity = tonumber(getenv("RR_MT_AFFINITY"))
	local reseeds = {}
	local pend = MT.pending
	MT.pending = nil
	MT.early_done = false
	if pend then
		pend.rec.stale = pend.rec.start_seed ~= game:GetSeeds():GetStartSeed()
		reseeds[#reseeds + 1] = pend.rec
	end
	if MT.available then
		local rs = replay_salts()
		local salt
		if rs then salt = rs.POST_GAME_STARTED
		else salt = (pend and not pend.rec.stale and pend.rec.salt) or MT.new_salt() end
		if salt then reseeds[#reseeds + 1] = MT.reseed("POST_GAME_STARTED", salt, rs ~= nil) end
		hdr.salt = salt or (pend and pend.rec.salt)
		hdr.start_seed = game:GetSeeds():GetStartSeed()
		hdr.seeded_with = hdr.salt and ((hdr.start_seed ~ hdr.salt) & 0xFFFFFFFF) or nil
	end
	hdr.reseeds = reseeds
	return hdr
end

mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, guarded("game_started", function(_, continued)
	if H then diag("game_started", continued == true) end
	-- reseed 必须在本回调里任何其它工作之前
	local okm, mth = pcall(mt_on_game_started, continued)
	if not okm then fail("mt_reseed", mth); mth = nil end
	open_run(continued, mth)
	-- 回放：录制里开局前（局外）别的 mod 的调用（ph = gs）
	if R and R.replay and RP.api_game_started then RP.api_game_started() end
	-- 回放局：启用鼠标还原（rr_mt.dll 的 RRMT_SetMouse，需 RR_MT_MOUSE=1），开局光标位置无声设置（phase8-mouse）
	if R and R.replay and RP.mouse_start then
		local okm, em = pcall(RP.mouse_start, MT)
		if not okm then fail("mouse_start", em) end
	end
	-- 续局回放：出生位置由 reseed 之前的随机数决定，钉回录制第 1 帧的位置（phase7-continue；非续局不做）
	if continued and R and R.replay and RP.apply_start_pos then
		local k = RP.apply_start_pos()
		if H and k > 0 then diag("start_pos_pinned", k) end
	end
end))

mod:AddCallback(ModCallbacks.MC_POST_GAME_END, guarded("game_end", function(_, over)
	ev("game_end", "[" .. tostring(over == true) .. "]")
	if R then R.fh:flush() end
	if over then close_run("death") end
end))

mod:AddCallback(ModCallbacks.MC_PRE_GAME_EXIT, guarded("game_exit", function(_, save)
	MT.early_done = false
	ev("exit", "[" .. tostring(save == true) .. "]")
	close_run(save and "exit_save" or "exit")
end))

mod:AddCallback(ModCallbacks.MC_POST_NEW_LEVEL, guarded("new_level", function()
	if R and R.n > 0 and game:GetFrameCount() >= R.gf_last then line(quiet(ST.level_line, R.n + 1)) end
end))

mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, guarded("new_room", function()
	if H and not R then diag("new_room_no_run", nil) end
	-- gf 变小 = 新局已在生成（例如按住 R 重开且没有先触发 PRE_GAME_EXIT）：不写进旧局
	if R and R.n > 0 and game:GetFrameCount() >= R.gf_last then
		-- 回放：换房期间别的 mod 的调用（父调用执行中的嵌套子调用 / 本帧更新里换房之后的调用）
		if R.replay and RP.api_new_room then RP.api_new_room(R.n + 1) end
		line(quiet(ST.room_line, R.n + 1))
		R.force_kf = true
		R.fh:flush()
	end
end))

mod:AddCallback(ModCallbacks.MC_EXECUTE_CMD, guarded("execute_cmd", function(_, cmd, params)
	ev("cmd", "[" .. ST.jstr(cmd) .. "," .. ST.jstr(params) .. "]")
end))

mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, guarded("take_dmg", function(_, e, amount, flags, source, countdown)
	if not R then return nil end
	local p = e:ToPlayer()
	if p then
		local st, sv = -1, -1
		if source then st, sv = source.Type or -1, source.Variant or -1 end
		ev("damage", "[" .. ST.player_num(p) .. "," .. ST.fx(amount) .. "," .. tostring(flags) .. "," .. st .. "," .. sv .. ","
			.. tostring(countdown) .. "]")
	end
	return nil
end))

mod:AddCallback(ModCallbacks.MC_PRE_USE_ITEM, guarded("use_item", function(_, id)
	ev("use_item", "[" .. tostring(id) .. "]")
	return nil
end))

mod:AddCallback(ModCallbacks.MC_USE_CARD, guarded("use_card", function(_, card)
	ev("use_card", "[" .. tostring(card) .. "]")
end))

mod:AddCallback(ModCallbacks.MC_USE_PILL, guarded("use_pill", function(_, effect)
	ev("use_pill", "[" .. tostring(effect) .. "]")
end))

------------------------------------------------------------------ harness / replay（只在测试时）
local ctx = {
	out = OUT, jstr = ST.jstr, jval = ST.jval, fx = ST.fx, fail = fail, status = write_status, load_module = load_module,
	line = line, exec = API.exec_raw, api = API,
	run = function() return R end,
	close_run = close_run,
	bench = function(n)
		local r = {n = n, entities = #Isaac.GetRoomEntities(), in_run = R ~= nil}
		if R then
			local t = os.clock()
			IN.reading = true
			-- dry = true：v2 的字典/外观缓存在基准结束时还原（基准里新分配的字典条目不会写进文件）
			for _ = 1, n do local b = {}; ST.frame_parts(R.n, b, true, false, true); concat(b) end
			r.frame_ms_keyframe = (os.clock() - t) * 1000 / n
			t = os.clock()
			for _ = 1, n do local b = {}; ST.frame_parts(R.n, b, false, false, true); concat(b) end
			r.frame_ms_delta = (os.clock() - t) * 1000 / n
			local b = {}; ST.frame_parts(R.n, b, true, false, true); r.frame_bytes_keyframe = #concat(b)
			r.format = ST.FORMAT
			-- 同一状态下关掉 v2 外观再测一次（= v1 的工作量），差值即 v2 新增的 Lua 开销
			ST.appearance_enabled = false
			t = os.clock()
			for _ = 1, n do local b2 = {}; ST.frame_parts(R.n, b2, false, false, true); concat(b2) end
			r.frame_ms_delta_noapp = (os.clock() - t) * 1000 / n
			t = os.clock()
			for _ = 1, n do local b2 = {}; ST.frame_parts(R.n, b2, true, false, true); concat(b2) end
			r.frame_ms_keyframe_noapp = (os.clock() - t) * 1000 / n
			local b3 = {}; ST.frame_parts(R.n, b3, true, false, true); r.frame_bytes_keyframe_noapp = #concat(b3)
			ST.appearance_enabled = true
			IN.reading = false
			r.input_read_ms = IN.bench(Isaac.GetPlayer(0), n * 30)
		end
		return r
	end,
}
if S.writable then
	local ctl = read_file(OUT .. "/irr_control.json")
	if ctl then
		local ok, h = pcall(function() return load_module("harness")(ctx, ctl) end)
		if ok then H = h else fail("harness_load", h) end
		local rpath = OUT .. "/irr_replay.lua"
		if read_file(rpath) then
			local ok2, rp = pcall(function() return load_module("replay").load(rpath) end)
			if ok2 and rp then RP = rp; ctx.replay_obj = rp; if rp.set_api then rp.set_api(API, fail, line) end else fail("replay_load", rp) end
		end
		-- 观战模式 / 实录导出（replay_run.py --watch / --capture 放置 irr_watch.json）：只在回放时加载
		local wcfg = RP and read_file(OUT .. "/irr_watch.json")
		if wcfg then
			ctx.dll = MT.dll
			local okw, w = pcall(function() return load_module("watch")(ctx, wcfg) end)
			if okw and w then W = w; IN.watch = true else fail("watch_load", w) end
		end
	end
	write_status()
end
