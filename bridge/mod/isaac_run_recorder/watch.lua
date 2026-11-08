-- 观战模式与实录导出（phase6-watch；只在回放且 <IRR_DIR>/irr_watch.json 存在时由 main.lua 加载）。
-- 由 replay_run.py --watch / --capture 放置 irr_watch.json：
--   {"mode":"watch"|"capture", "frames":N, "speed":1, "paused":false, "start_target":T, "start_kind":"rewind",
--    "timeline":"<irr_dir>/irr_watch_timeline.lua", "shots_dir":"...", "shot_at":{"n":"path",...}, "gui":true}
-- 速率靠 rr_mt.dll 的观战钩子（RRW_Set）：暂停 = 速率 0 = DLL 整次跳过之后的 Manager::Update（渲染照常），不用游戏自己的
-- 暂停菜单，所以逻辑看到的迭代序列与不暂停时相同。热键在 MC_POST_RENDER 里读：首选 DLL 子类化窗口过程收集的按键（RRW_Keys，
-- 挂起时也有效），没有 DLL 时退回 Input.IsButtonPressed(Keyboard.*, 0)。
-- 观众的按键不进游戏逻辑：玩家 12 个动作由回放覆盖，其余动作查询由 inputs.lua（M.watch）返回"没按"。
-- 往回跳：写 watchreq.txt 请求 Python 重启回放进程（新进程按录制从头精确重跑，gate 全速到目标帧，屏幕显示"正在倒带"）。
-- 控制文件 watchctl.txt（Python 小命令行 / 自动化写，每行 "<id> <命令> [参数]"）：goto N | speed X | pause | resume |
--   toggle | step | room next|prev | home | cam free|follow | campos X Y | hint on|off | shot PATH | shotat N PATH | quit
-- 状态：watchstate.json（约每 0.25 s）与 watchlog.jsonl（到达目标帧、命令、倒带请求等事件）。
local json = require("json")
local game = Game()

local SPEEDS = {0.25, 0.5, 1, 2, 4, 8, 16}   -- 16 = 不限速（turbo 上限）
local ROOM_TYPES = {"普通", "商店", "错误房", "宝箱房", "Boss", "小Boss", "隐藏房", "超级隐藏房", "街机", "诅咒房",
	"挑战房", "图书馆", "献祭房", "恶魔房", "天使房", "地下通道", "Boss挑战", "卧室", "卧室", "宝库", "骰子房", "黑市",
	"大门", "星象房", "传送房", "传送出口", "", "", "极隐藏房"}
local STAGE_ZH = {Basement = "地下室", Cellar = "地窖", ["Burning Basement"] = "燃烧地下室", Caves = "洞穴",
	Catacombs = "墓穴", ["Flooded Caves"] = "淹水洞穴", Depths = "深牢", Necropolis = "坟场", ["Dank Depths"] = "阴湿深牢",
	Womb = "子宫", Utero = "子宫", ["Scarred Womb"] = "伤痕子宫", Downpour = "下水道", Dross = "污浊之地", Mines = "矿洞",
	Ashpit = "灰坑", Mausoleum = "陵墓", Gehenna = "炼狱", Corpse = "尸骸", ["Blue Womb"] = "???", Sheol = "阴间",
	Cathedral = "大教堂", ["Dark Room"] = "暗室", Chest = "宝箱", Void = "虚空", Home = "家"}
local STAGE_SHORT = {Basement = "B", Cellar = "B", ["Burning Basement"] = "B", Caves = "C", Catacombs = "C",
	["Flooded Caves"] = "C", Depths = "D", Necropolis = "D", ["Dank Depths"] = "D", Womb = "W", Utero = "W",
	["Scarred Womb"] = "W", ["Blue Womb"] = "BW", Sheol = "S", Cathedral = "Ca", ["Dark Room"] = "DR", Chest = "Ch",
	Void = "V", Downpour = "B", Dross = "B", Mines = "C", Ashpit = "C", Mausoleum = "D", Gehenna = "D", Corpse = "W", Home = "H"}
local HINT_LINES = {"空格 暂停/继续   . 单步", "[ ] 或 - + 变速", "PgUp/PgDn 上/下一房间", "Home 回到开局   G 跳转",
	"方向键 平移镜头   C 跟随", "F12 截图   H 收起"}
local STAGE_COLOR = {  -- 进度条每层的底色（取各层地面的主色调）
	{0.55, 0.40, 0.28}, {0.55, 0.40, 0.28}, {0.45, 0.45, 0.42}, {0.45, 0.45, 0.42}, {0.36, 0.30, 0.34}, {0.36, 0.30, 0.34},
	{0.62, 0.25, 0.25}, {0.62, 0.25, 0.25}, {0.35, 0.45, 0.60}, {0.70, 0.62, 0.40}, {0.25, 0.20, 0.20}, {0.40, 0.30, 0.10},
	{0.15, 0.15, 0.25}}

return function(ctx, cfg_text)
	local cfg = {}
	pcall(function() cfg = json.decode(cfg_text) or {} end)
	local W = {mode = cfg.mode or "watch"}
	local OUT = ctx.out

	------------------------------------------------------------------ DLL
	local dll = ctx.dll
	local function lf(name)
		if not dll or not package or not package.loadlib then return nil end
		local ok, f = pcall(package.loadlib, dll, name)
		if ok and type(f) == "function" then return f end
		return nil
	end
	local RSet, RInfo, RMark, RCapClose, RShot, RShotAt = lf("RRW_Set"), lf("RRW_Info"), lf("RRW_Mark"), lf("RRW_CapClose"),
		lf("RRW_Shot"), lf("RRW_ShotAt")
	local RCapFinal = lf("RRW_CapFinal")
	local avail = false
	if RInfo then
		local ok, a = pcall(RInfo)
		avail = ok and a == 1
	end
	W.dll_ok = avail
	local function set_rate(r, pmin)
		if avail and RSet then RSet(r, pmin or 0) end
		W.rate_set = r
	end

	------------------------------------------------------------------ 时间轴（replay_run 生成）
	local TL = {frames = cfg.frames or 0, levels = {}, rooms = {}, events = {}}
	if cfg.timeline then
		local ok, t = pcall(dofile, cfg.timeline)
		if ok and type(t) == "table" then TL = t else ctx.fail("watch_timeline", t) end
	end
	local N = cfg.frames or TL.frames or 0
	if N <= 0 then N = TL.frames or 1 end

	local function room_at(n)
		local best = nil
		for _, r in ipairs(TL.rooms) do
			if r.n <= n then best = r else break end
		end
		return best
	end
	local function level_at(n)
		local best = nil
		for _, l in ipairs(TL.levels) do
			if l.n <= n then best = l else break end
		end
		return best
	end

	------------------------------------------------------------------ 状态
	local st = {n = 0, si = 3, paused = cfg.paused == true, target = nil, kind = nil, t0 = 0, n0 = 0, step_to = nil,
		ended = false, rewinding = false, cam_free = false, cam_off = nil, cam_reset = false, cam_room = nil, hint = cfg.hint ~= false,
		input = nil, ctl_id = 0, renders = 0, fps_t = 0, fps_n = 0, fps = 0, last_state = -1e9, log_n = 0,
		cap_closed = false, cap_last = nil, shots = 0, quit = false, ff_eta = nil}
	for i, s in ipairs(SPEEDS) do if s == cfg.speed then st.si = i end end
	W.st = st

	local function now_ms() return Isaac.GetTime() end

	local function log_ev(name, extra)
		local fh = io.open(OUT .. "/watchlog.jsonl", "a")
		if not fh then return end
		local parts = {'{"ev":', ctx.jstr(name), ',"n":', tostring(st.n), ',"t":', tostring(now_ms()), ',"os":', tostring(os.time())}
		for k, v in pairs(extra or {}) do parts[#parts + 1] = ',' .. ctx.jstr(k) .. ':' .. ctx.jval(v) end
		parts[#parts + 1] = '}'
		fh:write(table.concat(parts), "\n")
		fh:close()
	end

	local function apply()
		if W.mode == "capture" then set_rate(-1, 250); return end
		if st.rewinding and st.inproc then set_rate(-1, 100); return end
		if st.rewinding or st.quit then set_rate(0, 0); return end
		-- 回放局开出来之前（菜单、harness 的第一个局）：不限速、正常呈现，暂停/倍速只对回放局生效
		if not st.live then set_rate(-1, 1000 / 60); return end
		if st.target then set_rate(-1, 100); return end
		if st.step_to then set_rate(1, 0); return end
		if st.paused or st.ended then set_rate(0, 0); return end
		local s = SPEEDS[st.si]
		if s >= 16 then set_rate(-1, 1000 / 60)
		elseif s > 1 then set_rate(s, 1000 / 60)
		else set_rate(s, 0) end
	end

	-- 往回跳：Python 重启回放进程，从头精确重跑到 T（新进程开局后由 start_target 接手）
	-- 两种倒带（cfg.rewind）：
	--   restart（默认）：停住，关局，Python 结束本进程、重拷开局存档、重开回放进程（新进程按 start_target 全速到 T）。
	--   inproc：关局后不停（游戏要跑起来才能重开），Python 在同一进程里再发一次 harness 的 start（seed 命令）开新的回放局，
	--           本模块在新局第 1 帧接手、全速到 T。省掉进程启动，但存档里的解锁/统计在本进程内存里已被上一局改过，
	--           道具池等可能不同（见 README §8.3）；用 compare 验证。
	local function request_rewind(T, why)
		st.rewinding = true
		st.rewind_to = T
		st.inproc = cfg.rewind == "inproc"
		apply()
		local fh = io.open(OUT .. "/watchreq.txt", "w")
		if fh then
			fh:write(("rewind %d %d %d %d %s %s\n"):format(T, st.si, st.paused and 1 or 0, st.hint and 1 or 0,
				(why and why ~= "") and why or "-", st.inproc and "inproc" or "restart"))
			fh:close()
		end
		log_ev("rewind_request", {target = T, from = st.n, why = why, inproc = st.inproc})
		-- 先停住（restart：上面 rate 0），再关局把回放文件写完（end 行 reason=rewind）
		ctx.close_run("rewind")
	end

	local function go_to(T, why)
		T = math.floor(T)
		if T < 1 then T = 1 end
		if T > N then T = N end
		st.input = nil
		if st.rewinding then return end
		if T == st.n then return end
		if T > st.n then
			st.target, st.kind, st.t0, st.n0 = T, "ff", now_ms(), st.n
			st.fps_t, st.fps_n, st.fps = now_ms(), st.n, 0
			log_ev("ff_start", {target = T, why = why})
			apply()
		else
			request_rewind(T, why)
		end
	end

	local function room_jump(dir)
		local rooms = TL.rooms
		if #rooms == 0 then return end
		local cur = 0
		for i, r in ipairs(rooms) do if r.n <= st.n then cur = i end end
		if dir > 0 then
			if rooms[cur + 1] then go_to(rooms[cur + 1].n, "room_next") end
		else
			local r = rooms[cur]
			if r and st.n - r.n > 45 then go_to(r.n, "room_prev")
			elseif rooms[cur - 1] then go_to(rooms[cur - 1].n, "room_prev")
			elseif r then go_to(r.n, "room_prev") end
		end
	end

	local function set_speed(i)
		if i < 1 then i = 1 end
		if i > #SPEEDS then i = #SPEEDS end
		st.si = i
		log_ev("speed", {speed = SPEEDS[i]})
		apply()
	end

	local function toggle_pause()
		if st.target or st.rewinding then return end
		if st.ended then return end
		st.paused = not st.paused
		st.step_to = nil
		log_ev(st.paused and "pause" or "resume", {})
		apply()
	end

	local function step()
		if st.target or st.rewinding or st.ended then return end
		st.paused = true
		st.step_to = st.n + 1
		log_ev("step", {to = st.step_to})
		apply()
	end

	local function shot(path)
		if not RShot then return end
		st.shots = st.shots + 1
		if not path or path == "" then
			path = (cfg.shots_dir or OUT) .. ("/watch-%06d-%02d.bmp"):format(st.n, st.shots)
		end
		RShot(path)
		log_ev("shot", {path = path})
	end

	if cfg.shot_at and RShotAt then
		for k, p in pairs(cfg.shot_at) do RShotAt(tonumber(k), p) end
	end

	------------------------------------------------------------------ 逐帧
	function W.skipped_render()
		if not avail then return false end
		local _, skip, _, _, _, _, _, _, _, _, _, _, _, draw = RInfo()
		st.draw_this = draw ~= 0
		return skip == 1
	end

	local function finish_capture()
		if st.cap_closed then return end
		st.cap_closed = true
		local frames = RCapClose and RCapClose() or -1
		log_ev("capture_closed", {frames = frames, last = st.cap_last})
		local fh = io.open(OUT .. "/capture_done.txt", "w")
		if fh then fh:write(tostring(frames)); fh:close() end
	end

	-- main.lua 每个 MC_POST_UPDATE 调用；n = 回放局本帧号（不在回放局时 nil）
	function W.on_update(n)
		if W.mode == "capture" then
			if n then
				-- 只导出录制的 1..N 帧（死亡局回放会多跑 90 帧，那些不要）
				st.n = n
				if n <= N + 1 and RMark then RMark(n) end
				if n == N and RCapFinal then RCapFinal(N) end
				if n >= N + 1 then finish_capture(); return end
				st.cap_last = n
			elseif st.cap_last and not st.cap_closed then
				-- 回放局已在第 cap_last 帧被帧上限关掉：这次更新之前的最后一次渲染就是 cap_last 帧的最后一次渲染
				if RMark then RMark(st.cap_last + 1) end
				finish_capture()
			end
			return
		end
		if not n then return end
		st.n = n
		if not st.live then st.live = true; apply() end
		if st.rewinding and st.inproc and n == 1 then
			-- 进程内倒带：新的回放局开出来了
			st.rewinding = false
			st.ended, st.step_to = false, nil
			st.fps_t, st.fps_n, st.fps = now_ms(), 1, 0
			if st.rewind_to > 1 then
				st.target, st.kind, st.t0, st.n0 = st.rewind_to, "rewind", now_ms(), 1
				log_ev("rewind_start", {target = st.target, inproc = true})
			end
			apply()
		end
		if RMark then RMark(n) end
		local t = now_ms()
		if t - st.fps_t >= 500 then
			st.fps = (n - st.fps_n) * 1000 / math.max(1, t - st.fps_t)
			st.fps_t, st.fps_n = t, n
		end
		if n == 1 and cfg.start_target and cfg.start_target > 1 and not st.started then
			st.started = true
			st.target, st.kind, st.t0, st.n0 = math.min(cfg.start_target, N), cfg.start_kind or "rewind", t, 1
			st.fps_t, st.fps_n, st.fps = t, 1, 0
			log_ev("rewind_start", {target = st.target})
			apply()
		end
		if st.step_to and n >= st.step_to then
			st.step_to = nil
			st.paused = true
			apply()
		end
		if st.target and n >= st.target then
			log_ev("arrive", {target = st.target, kind = st.kind, from = st.n0, ms = t - st.t0})
			st.target, st.kind = nil, nil
			apply()
		end
		if n >= N and not st.ended then
			st.ended = true
			log_ev("end", {})
			apply()
		end
		if W.cam_apply then W.cam_apply() end
	end

	------------------------------------------------------------------ 热键
	-- 首选 DLL 的窗口消息收集（RRW_Keys / RRW_Held，Windows VK 码）：挂起时原 Manager::Update 整次跳过，引擎键盘状态
	-- （Input.IsButtonPressed）不更新，但窗口消息照常派发。没有 DLL 时退回 Input.IsButtonPressed（只能在不挂起时用）。
	local RKeys, RHeld = lf("RRW_Keys"), lf("RRW_Held")
	local VK = {SPACE = 0x20, EQUAL = 0xBB, MINUS = 0xBD, ADD = 0x6B, SUB = 0x6D, LBR = 0xDB, RBR = 0xDD, PERIOD = 0xBE,
		DECIMAL = 0x6E, PGUP = 0x21, PGDN = 0x22, HOME = 0x24, G = 0x47, C = 0x43, H = 0x48, F12 = 0x7B, ENTER = 0x0D,
		ESC = 0x1B, BACK = 0x08, LEFT = 0x25, UP = 0x26, RIGHT = 0x27, DOWN = 0x28, A = 0x41, D = 0x44, W = 0x57, S = 0x53}
	local K = Keyboard
	local FALLBACK = {[VK.SPACE] = K.KEY_SPACE, [VK.EQUAL] = K.KEY_EQUAL, [VK.MINUS] = K.KEY_MINUS, [VK.ADD] = K.KEY_KP_ADD,
		[VK.SUB] = K.KEY_KP_SUBTRACT, [VK.LBR] = K.KEY_LEFT_BRACKET, [VK.RBR] = K.KEY_RIGHT_BRACKET, [VK.PERIOD] = K.KEY_PERIOD,
		[VK.PGUP] = K.KEY_PAGE_UP, [VK.PGDN] = K.KEY_PAGE_DOWN, [VK.HOME] = K.KEY_HOME, [VK.G] = K.KEY_G, [VK.C] = K.KEY_C,
		[VK.H] = K.KEY_H, [VK.F12] = K.KEY_F12, [VK.ENTER] = K.KEY_ENTER, [VK.ESC] = K.KEY_ESCAPE, [VK.BACK] = K.KEY_BACKSPACE,
		[VK.LEFT] = K.KEY_LEFT, [VK.UP] = K.KEY_UP, [VK.RIGHT] = K.KEY_RIGHT, [VK.DOWN] = K.KEY_DOWN, [VK.A] = K.KEY_A,
		[VK.D] = K.KEY_D, [VK.W] = K.KEY_W, [VK.S] = K.KEY_S}
	for i = 0, 9 do FALLBACK[0x30 + i] = K.KEY_0 + i; FALLBACK[0x60 + i] = K.KEY_KP_0 + i end
	local use_dll_keys = avail and RKeys ~= nil and RHeld ~= nil
	W.dll_keys = use_dll_keys
	local prev = {}
	local edges = {}
	local function poll_edges()
		edges = {}
		if use_dll_keys then
			local s = RKeys()
			if #s > 0 then
				local vks = {}
				for i = 1, #s do edges[s:byte(i)] = true; vks[#vks + 1] = s:byte(i) end
				if cfg.log_keys then log_ev("keys", {vk = vks, input = st.input}) end
			end
		else
			for vk, key in pairs(FALLBACK) do
				local d = Input.IsButtonPressed(key, 0)
				if d and not prev[vk] then edges[vk] = true end
				prev[vk] = d
			end
		end
	end
	local function pressed(vk) return edges[vk] == true end
	local function down(vk)
		if use_dll_keys then return RHeld(vk) == 1 end
		return FALLBACK[vk] ~= nil and Input.IsButtonPressed(FALLBACK[vk], 0)
	end

	local function read_keys()
		poll_edges()
		if st.input then
			for i = 0, 9 do
				if (pressed(0x30 + i) or pressed(0x60 + i)) and #st.input < 6 then st.input = st.input .. tostring(i) end
			end
			if pressed(VK.BACK) then st.input = st.input:sub(1, -2) end
			if pressed(VK.ENTER) then
				local v = tonumber(st.input)
				st.input = nil
				if v then go_to(v, "goto_key") end
				return
			end
			if pressed(VK.ESC) or pressed(VK.G) then st.input = nil end
			return
		end
		if pressed(VK.SPACE) then toggle_pause() end
		if pressed(VK.EQUAL) or pressed(VK.ADD) or pressed(VK.RBR) then set_speed(st.si + 1) end
		if pressed(VK.MINUS) or pressed(VK.SUB) or pressed(VK.LBR) then set_speed(st.si - 1) end
		if pressed(VK.PERIOD) or pressed(VK.DECIMAL) then step() end
		if pressed(VK.PGDN) then room_jump(1) end
		if pressed(VK.PGUP) then room_jump(-1) end
		if pressed(VK.HOME) then go_to(1, "home") end
		if pressed(VK.G) then st.input = "" end
		if pressed(VK.C) then
			st.cam_free = not st.cam_free
			st.cam_reset = true
			log_ev("camera", {free = st.cam_free, by = "C"})
		end
		if pressed(VK.H) then st.hint = not st.hint end
		if pressed(VK.F12) then shot(nil) end
	end

	local G, load_gui  -- GUI 资源（下文定义；glyphs 探针也要用）
	------------------------------------------------------------------ 控制文件
	local function read_ctl()
		local fh = io.open(OUT .. "/watchctl.txt", "r")
		if not fh then return end
		local text = fh:read("*a")
		fh:close()
		for ln in text:gmatch("[^\r\n]+") do
			local id, cmd, rest = ln:match("^(%d+)%s+(%S+)%s*(.-)%s*$")
			id = tonumber(id)
			if id and id > st.ctl_id then
				st.ctl_id = id
				local a, b = rest:match("^(%S+)%s*(.-)$")
				log_ev("ctl", {id = id, cmd = cmd, args = rest})
				if cmd == "goto" then go_to(tonumber(a) or st.n, "ctl")
				elseif cmd == "speed" then
					local v = tonumber(a)
					for i, s in ipairs(SPEEDS) do if s == v then set_speed(i) end end
				elseif cmd == "pause" then if not st.paused then toggle_pause() end
				elseif cmd == "resume" then if st.paused then toggle_pause() end
				elseif cmd == "toggle" then toggle_pause()
				elseif cmd == "step" then step()
				elseif cmd == "room" then room_jump(a == "prev" and -1 or 1)
				elseif cmd == "home" then go_to(1, "home")
				elseif cmd == "cam" then
					st.cam_free = (a == "free")
					st.cam_reset = true
					st.cam_off = Vector(0, 0)
				elseif cmd == "campos" then
					st.cam_free = true
					st.cam_off = Vector(tonumber(a) or 0, tonumber(b) or 0)
				elseif cmd == "camdelta" then
					st.cam_free = true
					st.cam_delta = Vector(tonumber(a) or 0, tonumber(b) or 0)
				elseif cmd == "hint" then st.hint = (a ~= "off")
				elseif cmd == "input" then st.input = a or ""
				elseif cmd == "shot" then shot(a)
				elseif cmd == "shotat" and RShotAt then RShotAt(tonumber(a) or 0, b)
				elseif cmd == "glyphs" then
					-- 字形探针：CJK 字体里缺哪些字（宽度为 0）
					if not G then load_gui() end
					local miss = {}
					for ch in rest:gmatch(utf8 and utf8.charpattern or ".") do
						if G.utf8 and G.cjk:GetStringWidthUTF8(ch) <= 0 then miss[#miss + 1] = ch end
					end
					log_ev("glyphs", {missing = table.concat(miss, ""), tested = rest})
				elseif cmd == "probe" then
					-- API 探针：Room / Game 的元表里有哪些方法与属性（判断有没有镜头 API）
					local out = {}
					local function keys(obj, tag)
						local mt = getmetatable(obj)
						local names = {}
						local seen = {}
						local function scan(t, depth)
							if type(t) ~= "table" or seen[t] or depth > 3 then return end
							seen[t] = true
							for k, v in pairs(t) do
								if type(k) == "string" then names[#names + 1] = k end
								if type(v) == "table" and (k == "__index" or k == "__propget" or k == "__propset" or k == "__class") then
									scan(v, depth + 1)
								end
							end
						end
						scan(mt, 0)
						table.sort(names)
						out[tag] = table.concat(names, " ")
					end
					keys(game:GetRoom(), "room")
					keys(game, "game")
					out.camera_global = tostring(_G.Camera)
					local ok, v = pcall(function() return game.ScreenShakeOffset end)
					out.shake = ok and v and (v.X .. "," .. v.Y) or tostring(v)
					log_ev("probe", out)
				elseif cmd == "quit" then
					st.quit = true
					apply()
					ctx.close_run("watch_quit")
				end
			end
		end
	end

	------------------------------------------------------------------ 自由镜头
	-- J746 的 Lua 没有镜头 API（Room 没有 GetCamera，全局没有 Camera 类；probe 实测），可写的只有 Game.ScreenShakeOffset
	-- （屏幕震动偏移，整个房间层按它平移渲染）。自由镜头 = 每次渲染前写这个偏移；C 切回跟随时写回 (0,0)。不能缩放（没有 API）。
	-- 偏移在 MC_POST_RENDER（影响下一次渲染，挂起时也有效）与 MC_POST_UPDATE（更新后、本迭代渲染前）各写一次。
	local cam_api = nil
	local function cam_apply()
		if not st.cam_free and not st.cam_reset then return end
		local ok = pcall(function()
			if st.cam_free then
				game.ScreenShakeOffset = Vector(st.cam_off.X, st.cam_off.Y)
			else
				game.ScreenShakeOffset = Vector(0, 0)
			end
		end)
		cam_api = ok
		if not st.cam_free then st.cam_reset = false end
	end
	W.cam_apply = cam_apply

	local function cam_update()
		local idx = game:GetLevel():GetCurrentRoomIndex()
		if st.cam_room ~= idx then
			-- 换房：回到跟随
			if st.cam_room ~= nil and st.cam_free then st.cam_free = false; st.cam_reset = true end
			st.cam_room = idx
			st.cam_off = Vector(0, 0)
		end
		st.cam_off = st.cam_off or Vector(0, 0)
		local dx, dy = 0, 0
		if not st.input then
			if down(VK.LEFT) or down(VK.A) then dx = dx - 1 end
			if down(VK.RIGHT) or down(VK.D) then dx = dx + 1 end
			if down(VK.UP) or down(VK.W) then dy = dy - 1 end
			if down(VK.DOWN) or down(VK.S) then dy = dy + 1 end
		end
		if st.cam_delta then dx, dy = dx + st.cam_delta.X, dy + st.cam_delta.Y; st.cam_delta = nil end
		if (dx ~= 0 or dy ~= 0) and not st.cam_free then
			st.cam_free = true
			st.cam_off = Vector(0, 0)
			log_ev("camera", {free = true, by = "pan"})
		end
		if st.cam_free then
			-- 镜头往左看 = 画面往右移：偏移取反；范围按房间大小限制
			local room = game:GetRoom()
			local tl, br = room:GetTopLeftPos(), room:GetBottomRightPos()
			local mx = math.max(80, (br.X - tl.X) * 0.6)
			local my = math.max(50, (br.Y - tl.Y) * 0.6)
			local ox = math.max(-mx, math.min(mx, st.cam_off.X - dx * 4))
			local oy = math.max(-my, math.min(my, st.cam_off.Y - dy * 4))
			st.cam_off = Vector(ox, oy)
		end
		cam_apply()
	end

	------------------------------------------------------------------ GUI（游戏字体 + 游戏贴图 + mod 自带 1 像素贴图）
	-- 字体：font/cjk/lanapixel.fnt（游戏自带的中日韩像素字体，UTF-8）画中文，font/pftempestasevencondensed.fnt（游戏 HUD
	-- 数字字体）画帧号/时间/倍速，font/luaminioutlined.fnt 画小字；图标：gfx/ui/ui_hearts.anm2（受伤）、gfx/ui/minimap_icons.anm2
	-- （Boss）。像素字体只用整数缩放（缩小会丢像素），布局按 GetLineHeight / GetStringWidth 算。
	G = nil
	load_gui = function()
		G = {fonts = {}, icons = {}}
		local function font(path)
			local f = Font()
			local ok = pcall(function() f:Load(path) end)
			local loaded = ok and (not f.IsLoaded or f:IsLoaded())
			G.fonts[path] = loaded
			return loaded and f or nil
		end
		G.cjk = font("font/cjk/lanapixel.fnt")
		G.num = font("font/pftempestasevencondensed.fnt") or font("font/terminus.fnt")
		G.tiny = font("font/luaminioutlined.fnt") or G.num
		G.utf8 = G.cjk ~= nil and G.cjk.DrawStringUTF8 ~= nil
		G.lh = (G.cjk and G.cjk:GetLineHeight()) or 10
		G.nlh = (G.num and G.num:GetLineHeight()) or 10
		G.tlh = (G.tiny and G.tiny:GetLineHeight()) or 8
		local px = Sprite()
		local okp = pcall(function() px:Load("gfx/irr_watch/pixel.anm2", true); px:Play("Idle", true) end)
		G.px = okp and px or nil
		local function icon(path, anim, frame)
			local s = Sprite()
			local ok = pcall(function() s:Load(path, true); s:SetFrame(anim, frame or 0) end)
			local loaded = ok and (not s.IsLoaded or s:IsLoaded())
			G.icons[path .. "#" .. anim] = loaded
			return loaded and s or nil
		end
		G.heart = icon("gfx/ui/ui_hearts.anm2", "RedHeartFull")
		G.boss = icon("gfx/ui/minimap_icons.anm2", "IconBoss")
		log_ev("gui_loaded", {fonts = G.fonts, icons = G.icons, pixel = G.px ~= nil, utf8 = G.utf8, lh = G.lh, nlh = G.nlh,
			tlh = G.tlh})
	end

	-- GUI 每次渲染要画几百个矩形/字：Vector / Color / KColor 都复用（不每次新建 userdata），把 Lua GC 压力降到最低
	-- （phase6 实测：带 GUI 的观战整局回放偶见单帧网格读数异常，怀疑与 GC 时机有关，见 REPORT）
	local VS, VP, V0 = Vector(1, 1), Vector(0, 0), Vector(0, 0)
	local CC, KC = {}, {}
	local function q(v) return math.floor(v * 255 + 0.5) end
	local function color(r, g, b, a)
		local k = q(r) | (q(g) << 8) | (q(b) << 16) | (q(a) << 24)
		local c = CC[k]
		if not c then c = Color(r, g, b, a, 0, 0, 0); CC[k] = c end
		return c
	end
	local function kcolor(r, g, b, a)
		local k = q(r) | (q(g) << 8) | (q(b) << 16) | (q(a) << 24)
		local c = KC[k]
		if not c then c = KColor(r, g, b, a); KC[k] = c end
		return c
	end
	local function rect(x, y, w, h, r, g, b, a)
		if not G.px or w <= 0 or h <= 0 then return end
		VS.X, VS.Y = w, h
		G.px.Scale = VS
		G.px.Color = color(r, g, b, a)
		VP.X, VP.Y = math.floor(x - G.ox), math.floor(y - G.oy)
		G.px:Render(VP, V0, V0)
	end

	-- 中文（CJK 字体，UTF-8）；没有 CJK 字体时用英文 en 降级
	local function zw(s, en)
		if G.utf8 then return G.cjk:GetStringWidthUTF8(s) end
		if G.num then return G.num:GetStringWidth(en or s) end
		return #(en or s) * 6
	end
	local function text(s, en, x, y, r, g, b, a, center, box)
		local c = kcolor(r, g, b, a)
		if G.utf8 then G.cjk:DrawStringUTF8(s, math.floor(x), math.floor(y), c, box or 0, center or false); return end
		if G.num then G.num:DrawString(en or s, math.floor(x), math.floor(y), c, box or 0, center or false)
		else Isaac.RenderText(en or s, x, y, r, g, b, a) end
	end
	local function num(s, x, y, r, g, b, a, f)
		f = f or G.num
		if f then f:DrawString(s, math.floor(x), math.floor(y), kcolor(r, g, b, a), 0, false)
		else Isaac.RenderText(s, x, y, r, g, b, a) end
	end
	local function numw(s, f)
		f = f or G.num
		return f and f:GetStringWidth(s) or #s * 6
	end
	local function icon(spr, x, y, sc)
		if not spr then return end
		VS.X, VS.Y = sc, sc
		spr.Scale = VS
		VP.X, VP.Y = math.floor(x - G.ox), math.floor(y - G.oy)
		spr:Render(VP, V0, V0)
	end

	local function mmss(frames)
		local s = math.floor(frames / 30)
		return ("%02d:%02d"):format(s // 60, s % 60)
	end

	local function level_label(l)
		if not l then return "", "" end
		local base, roman = (l.name or ""):match("^(.-)%s*([IVX]*)$")
		local zh = STAGE_ZH[base] or base
		return zh .. (roman ~= "" and (" " .. roman) or ""), l.name or ""
	end

	local function speed_label()
		if st.rewinding then return "REW" end
		if st.target then return ">>>" end
		if st.step_to then return "STEP" end
		if st.paused or st.ended then return "II" end
		local s = SPEEDS[st.si]
		if s >= 16 then return "x16 MAX" end
		return "x" .. ((s == 0.25 and "0.25") or (s == 0.5 and "0.5") or tostring(math.floor(s)))
	end

	local function draw_gui()
		if not G then load_gui() end
		-- 贴图渲染会跟着屏幕震动偏移（自由镜头就是写它）一起平移，字不会：贴图位置减去当前偏移，GUI 固定在屏幕上
		G.ox, G.oy = 0, 0
		pcall(function() local so = game.ScreenShakeOffset; G.ox, G.oy = so.X, so.Y end)
		local sw, sh = 480, 270
		if Isaac.GetScreenWidth then sw, sh = Isaac.GetScreenWidth(), Isaac.GetScreenHeight() end
		local n = st.n
		local lh = G.lh
		-- 底部面板：第 1 行 层/房间 + 倍速 + 帧号/时间；第 2 行 各层名；最下 进度条
		local barH = math.max(7, (G.tlh or 7) + 1)
		local barY = sh - barH - 3
		local row1 = barY - lh - 3
		local panelY = row1 - 3
		local x0, x1 = 12, sw - 12
		local bw = x1 - x0
		local function fx(f) return x0 + bw * (f - 1) / math.max(1, N - 1) end
		rect(0, panelY, sw, sh - panelY, 0.04, 0.02, 0.02, 0.66)
		rect(0, panelY, sw, 1, 0.55, 0.45, 0.35, 0.8)
		rect(x0 - 1, barY - 1, bw + 2, barH + 2, 0, 0, 0, 0.95)
		for i, l in ipairs(TL.levels) do
			local a = fx(l.n)
			local b = TL.levels[i + 1] and fx(TL.levels[i + 1].n) or x1
			local c = STAGE_COLOR[l.stage] or {0.5, 0.5, 0.5}
			rect(a, barY, b - a, barH, c[1] * 0.45, c[2] * 0.45, c[3] * 0.45, 1)
			local pa = math.min(b, fx(n))
			if pa > a then rect(a, barY, pa - a, barH, math.min(1, c[1] * 1.3), math.min(1, c[2] * 1.3), math.min(1, c[3] * 1.3), 1) end
			rect(a, barY - 2, 1, barH + 2, 0.95, 0.9, 0.8, 0.9)
			-- 层名（小字，画在条里）：B1 B2 C1 ...（层号 = stage）
			local base, roman = (l.name or ""):match("^(.-)%s*([IVX]*)$")
			local short = (STAGE_SHORT[base] or "L") .. ((roman == "II" and "2") or (roman == "I" and "1") or "")
			if G.tiny and b - a > numw(short, G.tiny) + 4 then
				local cur = n >= l.n and (not TL.levels[i + 1] or n < TL.levels[i + 1].n)
				num(short, a + 2, barY + math.floor((barH - G.tlh) / 2), 1, 1, 1, cur and 1 or 0.65, G.tiny)
			end
		end
		for _, r in ipairs(TL.rooms) do rect(fx(r.n), barY + barH - 2, 1, 2, 0, 0, 0, 0.5) end
		for _, e in ipairs(TL.events) do
			local x = fx(e.n)
			if e.k == "damage" then rect(x, barY - 2, 1, 2, 1, 0.12, 0.12, 1)
			elseif e.k == "boss" then rect(x - 1, barY - 3, 3, 3, 1, 0.8, 0.15, 1)
			elseif e.k == "death" then rect(x - 1, barY - 5, 3, 5, 0.85, 0.85, 0.85, 1)
			elseif e.k == "pause" then rect(x, barY + barH, 1, 2, 0.45, 0.7, 1, 1) end
		end
		local px = fx(n)
		if st.target then rect(fx(st.target), barY - 3, 1, barH + 6, 0.4, 0.9, 1, 1) end
		rect(px - 1, barY - 3, 3, barH + 6, 1, 1, 1, 1)
		rect(px - 2, barY - 4, 5, 1, 1, 1, 1, 1)
		-- 第 1 行
		local l = level_at(n)
		local r = room_at(n)
		local zh, en = level_label(l)
		local rt = r and (ROOM_TYPES[r.type] or tostring(r.type)) or ""
		local lt = zh .. (r and ("  ·  " .. rt .. "  #" .. r.idx) or "")
		local le = en .. (r and ("  room " .. r.idx) or "")
		text(lt, le, x0, row1, 1, 0.95, 0.85, 1)
		local leftEnd = x0 + zw(lt, le)
		-- 右下角留给游戏自己的 HUD（口袋道具/卡牌名），数字靠右但让开约 100 个单位
		local right = ("%d / %d   %s / %s"):format(n, N, mmss(n), mmss(N))
		local rw = numw(right)
		local ny = row1 + math.floor((lh - G.nlh) / 2)
		x1 = sw - 100
		num(right, x1 - rw, ny, 1, 1, 1, 1)
		local sp = speed_label()
		local sc = (st.paused or st.ended) and {1, 0.85, 0.3} or ((st.target or st.rewinding) and {0.4, 0.9, 1} or {0.65, 1, 0.55})
		num(sp, x1 - rw - numw(sp) - 12, ny, sc[1], sc[2], sc[3], 1)
		-- 图例（游戏自带贴图）
		local lgx = x1 - rw - numw(sp) - 12
		if G.heart and G.boss and lgx - 70 > leftEnd + 8 then
			local tx = lgx - 70
			icon(G.heart, tx, row1 + lh / 2 + 1, 0.5)
			text("受伤", "hit", tx + 6, row1, 0.85, 0.75, 0.7, 0.9)
			icon(G.boss, tx + 34, row1 + lh / 2 + 1, 0.7)
			text("Boss", "boss", tx + 41, row1, 0.85, 0.75, 0.7, 0.9)
		end
		-- 暂停标志（屏幕上方正中）
		if (st.paused or st.ended) and not st.target and not st.rewinding and not st.step_to then
			rect(sw / 2 - 9, 10, 6, 18, 1, 1, 1, 0.85)
			rect(sw / 2 + 3, 10, 6, 18, 1, 1, 1, 0.85)
			local t = st.ended and "回放结束" or "暂停"
			text(t, st.ended and "END" or "PAUSED", sw / 2 - zw(t, t) / 2, 30, 1, 1, 1, 0.95)
		end
		-- 快进 / 倒带
		if st.target or st.rewinding then
			local tgt = st.target or st.rewind_to or 0
			-- 速度估计：最近 0.5 s 的帧/秒；刚开始还没测出来时按 400 帧/秒（gate 全速的典型值）先估
			local fps = st.fps > 0 and st.fps or 400
			local left = (tgt - n) / fps
			local rew = st.rewinding or st.kind == "rewind"
			local l1 = ("%s 至 %d"):format(rew and "正在倒带" or "正在快进", tgt)  -- lanapixel 没有“帧”字
			local e1 = ("%s to frame %d"):format(rew and "REWINDING" or "FAST-FORWARD", tgt)
			local l2 = st.rewinding and "重启回放进程，从开局精确重跑…" or (left and ("剩余约 %d 秒"):format(math.ceil(left)) or "估算中…")
			local e2 = st.rewinding and "restarting replay" or (left and ("~%d s"):format(math.ceil(left)) or "")
			local boxw = math.max(zw(l1, e1), zw(l2, e2)) + 24
			local bx, by = sw / 2 - boxw / 2, sh / 2 - lh * 2
			rect(bx, by, boxw, lh * 2 + 14, 0.04, 0.02, 0.02, 0.82)
			rect(bx, by, boxw, 1, 0.75, 0.6, 0.4, 1)
			rect(bx, by + lh * 2 + 13, boxw, 1, 0.75, 0.6, 0.4, 1)
			text(l1, e1, bx, by + 2, 1, 0.95, 0.85, 1, true, boxw)
			text(l2, e2, bx, by + lh + 2, 0.85, 0.85, 0.85, 1, true, boxw)
			local from = st.n0 or 1
			local frac = (tgt > from) and (n - from) / (tgt - from) or 0
			rect(bx + 10, by + lh * 2 + 7, boxw - 20, 3, 0.2, 0.2, 0.2, 1)
			rect(bx + 10, by + lh * 2 + 7, (boxw - 20) * math.max(0, math.min(1, frac)), 3, 0.4, 0.9, 1, 1)
		end
		-- 跳帧输入
		if st.input then
			local t = "跳转至：" .. st.input .. "_"
			local w = math.max(zw(t, t), 90) + 24
			local s2 = "Enter 确认  Esc 取消"
			rect(sw / 2 - w / 2, sh / 2 - lh * 4, w, lh * 2 + 8, 0.04, 0.02, 0.02, 0.88)
			rect(sw / 2 - w / 2, sh / 2 - lh * 4, w, 1, 0.75, 0.6, 0.4, 1)
			text(t, "GOTO " .. st.input .. "_", sw / 2 - w / 2, sh / 2 - lh * 4 + 3, 1, 1, 1, 1, true, w)
			text(s2, "Enter / Esc", sw / 2 - w / 2, sh / 2 - lh * 3 + 3, 0.7, 0.7, 0.7, 1, true, w)
		end
		-- 热键提示（右侧、小地图下方）
		if st.hint then
			local lines = HINT_LINES
			local w = 0
			for _, t in ipairs(lines) do w = math.max(w, zw(t, t)) end
			w = w + 10
			local hx, hy = sw - w - 6, 62
			rect(hx, hy, w, #lines * lh + 6, 0.04, 0.02, 0.02, 0.6)
			rect(hx, hy, 1, #lines * lh + 6, 0.75, 0.6, 0.4, 0.8)
			for i, t in ipairs(lines) do text(t, t, hx + 6, hy + 3 + (i - 1) * lh, 0.95, 0.9, 0.8, 0.95) end
		else
			local t = "H  热键"
			text(t, "H keys", sw - zw(t, t) - 8, 62, 0.9, 0.85, 0.75, 0.7)
		end
		if st.cam_free then
			local t = "自由镜头（C 跟随玩家）"
			text(t, "free camera (C)", sw / 2 - zw(t, t) / 2, panelY - lh - 2, 0.6, 0.9, 1, 0.95)
		end
	end

	local function write_state()
		local fh = io.open(OUT .. "/watchstate.tmp", "w")
		if not fh then return end
		local info = {}
		if avail then info = {RInfo()} end
		fh:write(ctx.jval({n = st.n, N = N, paused = st.paused, speed = SPEEDS[st.si], target = st.target, kind = st.kind,
			ended = st.ended, rewinding = st.rewinding, cam_free = st.cam_free, cam_api = cam_api,
			cam_off = st.cam_off and {st.cam_off.X, st.cam_off.Y} or nil, dll_keys = W.dll_keys, hint = st.hint, input = st.input,
			fps = st.fps, dll = avail, rate = info[8], iterations = info[3], skipped = info[4], presents = info[5],
			presents_skipped = info[6], wall_ms = info[7], cap_frames = info[9], cap_dups = info[10], cap_w = info[11],
			cap_h = info[12], ctl_id = st.ctl_id, t = now_ms(), mode = W.mode, quit = st.quit,
			gui = G and {fonts = G.fonts, icons = G.icons, pixel = G.px ~= nil, utf8 = G.utf8} or nil}))
		fh:close()
		-- os.rename 在目标存在时失败（Windows）：先删再改名
		local src, dst = OUT .. "/watchstate.tmp", OUT .. "/watchstate.json"
		if not os.rename(src, dst) then os.remove(dst); os.rename(src, dst) end
		st.last_state = now_ms()
	end

	function W.on_render()
		st.renders = st.renders + 1
		if W.mode == "capture" then
			-- 回放局已关（帧上限）但游戏不再做逻辑更新（录制以暂停菜单里退出结束时，帧计数停住、MC_POST_UPDATE 不再来）：
			-- 等 60 次渲染仍没有更新，就把最近一次渲染当作最后一帧的"最后一次渲染"写出并结束导出
			if st.cap_last and not st.cap_closed and ctx.run() == nil then
				st.after_close = (st.after_close or 0) + 1
				if st.after_close >= 60 then
					if RMark then RMark(st.cap_last + 1) end
					log_ev("capture_finish_by_render", {renders = st.after_close})
					finish_capture()
				end
			end
			if now_ms() - st.last_state > 500 then write_state() end
			return
		end
		-- 回放局开出来之前不响应热键与控制命令（按键沿照样取走，免得开局后误触发）
		if not st.live then
			if use_dll_keys then RKeys() end
			if now_ms() - st.last_state > 250 then write_state() end
			return
		end
		local okk, ek = pcall(read_keys)
		if not okk then ctx.fail("watch_keys", ek) end
		-- 控制文件：挂起/慢放时每次渲染都读（反应快），其余每 6 次渲染读一次
		if st.renders % 6 == 0 or st.paused or SPEEDS[st.si] < 1 then
			local okc, ec = pcall(read_ctl)
			if not okc then ctx.fail("watch_ctl", ec) end
		end
		local okm, em = pcall(cam_update)
		if not okm then ctx.fail("watch_cam", em); cam_api = false end
		-- 这次渲染不提交绘制（快进/高倍速时 DLL 节流呈现）就不画 GUI，省 CPU
		if cfg.gui ~= false and st.draw_this ~= false then
			local okg, eg = pcall(draw_gui)
			if not okg then
				st.gui_err = (st.gui_err or 0) + 1
				if st.gui_err <= 3 then ctx.fail("watch_gui", eg) end
			end
		end
		if now_ms() - st.last_state > 250 then write_state() end
	end

	apply()
	log_ev("watch_loaded", {mode = W.mode, dll = avail, frames = N, start_target = cfg.start_target,
		rooms = #TL.rooms, levels = #TL.levels, events = #TL.events})
	return W
end
