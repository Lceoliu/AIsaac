-- 全局 MT（引擎 GlobalMT）的开局 reseed 与每帧只读校验（DESIGN_BRIEF §7 E3/E4/E7）。
-- 依赖 rl/bridge/engine/j746_rec 的 rr_mt.dll：录制会话用挂起式注入器在游戏启动前注入，Python 把它的路径放进
-- 环境变量 IRR_MT_DLL，这里用 package.loadlib 取已加载模块的导出函数（同一路径 = 同一模块，不会二次初始化）。
--   RRMT_Seed(seed) -> cursor, hash   调用引擎自己的 GlobalMT_Seed（会改变之后的随机数流）
--   RRMT_State()    -> cursor, hash, reads   只读（624 个状态字 + cursor 的 FNV-1a；两份拷贝一致才返回，reads = 读的次数），不推进 MT
--   RRMT_Info()     -> ready, profile_calls, seed_calls, pe_timestamp, size_of_image
-- 没有 IRR_MT_DLL（-NoDll 会话）、loadlib 失败、或 DLL 的身份门没通过（EXE 不是 J746，RRMT_State 不返回值）时
-- available=false：不 reseed、帧行不写 mt，录制照常（E7）。
-- salt 生成只用 os.time / os.clock / Isaac.GetTime / Isaac.GetFrameCount / 表地址的哈希，不碰任何游戏 RNG，
-- 也不用 math.random（Lua 5.3 的 math.random 是 C rand()）。
local M = {available = false, reason = "not_initialised", dll = nil, info = nil, early_done = false, pending = nil,
	counter = 0, seed_fn = nil, state_fn = nil}
local game = Game()

function M.init(getenv, hash)
	M.hash = hash
	local path = getenv("IRR_MT_DLL")
	if not path or path == "" then M.reason = "no_dll"; return M end
	M.dll = path
	if not package or not package.loadlib then M.reason = "package.loadlib unavailable"; return M end
	local okc, seed_fn, e1 = pcall(package.loadlib, path, "RRMT_Seed")
	local oks, state_fn, e2 = pcall(package.loadlib, path, "RRMT_State")
	local oki, info_fn = pcall(package.loadlib, path, "RRMT_Info")
	local oku, unstable_fn = pcall(package.loadlib, path, "RRMT_UnstableReads")
	if oku and unstable_fn then M.unstable_fn = unstable_fn end
	if not (okc and seed_fn and oks and state_fn) then
		-- loadlib 的错误信息是系统代码页（中文 Windows 上是 GBK）：非 ASCII 字节换成 ?，去掉换行，保证写出的 JSON 是合法 UTF-8
		local msg = tostring(okc and e1 or seed_fn) .. " / " .. tostring(oks and e2 or state_fn)
		M.reason = "loadlib failed: " .. msg:gsub("[\128-\255]", "?"):gsub("[\r\n]+", " ")
		return M
	end
	if oki and info_fn then
		local ok, ready, pcalls, scalls, ts, size = pcall(info_fn)
		if ok then M.info = {ready = ready, profile_calls = pcalls, seed_calls = scalls, pe_timestamp = ts, size_of_image = size} end
	end
	local ok, c, h = pcall(state_fn)
	if not ok or c == nil or h == nil then
		M.reason = "identity gate failed (RRMT_State returned nothing)"
		return M
	end
	M.seed_fn, M.state_fn = seed_fn, state_fn
	M.available = true
	M.reason = "ok"
	return M
end

-- 只读 (cursor, hash)；不可用时 nil
function M.state()
	if not M.available then return nil end
	return M.state_fn()
end

-- DLL 自进程启动以来"两份拷贝不一致"的 MT 读数次数（rr_mt.cpp stableSnapshot；不可用时 nil）
function M.unstable_reads()
	if not M.unstable_fn then return nil end
	local ok, n = pcall(M.unstable_fn)
	if ok then return n end
	return nil
end

-- phase8-mouse：回放还原录制的鼠标。rr_mt.dll 只有在环境变量 RR_MT_MOUSE=1（replay_run.py 给回放进程设）且 J746 身份门
-- 通过时才导出可用的 RRMT_SetMouse(x, y, buttons, mode) / RRMT_GetMouse()（见 rr_mt.cpp）；旧 DLL 没有这两个导出。
-- 返回是否可用；原因记在 M.mouse_reason。只在回放局开局时调用（窗口已建好）。
function M.mouse_init()
	if M.mouse_checked then return M.mouse_ok end
	M.mouse_checked = true
	M.mouse_ok = false
	if not M.dll or not package or not package.loadlib then M.mouse_reason = "no_dll"; return false end
	local ok1, setf = pcall(package.loadlib, M.dll, "RRMT_SetMouse")
	local ok2, getf = pcall(package.loadlib, M.dll, "RRMT_GetMouse")
	if not (ok1 and setf and ok2 and getf) then M.mouse_reason = "dll has no RRMT_SetMouse (old build)"; return false end
	local ok3, x = pcall(getf)
	if not ok3 or x == nil then
		M.mouse_reason = "RRMT_GetMouse returned nothing (RR_MT_MOUSE!=1, identity gate, or no window)"
		return false
	end
	M.mouse_set, M.mouse_get = setf, getf
	M.mouse_ok = true
	M.mouse_reason = "ok"
	return true
end

-- phase10-mods-console：控制台命令。rr_mt.dll 在环境变量 RR_MT_CONSOLE=1（录制会话 --allow-console）且身份门 + 代码签名
-- 通过时挂引擎的 Console::SubmitInput，把玩家在调试控制台提交的输入行排队；RRMT_ConsoleTake() 取走一条（line, k, 剩余数）。
-- 返回是否可用；原因记在 M.console_reason。
function M.console_init()
	if M.console_checked then return M.console_ok end
	M.console_checked = true
	M.console_ok = false
	if not M.dll or not package or not package.loadlib then M.console_reason = "no_dll"; return false end
	local ok1, takef = pcall(package.loadlib, M.dll, "RRMT_ConsoleTake")
	local ok2, infof = pcall(package.loadlib, M.dll, "RRMT_ConsoleInfo")
	if not (ok1 and takef and ok2 and infof) then M.console_reason = "dll has no RRMT_ConsoleTake (old build)"; return false end
	local ok3, on = pcall(infof)
	if not ok3 or on ~= 1 then
		M.console_reason = "console hook not installed (RR_MT_CONSOLE!=1, identity gate or code signature)"
		return false
	end
	M.console_take, M.console_info = takef, infof
	M.console_ok = true
	M.console_reason = "ok"
	return true
end

-- phase11-human3：写 Seeds 的"自定义种子局"标记（rr_mt.dll 的 RRMT_SetCustomRun；回放自然局的录制时把 `seed` 命令
-- 开出的局改回自然局）。返回 ok(1/0), 旧值, 说明；旧 DLL 没有这个导出时 ok = 0
function M.set_custom_run(seeds, start_seed, value)
	if not M.dll or not package or not package.loadlib then return 0, -1, "no_dll" end
	if M.setcr == nil then
		local ok, f = pcall(package.loadlib, M.dll, "RRMT_SetCustomRun")
		M.setcr = (ok and f) or false
	end
	if not M.setcr then return 0, -1, "dll has no RRMT_SetCustomRun (old build)" end
	return M.setcr(seeds, start_seed, value)
end

-- phase12-human4（诊断）：rr_mt.dll 在 RR_MT_TRACE=1 时挂 GlobalMT_Next，RRMT_Trace() 取走自上次以来主线程每次取数的
-- 调用者 RVA（十六进制、空格分隔）。没有这个导出或没装钩子时 trace_ok = false。只给 harness 诊断（irr_diag.jsonl）用。
function M.trace_init()
	if M.trace_checked then return M.trace_ok end
	M.trace_checked = true
	M.trace_ok = false
	if not M.dll or not package or not package.loadlib then return false end
	local ok, f = pcall(package.loadlib, M.dll, "RRMT_Trace")
	if not (ok and f) then return false end
	local ok2, s = pcall(f)
	if not ok2 or s == nil then return false end
	M.trace_fn = f
	M.trace_ok = true
	return true
end

-- 返回 (rva 串, 其它线程取数累计, 丢失条数, 总取数累计)；不可用时 nil
function M.trace_take()
	if not M.trace_ok then return nil end
	local ok, s, other, _, lost, total = pcall(M.trace_fn)
	if not ok then return nil end
	return s, other, lost, total
end

-- 每局随机的 32 位 salt（非 0）
function M.new_salt()
	M.counter = M.counter + 1
	local fl = math.floor
	local s = string.pack("<jdjjjs", fl(os.time()), os.clock() + 0.0, fl(Isaac.GetTime()), fl(Isaac.GetFrameCount()), M.counter,
		tostring({}))
	local h = M.hash(s)
	if h == 0 then h = 1 end
	return h
end

-- reseed 一次：值 = StartSeed xor salt。返回记录（写进 header.mt.reseeds）
-- paused（recorder 0.4.2，phase12）：reseed 当时引擎是否认为游戏暂停（game:IsPaused()）。玩家在控制台敲 rewind/restart 时，
-- 下一局开局时控制台还开着 = 暂停；开局前那一轮实体预更新里 Entity::IsFrame 因此恒为 false，回放要照样暂停（replay.lua）
function M.reseed(at, salt, replayed)
	local start = game:GetSeeds():GetStartSeed()
	local value = (start ~ salt) & 0xFFFFFFFF
	local paused = game:IsPaused()
	local c0, h0 = M.state_fn()
	local c1, h1 = M.seed_fn(value)
	return {at = at, gf = game:GetFrameCount(), ifc = Isaac.GetFrameCount(), t = Isaac.GetTime(), start_seed = start,
		salt = salt, seeded_with = value, before = {c0, h0}, after = {c1, h1}, from_replay = replayed or false, paused = paused}
end

-- phase12：回放开局窗口强制暂停（rr_mt.dll 在 RR_MT_FORCEPAUSE=1 时挂 Game::IsPaused、导出 RRMT_ForcePause）。
-- 返回 (ok, 之前的状态, 累计强制次数)；没有这个导出 / 没装钩子时 ok = false
function M.force_pause(on)
	if not M.dll or not package or not package.loadlib then return false, nil, nil end
	if M.fp_fn == nil then
		local ok, f = pcall(package.loadlib, M.dll, "RRMT_ForcePause")
		M.fp_fn = (ok and f) or false
	end
	if not M.fp_fn then return false, nil, nil end
	local ok, prev, hits = pcall(M.fp_fn, on and 1 or 0)
	if not ok or prev == nil then return false, nil, nil end
	return true, prev, hits
end

-- header.mt 的公共部分
function M.header_base()
	return {available = M.available, reason = M.reason, dll = M.dll, info = M.info}
end

return M
