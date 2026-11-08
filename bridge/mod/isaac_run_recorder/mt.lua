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
function M.reseed(at, salt, replayed)
	local start = game:GetSeeds():GetStartSeed()
	local value = (start ~ salt) & 0xFFFFFFFF
	local c0, h0 = M.state_fn()
	local c1, h1 = M.seed_fn(value)
	return {at = at, gf = game:GetFrameCount(), ifc = Isaac.GetFrameCount(), t = Isaac.GetTime(), start_seed = start,
		salt = salt, seeded_with = value, before = {c0, h0}, after = {c1, h1}, from_replay = replayed or false}
end

-- header.mt 的公共部分
function M.header_base()
	return {available = M.available, reason = M.reason, dll = M.dll, info = M.info}
end

return M
