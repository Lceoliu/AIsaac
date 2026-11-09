-- 第三方 mod 对游戏状态的"玩家操作之外"的改动（recorder 0.4.0，phase10-mods-console）：录"效果"而不是录 mod 的输入。
-- 例：GoodTrip（小地图传送）按住地图键 + 鼠标/射击键选房间后调用 Game():StartRoomTransition / Game():ChangeRoom /
-- player:TakeDamage / player:SetShootingCooldown、改 level.EnterDoor/LeaveDoor、Level:RemoveCurses/AddCurse、
-- Isaac.ExecuteCommand("restart")。这些调用的触发条件（地图键 = ButtonAction 12、鼠标、mod 内部状态）都不在回放注入的
-- 12 个动作里，回放时 mod 自己复现不了；所以录制时把调用本身（帧、帧内位置、方法、参数、来源 mod）写成 api 行，
-- 回放时把该 mod 临时禁用（replay_run.py 放 disable.it），由 replay.lua 在同一位置原样调用。
--
-- 怎么截获（J746 实测，phase10 probe1/probe2）：
--   * 不能把 Lua 函数 rawset 进 C 类元表（下一次查找时 C 的 oldIndex 断言 "Bad Index!" 崩溃，probe1）；
--   * 照游戏自己 resources/scripts_rep/main.lua 的 BeginClass/EndClass：替换类实例元表的 __index（被监视的键返回包装函数，
--     其余转给原 __index），属性写入替换 __newindex；全局表 Isaac 的函数直接赋值替换。别的 mod 的
--     Game():ChangeRoom(...)、player:SetShootingCooldown(2)、level.EnterDoor = -1、Isaac.ExecuteCommand(...) 都是
--     每次调用时动态查找，所以都会走到包装（probe2：另一个 mod 的调用全部计到）。
--   * 包装只在"不是本 mod 自己发起"时记录（internal 计数）；来源 = 调用栈上第一个 mods/<目录>/ 下的文件。
--
-- 帧内位置（ph）：调用时比较 Game 帧计数与本 mod 最后一次自己回调的标记（Isaac.GetFrameCount 在每次迭代的输入轮询/逻辑更新
-- 之后、渲染之前 +1，probe4 实测）：
--   "u"  逻辑更新进行中（gf 已 +1，本 mod 的 MC_POST_UPDATE 还没跑）：第 n 帧的更新里（rc = 1：本帧更新里已经换了房间）
--   "p"  本 mod 的 MC_POST_UPDATE 之后、同一次迭代（别的 mod 的 MC_POST_UPDATE 排在本 mod 后面时）
--   "r"  第 r 次渲染（= 上一次 MC_POST_UPDATE 之后第 r 次 MC_POST_RENDER；af = 1：发生在本 mod 的渲染回调之后）
--   "q"  第 r 次渲染之后、下一次渲染之前的输入轮询里（例如别的 mod 的 MC_INPUT_ACTION）
--   "gs" 局还没开（开局 MC_POST_GAME_STARTED 之前）：写在 header 之后，n = 1
-- 嵌套：在另一个被记录的调用执行期间发生的调用（例如 ChangeRoom 同步触发 MC_POST_NEW_ROOM，里面又 AddCurse）带 in = 父调用 id。
local M = {}
local game = Game()
local concat = table.concat

M.VERSION = 1
-- 监视的方法：类 -> 方法名列表；属性写入：类 -> 属性名；全局函数：表名 -> 函数名
M.METHODS = {
	Game = {"ChangeRoom", "StartRoomTransition"},
	Level = {"ChangeRoom", "RemoveCurses", "AddCurse", "SetStage"},
	EntityPlayer = {"SetShootingCooldown", "TakeDamage", "AddControlsCooldown", "UseActiveItem", "UseCard", "UsePill",
		"AnimateTeleport"},
	SFXManager = {"Play"},
}
M.PROPS = {Level = {"EnterDoor", "LeaveDoor"}}
M.GLOBALS = {Isaac = {"ExecuteCommand"}}

M.internal = 0          -- >0：本 mod 自己的调用（harness 命令、回放执行），不记录
M.on = false            -- 包装是否装上
M.installed = {}        -- 装上的项（写进 header.api）
M.failed = {}           -- 没装上的项与原因
M.seq = 0               -- 本局调用序号（id）
M.depth = 0
M.stack = {}            -- 正在执行的被记录调用的 id
M.pending = {}          -- 局外（R == nil）发生的调用，开局后写在 header 之后
M.counts = {}           -- 本局各方法记录次数
-- 本 mod 最后一次自己的回调：kind = "U"（MC_POST_UPDATE）/ "R"（MC_POST_RENDER）/ "Q"（MC_INPUT_ACTION），
-- ifc = 当时的 Isaac.GetFrameCount()，rs = 当时"上次更新之后的渲染次数"
M.mk, M.mk_ifc, M.mk_rs = "U", -1, 0

local ctx = nil   -- main.lua 注入：run() -> R、line(s)、jstr、renders()（上次更新之后的渲染次数）、self_dir（本 mod 目录名）
local mt_cache = {}

function M.mark(kind, rs)
	M.mk = kind
	M.mk_ifc = Isaac.GetFrameCount()
	M.mk_rs = rs or 0
end

------------------------------------------------------------------ 参数编码
local function class_mt(name)
	if mt_cache[name] ~= nil then return mt_cache[name] or nil end
	local mt = nil
	pcall(function()
		local T = _G[name]
		if type(T) == "function" then mt = getmetatable(T())
		elseif T ~= nil then mt = getmetatable(T).__class end
	end)
	mt_cache[name] = mt or false
	return mt
end

local function player_index(p)
	local h = GetPtrHash(p)
	for i = 0, game:GetNumPlayers() - 1 do
		local q = Isaac.GetPlayer(i)
		if q and GetPtrHash(q) == h then return i end
	end
	return nil
end

local function fnum(x)
	if x ~= x or x == math.huge or x == -math.huge then return "null" end
	return ("%.17g"):format(x)
end

-- 一个参数 -> JSON（见 README §4.7 api 行）
local function enc(v)
	local t = type(v)
	if t == "nil" then return "null"
	elseif t == "boolean" then return v and "true" or "false"
	elseif t == "string" then return ctx.jstr(v)
	elseif t == "number" then
		if math.type(v) == "integer" then return tostring(v) end
		return '{"f":' .. fnum(v) .. '}'
	elseif t == "userdata" then
		local mt = getmetatable(v)
		if mt == class_mt("Vector") then return '{"v":[' .. fnum(v.X) .. "," .. fnum(v.Y) .. "]}" end
		if mt == class_mt("EntityRef") then
			local e = v.Entity
			if e == nil then return '{"ref":null}' end
			local p = e:ToPlayer()
			local k = p and player_index(p)
			if k then return '{"ref":{"p":' .. k .. "}}" end
			return '{"ref":{"e":[' .. e.Type .. "," .. e.Variant .. "," .. e.Index .. "]}}"
		end
		local ok, p = pcall(function() return v:ToPlayer() end)
		if ok and p then
			local k = player_index(p)
			if k then return '{"p":' .. k .. "}" end
		end
		local ok2, idx = pcall(function() return v.Index end)
		if ok2 and idx then return '{"e":[' .. v.Type .. "," .. v.Variant .. "," .. idx .. "]}" end
		return '{"u":' .. ctx.jstr(mt and rawget(mt, "__type") or "userdata") .. "}"
	end
	return '{"u":' .. ctx.jstr(t) .. "}"
end

-- 来源 mod：从 level0 起往上找调用栈里第一个 mods/<目录>/ 下、且不是本 mod 的文件（本 mod 自己的帧——包装函数、
-- harness、replay——跳过）；只找到本 mod = 本 mod 自己的调用（不记录）；找不到 mod 目录时返回 nil 与第一个 Lua 源文件名
local function source_of(level0)
	local first, saw_self = nil, false
	for lv = level0, level0 + 14 do
		local info = debug.getinfo(lv, "S")
		if not info then break end
		local s = info.source or ""
		local m = s:match("[/\\]mods[/\\]([^/\\]+)[/\\]")
		if m and m ~= ctx.self_dir then return m, s end
		if m then saw_self = true
		elseif not first and info.what ~= "C" and s ~= "=[C]" then first = s end
	end
	if saw_self and not first then return ctx.self_dir, nil end
	return nil, first
end

------------------------------------------------------------------ 记录
local function phase(R)
	local gf = game:GetFrameCount()
	if gf > R.gf_last then
		local rc = (R.room_last ~= nil and game:GetLevel():GetCurrentRoomIndex() ~= R.room_last) and 1 or 0
		return "u", nil, nil, rc
	end
	local ifc = Isaac.GetFrameCount()
	if ifc > M.mk_ifc then return "r", M.mk_rs + 1, 0 end
	if M.mk == "R" then return "r", M.mk_rs, 1 end
	if M.mk == "U" then return "p", 0 end
	return "q", M.mk_rs
end

-- 返回调用 id（记录了）或 nil（不记录）
local function begin(mname, selfv, kind, args, nargs, level0)
	local src, srcfile = source_of(level0)
	if src ~= nil and src == ctx.self_dir then return nil end
	local enc_args = {}
	for i = 1, nargs do enc_args[i] = enc(args[i]) end
	M.seq = M.seq + 1
	local id = M.seq
	local parent = M.stack[#M.stack]
	local R = ctx.run()
	local parts = {'"id":', id, ',"m":', ctx.jstr(mname)}
	if kind == "set" then parts[#parts + 1] = ',"set":1' end
	if selfv then parts[#parts + 1] = ',"self":' .. selfv end
	parts[#parts + 1] = ',"a":[' .. concat(enc_args, ",") .. "]"
	parts[#parts + 1] = ',"src":' .. ctx.jstr(src or "?")
	if not src then parts[#parts + 1] = ',"srcfile":' .. ctx.jstr(srcfile) end
	if parent then parts[#parts + 1] = ',"in":' .. parent end
	M.counts[mname] = (M.counts[mname] or 0) + 1
	if R and game:GetFrameCount() >= R.gf_last then
		local ph, r, af, rc = phase(R)
		local s = '{"k":"api","n":' .. (R.n + 1) .. ',"gf":' .. game:GetFrameCount() .. ',"ph":"' .. ph .. '"'
			.. (r and (',"r":' .. r) or "") .. (af == 1 and ',"af":1' or "") .. (rc == 1 and ',"rc":1' or "")
			.. "," .. concat(parts) .. "}"
		ctx.line(s)
	elseif #M.pending < 256 then
		-- 局外（菜单、开局 MC_POST_GAME_STARTED 之前、重开时新局已在生成）：开局后写在 header 之后
		M.pending[#M.pending + 1] = '"gf":' .. game:GetFrameCount() .. ',"ph":"gs",' .. concat(parts)
	end
	return id
end

-- 执行原调用；被记录的调用在执行期间压栈（嵌套调用记 in = 它的 id）
local function run_call(id, f, ...)
	if not id then return f(...) end
	M.stack[#M.stack + 1] = id
	local r = table.pack(pcall(f, ...))
	M.stack[#M.stack] = nil
	if not r[1] then error(r[2], 0) end
	return table.unpack(r, 2, r.n)
end

local function self_enc(cls, obj)
	if cls == "EntityPlayer" then
		local k = player_index(obj)
		return k and ('{"p":' .. k .. "}") or '{"p":null}'
	end
	return nil
end

local function method_wrapper(cls, name, old_index)
	local mname = cls .. "." .. name
	return function(self, ...)
		local f = old_index(self, name)
		if M.internal > 0 or not ctx.recording() then return f(self, ...) end
		local n = select("#", ...)
		local id = begin(mname, self_enc(cls, self), "call", {...}, n, 3)
		return run_call(id, f, self, ...)
	end
end

------------------------------------------------------------------ 安装
function M.install(c)
	ctx = c
	for cls, names in pairs(M.METHODS) do
		local mt = class_mt(cls)
		local old = mt and rawget(mt, "__index")
		if type(old) ~= "function" then
			M.failed[#M.failed + 1] = cls .. ": no __index function"
		else
			local W = {}
			for _, name in ipairs(names) do
				W[name] = method_wrapper(cls, name, old)
				M.installed[#M.installed + 1] = cls .. "." .. name
			end
			local ok, err = pcall(rawset, mt, "__index", function(self, k)
				local w = W[k]
				if w ~= nil then return w end
				return old(self, k)
			end)
			if not ok then M.failed[#M.failed + 1] = cls .. ": " .. tostring(err) end
		end
	end
	for cls, names in pairs(M.PROPS) do
		local mt = class_mt(cls)
		local old = mt and rawget(mt, "__newindex")
		if type(old) ~= "function" then
			M.failed[#M.failed + 1] = cls .. ": no __newindex function"
		else
			local P = {}
			for _, name in ipairs(names) do
				P[name] = cls .. "." .. name
				M.installed[#M.installed + 1] = cls .. "." .. name .. "="
			end
			local ok, err = pcall(rawset, mt, "__newindex", function(self, k, v)
				local mname = P[k]
				if mname == nil or M.internal > 0 or not ctx.recording() then return old(self, k, v) end
				local id = begin(mname, self_enc(cls, self), "set", {v}, 1, 3)
				return run_call(id, old, self, k, v)
			end)
			if not ok then M.failed[#M.failed + 1] = cls .. " __newindex: " .. tostring(err) end
		end
	end
	for tname, names in pairs(M.GLOBALS) do
		local T = _G[tname]
		for _, name in ipairs(names) do
			local orig = T and T[name]
			if type(orig) ~= "function" then
				M.failed[#M.failed + 1] = tname .. "." .. name .. ": not a function"
			else
				local mname = tname .. "." .. name
				M.raw = M.raw or {}
				M.raw[mname] = orig
				local ok, err = pcall(function()
					T[name] = function(...)
						if M.internal > 0 or not ctx.recording() then return orig(...) end
						local id = begin(mname, nil, "call", {...}, select("#", ...), 3)
						return run_call(id, orig, ...)
					end
				end)
				if ok then M.installed[#M.installed + 1] = mname else M.failed[#M.failed + 1] = mname .. ": " .. tostring(err) end
			end
		end
	end
	M.on = #M.installed > 0
	return M
end

-- 本 mod 自己执行控制台命令（harness 命令）：不经过包装、不记 api 行
function M.exec_raw(cmd)
	local f = M.raw and M.raw["Isaac.ExecuteCommand"] or Isaac.ExecuteCommand
	M.internal = M.internal + 1
	local ok, err = pcall(f, cmd)
	M.internal = M.internal - 1
	if not ok then error(err, 0) end
end

-- 开局：清本局计数，把局外暂存的调用写在 header 之后（n = 1）
function M.on_run_open()
	M.seq = 0
	M.counts = {}
	M.stack = {}
	M.mark("U", 0)   -- 新局：渲染计数从 0 起（main.lua 的 IN.reset），标记当作"刚更新完"
	local out = {}
	for _, s in ipairs(M.pending) do out[#out + 1] = '{"k":"api","n":1,' .. s .. "}" end
	M.pending = {}
	return out
end

-- 开局时核对：用类表取到的实例元表（安装时没有玩家实例，EntityPlayer 用 getmetatable(EntityPlayer).__class）与真实实例的元表相同
function M.verify()
	local out = {}
	local function chk(cls, obj)
		if obj ~= nil then out[cls] = (getmetatable(obj) == class_mt(cls)) end
	end
	chk("Game", game)
	chk("Level", game:GetLevel())
	chk("EntityPlayer", Isaac.GetPlayer(0))
	chk("SFXManager", SFXManager())
	M.verified = out
	return out
end

function M.header()
	return {version = M.VERSION, installed = M.installed, failed = M.failed, metatables_match = M.verified}
end

------------------------------------------------------------------ 回放：原样调用（replay.lua 用）
local function dec(a)
	local t = a[1]
	if t == "i" or t == "s" or t == "b" then return a[2]
	elseif t == "f" then return a[2] + 0.0
	elseif t == "n" then return nil
	elseif t == "p" then return Isaac.GetPlayer(a[2])
	elseif t == "ref" then
		if a[2] == nil then return EntityRef(nil) end
		return EntityRef(Isaac.GetPlayer(a[2]))
	elseif t == "v" then return Vector(a[2], a[3])
	end
	error("api replay: cannot decode argument type " .. tostring(t))
end

-- e = 回放表条目 {id=, m="Game.StartRoomTransition", set=1?, self=玩家编号?, a={{"i",5},...}}；返回 ok, err
function M.replay_call(e, where)
	local cls, name = e.m:match("^([%w_]+)%.([%w_]+)$")
	local args = {}
	for i, a in ipairs(e.a or {}) do args[i] = dec(a) end
	local n = #(e.a or {})
	local obj
	if cls == "Game" then obj = game
	elseif cls == "Level" then obj = game:GetLevel()
	elseif cls == "EntityPlayer" then obj = Isaac.GetPlayer(e.self or 0)
	elseif cls == "SFXManager" then obj = SFXManager()
	elseif cls == "Isaac" then obj = nil
	else return false, "unknown class " .. tostring(cls) end
	-- 回放局里同样写一条 api 行（rp = 录制的 id，where = 实际执行点），compare 可对照两边的调用流
	local R = ctx.run()
	if R then
		local ph, r, af, rc = phase(R)
		ctx.line('{"k":"api","n":' .. (R.n + 1) .. ',"gf":' .. game:GetFrameCount() .. ',"ph":"' .. ph .. '"'
			.. (r and (',"r":' .. r) or "") .. (af == 1 and ',"af":1' or "") .. (rc == 1 and ',"rc":1' or "")
			.. ',"id":' .. e.id .. ',"m":' .. ctx.jstr(e.m) .. (e.set and ',"set":1' or "")
			.. (e.self and (',"self":{"p":' .. e.self .. "}") or "") .. ',"src":' .. ctx.jstr(e.src or "?")
			.. (e.parent and (',"in":' .. e.parent) or "") .. ',"rp":1,"where":' .. ctx.jstr(where) .. "}")
	end
	M.internal = M.internal + 1
	M.stack[#M.stack + 1] = e.id
	local ok, err = pcall(function()
		if cls == "Isaac" then
			Isaac[name](table.unpack(args, 1, n))
		elseif e.set then
			obj[name] = args[1]
		else
			obj[name](obj, table.unpack(args, 1, n))
		end
	end)
	M.stack[#M.stack] = nil
	M.internal = M.internal - 1
	return ok, err
end

-- 回放执行期间（replay.lua 执行某个录制调用时）当前的父调用 id（嵌套的子调用在 MC_POST_NEW_ROOM 等回调里执行）
function M.current_parent()
	return M.stack[#M.stack]
end

return M
