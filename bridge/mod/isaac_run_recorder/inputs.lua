-- 60 Hz tick 输入采集（INVESTIGATION §3 / DESIGN_BRIEF D3）。
-- 游戏每次 60 Hz 迭代都会对玩家实体查询一整组输入；有逻辑更新的迭代（"u"）里帧计数先 +1 再查询，
-- 没有逻辑更新的迭代是中间 tick（"i<j>"，j = 上次 MC_POST_UPDATE 之后的渲染次数）。
-- 记录模式：只读原生值（Input.*），回调一律返回 nil；用 reading 标志位跳过我们自己的 Input.* 读取。
-- 回放模式：玩家实体的 12 个动作返回记录值，并把返回值（引擎实际消费的值）记进回放文件。
local M = {}
local game = Game()
local concat, tostring = table.concat, tostring
local Input = Input
local H_PRESSED, H_TRIGGERED, H_VALUE = 0, 1, 2

local T = {lastUgf = -1, rs = 0, slot = nil, ticks = {}, state = {}, pend = nil, pend_end = nil}
M.reading = false
M.replay = nil       -- 回放对象（replay.lua），nil = 记录模式
M.n_next = 1         -- 当前 tick 所属的帧序号（本局第几个逻辑帧，从 1 起）
M.player_num = nil   -- function(EntityPlayer) -> 玩家编号（由 main 注入 state.player_num）
M.watch = false      -- 观战模式（watch.lua）：回放时非玩家 12 动作的查询也不读真实输入（见 on_query 末尾）

function M.reset(gf)
	T = {lastUgf = gf, rs = 0, slot = nil, ticks = {}, state = {}, pend = nil, pend_end = nil}
end

function M.on_render() T.rs = T.rs + 1 end
-- 上次 MC_POST_UPDATE 之后的渲染次数（pause/unpause 事件的第三个参数；之后的中间 tick 序号 = 它）
function M.renders_since_update() return T.rs end
function M.on_update(gf) T.lastUgf = gf; T.rs = 0 end

local function num(v)
	if v == nil then return "null" end
	if v == true then return "1" elseif v == false then return "0" end
	if v == 0 then return "0" elseif v == 1 then return "1" end
	if v ~= v then return "null" end
	return ("%.9g"):format(v)
end

local function flush_pend()
	if T.pend then
		if T.pend == T.pend_end then T.ticks[#T.ticks + 1] = '"i' .. T.pend .. '"'
		else T.ticks[#T.ticks + 1] = '"i' .. T.pend .. "-" .. T.pend_end .. '"' end
		T.pend, T.pend_end = nil, nil
	end
end

-- 关闭当前 tick。每个玩家维护 36 个值（12 动作 × value/pressed/triggered）的"当前状态"：本 tick 查询到的值
-- 覆盖进去，没被查询的沿用上一 tick（从未被查询过的为 null）。状态与上一 tick 不同才写出完整的 12 个三元组。
local function close_slot()
	local slot = T.slot
	if not slot then return end
	T.slot = nil
	local changed = nil
	local pnums = {}
	for pnum in pairs(slot.vals) do pnums[#pnums + 1] = pnum end
	table.sort(pnums)
	for _, pnum in ipairs(pnums) do
		local v = slot.vals[pnum]
		local st = T.state[pnum]
		if not st then st = {}; T.state[pnum] = st end
		local diff = false
		for i = 1, 36 do
			local x = v[i]
			if x ~= nil and st[i] ~= x then st[i] = x; diff = true end
		end
		if diff then
			local cells = {}
			for a = 0, 11 do
				cells[a + 1] = "[" .. num(st[H_VALUE * 12 + a + 1]) .. "," .. num(st[H_PRESSED * 12 + a + 1]) .. ","
					.. num(st[H_TRIGGERED * 12 + a + 1]) .. "]"
			end
			changed = changed or {}
			changed[#changed + 1] = "[" .. pnum .. ",[" .. concat(cells, ",") .. "]]"
		end
	end
	local kind = slot.kind
	if not changed and not slot.extra then
		local j = tonumber(kind:sub(2))
		if j and T.pend and T.pend_end == j - 1 then
			T.pend_end = j
		else
			flush_pend()
			if j then T.pend, T.pend_end = j, j else T.ticks[#T.ticks + 1] = '"' .. kind .. '"' end
		end
		return
	end
	flush_pend()
	local e = '["' .. kind .. '",[' .. (changed and concat(changed, ",") or "") .. "]"
	if slot.extra then e = e .. ",[" .. concat(slot.extra, ",") .. "]" end
	T.ticks[#T.ticks + 1] = e .. "]"
end

-- 帧结束（MC_POST_UPDATE）时取走本帧的 tick 列表（JSON 数组内容）
function M.take_ticks()
	close_slot()
	flush_pend()
	local s = concat(T.ticks, ",")
	T.ticks = {}
	return s
end

local function read_native(hook, action, ctrl)
	M.reading = true
	local ok, v
	if hook == H_VALUE then ok, v = pcall(Input.GetActionValue, action, ctrl)
	elseif hook == H_PRESSED then ok, v = pcall(Input.IsActionPressed, action, ctrl)
	else ok, v = pcall(Input.IsActionTriggered, action, ctrl) end
	M.reading = false
	if ok then return v end
	return nil
end

-- MC_INPUT_ACTION 主体；返回值即回调返回值（记录模式恒为 nil）
function M.on_query(entity, hook, action)
	local kind = (game:GetFrameCount() > T.lastUgf) and "u" or ("i" .. T.rs)
	local slot = T.slot
	if not slot or slot.kind ~= kind then
		close_slot()
		slot = {kind = kind, vals = {}, extra = nil, seen = {}}
		T.slot = slot
		if M.replay then M.replay.on_tick(M.n_next, kind) end
	end
	local p = entity and entity:ToPlayer()
	if p and action <= 11 then
		local pnum = M.player_num(p)
		local vals = slot.vals[pnum]
		if not vals then vals = {}; slot.vals[pnum] = vals end
		local idx = hook * 12 + action + 1
		if M.replay then
			local v = M.replay.value(pnum, hook, action)
			local ret
			if hook == H_VALUE then ret = v or 0.0 else ret = (v == 1 or v == true) end
			if vals[idx] == nil then vals[idx] = ret end
			return ret
		end
		if vals[idx] == nil then
			local v = read_native(hook, action, p.ControllerIndex)
			if v == nil then v = false end
			vals[idx] = v
		end
		return nil
	end
	-- 其它查询（实体为 nil 的全局查询、非玩家实体、action >= 12）：只记非零值
	local who = -1
	if p then who = M.player_num(p) elseif entity then who = -3 end
	local key = hook * 64 + action + (who + 3) * 4096
	-- 观战隔离：回放时观众的按键（Esc 开菜单、R 重开、F 全屏、M 静音、Tab 地图……）一律当作"没按"，与无人值守的回放进程相同
	local isolate = M.watch and M.replay ~= nil
	if slot.seen[key] then
		if slot.ret then
			local r = slot.ret[key]
			if r ~= nil then return r end
		end
		if isolate then return (hook == H_VALUE) and 0.0 or false end
		return nil
	end
	slot.seen[key] = true
	-- 回放模式：实体为 nil 的全局查询可由回放对象覆盖（菜单暂停的开菜单，见 replay.lua R.global）；
	-- 覆盖值同样记进 extra（回放文件里看得到注入了什么），同一 tick 的重复查询返回同一值
	local ov = nil
	if M.replay and M.replay.global and who == -1 then ov = M.replay.global(M.n_next, kind, hook, action) end
	if ov == nil and M.replay and M.replay.extra then ov = M.replay.extra(M.n_next, kind, hook, action, who) end
	local v
	if ov ~= nil then
		v = ov
		slot.ret = slot.ret or {}
		slot.ret[key] = ov
	elseif isolate then
		v = (hook == H_VALUE) and 0.0 or false
		return v
	else
		v = read_native(hook, action, p and p.ControllerIndex or 0)
	end
	if v and v ~= 0 then
		slot.extra = slot.extra or {}
		slot.extra[#slot.extra + 1] = "[" .. hook .. "," .. action .. "," .. who .. "," .. num(v) .. "]"
	end
	return ov
end

-- 基准：n 次原生读取 + 存储（开销测量用，不影响记录）
function M.bench(p, n)
	local t = os.clock()
	local scratch = {}
	for i = 1, n do
		local a = i % 12
		local h = i % 3
		local v = read_native(h, a, p.ControllerIndex)
		scratch[h * 12 + a + 1] = v
	end
	return (os.clock() - t) * 1000 / n
end

return M
