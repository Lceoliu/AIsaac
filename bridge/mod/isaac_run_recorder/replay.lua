-- 回放模式（仅回归测试用）：由 Python（replay_run.py）放置 <IRR_DIR>/irr_replay.lua 触发。
-- 该文件返回一个表：
--   start  = "seed XXXX XXXX"             开局命令（由 harness 的 start 命令执行）
--   frames = N                             回放帧数（到达后结束本局记录）
--   ticks  = {[n] = {[kind] = {[pnum] = {36 个值}}}}   只含输入发生变化的 tick；值下标 = hook*12+action+1
--   spawns = {["gf:t.v.s:occ"] = seed}     生成种子（按 帧号、类型、帧内第几次 钉回）
--   cmds   = {[n] = {"命令", ...}}        录制时 harness 在第 n 帧执行过的命令
--   mt     = {POST_PLAYER_INIT = salt, POST_GAME_STARTED = salt}   录制 header.mt.reseeds 里各时机的 salt
--            （main.lua 在同一时机用 StartSeed xor salt reseed 全局 MT；录制里没有的时机不 reseed；nil = 录制没有 MT）
--   pauses = {[n] = {{open = "i2", open_action = 15, unpause_rs = 2079, close_after = 10}, ...}}   菜单暂停（见下）；
--            {engine = true, unpause_rs = U} = 引擎自己的暂停（不注入，只做解除暂停后的 tick 对齐）
--   source = "原录制文件路径"
--   start_players = {[num] = {x100 = X, y100 = Y}, ...}   只在续局：录制第 1 帧各玩家位置（定点 x100）。续局的出生位置由 reseed
--            之前的随机数决定（phase7-continue），main.lua 在续局的 MC_POST_GAME_STARTED 里钉回（R.apply_start_pos）
--   mouse  = {[n] = {{r, sx100, sy100, buttons}, ...}}   录制的鼠标流（ms，屏幕坐标 x100 与按键位），mouse_init = {sx100, sy100, buttons}
--            （phase8-mouse：回放按"第 r-1 次渲染之后"把光标还原给引擎，见下面"鼠标"一节；--no-mouse 时没有这两个字段）
-- 注入覆盖玩家实体的 12 个动作；全局查询只在菜单暂停的开菜单 tick 注入一次（R.global）。
local M = {}

function M.load(path)
	local chunk, err = loadfile(path)
	if not chunk then return nil, err end
	local ok, tbl = pcall(chunk)
	if not ok then return nil, tbl end
	local R = {tbl = tbl, cur = {}, occ_gf = -1, occ = {}, pinned = 0, unpinned = 0, active = false,
		folded = 0, fold_frame = -1, fold_done = {}, pause_opens = 0, pause_unpauses = 0, remapped = 0, close_requests = 0,
		engine_pauses = 0, start_pos_pinned = 0, start_pos_moved = 0, pinstat = {}, pinlog = {}}

	-- tick 顺序：一帧的行里先是上次更新之后的中间 tick i1, i2, ...，最后是本帧的 "u"。
	local function kind_index(kind)
		if kind == "u" then return math.huge end
		return tonumber(kind:match("^i(%d+)"))
	end

	------------------------------------------------------------------ 菜单暂停（phase5-pause/REPORT.md）
	-- 录制里真人在第 n 帧的 tick open 按 Esc（TRIGGERED MENUBACK，实体 nil）打开暂停菜单：这一帧的 tick 序列 =
	-- 开菜单前的中间 tick + 暂停期间每个渲染一个 tick（玩家输入冻结、没有变化）+ 解除暂停后 2 次轮询（i<U>、i<U+1>）+ 更新 tick。
	-- 开菜单落在 i2（本应同时做逻辑更新的那次轮询）时这一帧玩家多走一个 60 Hz 子步；回放不暂停就没有这些 tick，从这一帧起分歧。
	-- 回放的做法：在同一 (帧, tick) 对 TRIGGERED MENUBACK 返回 true 打开菜单（菜单里的按键不经过 MC_INPUT_ACTION，Lua 关不掉），
	-- 暂停 close_after 个渲染后由 Python 发真实 Esc 关闭（main.lua 写 pausectl.txt 请求）；回放的暂停长度因此与录制不同，
	-- 解除暂停后的 tick 按"相对解除暂停"对齐：录制序号 j >= U_rec 的变化在回放序号 j - U_rec + U_rep 应用。
	local pstate = {}   -- [n] = {[i] = {opened, unpaused, u_rep}}
	local function plan_state(n, i)
		local s = pstate[n]
		if not s then s = {}; pstate[n] = s end
		local st = s[i]
		if not st then st = {}; s[i] = st end
		return st
	end

	-- 录制序号 j -> 回放里应在哪个序号应用（只对有菜单暂停的帧做变换）
	local function eff_index(n, j)
		local list = tbl.pauses and tbl.pauses[n]
		if not list or j == math.huge then return j end
		for i, pl in ipairs(list) do
			local st = plan_state(n, i)
			if st.unpaused then
				if j >= pl.unpause_rs then return j - pl.unpause_rs + st.u_rep end
				return math.min(j, st.u_rep)
			elseif st.opened then
				if j >= pl.unpause_rs then return 1e9 end  -- 解除暂停之前不应用（u = math.huge 时照样全部应用）
			end
		end
		return j
	end

	-- 回放进程的 tick 序列不必与录制一致（渲染节奏、菜单暂停长度），所以不是只查"当帧当 kind"，
	-- 而是把本帧所有（变换后）序号 <= 当前 tick、尚未应用的变化按录制顺序折叠进来。
	function R.on_tick(n, kind)
		local fr = tbl.ticks[n]
		if not fr then return end
		if R.fold_frame ~= n then R.fold_frame = n; R.fold_done = {} end
		local upto = kind_index(kind)
		if not upto then return end
		local ks = {}
		for k in pairs(fr) do
			if not R.fold_done[k] then
				local idx = kind_index(k)
				if idx then
					local e = eff_index(n, idx)
					if e <= upto then ks[#ks + 1] = {idx, k, e ~= idx} end
				end
			end
		end
		if #ks == 0 then return end
		table.sort(ks, function(a, b) return a[1] < b[1] end)
		for _, e in ipairs(ks) do
			for pnum, arr in pairs(fr[e[2]]) do R.cur[pnum] = arr end
			R.fold_done[e[2]] = true
			if e[3] then R.remapped = R.remapped + 1
			elseif e[2] ~= kind then R.folded = R.folded + 1 end
		end
	end

	function R.value(pnum, hook, action)
		local arr = R.cur[pnum]
		if not arr then return nil end
		return arr[hook * 12 + action + 1]
	end

	-- 全局查询（实体 nil）的注入：只在菜单暂停的开菜单 tick 对 TRIGGERED <open_action> 返回 true
	function R.global(n, kind, hook, action)
		local list = tbl.pauses and tbl.pauses[n]
		if not list or hook ~= 1 then return nil end
		for i, pl in ipairs(list) do
			if not pl.engine then
				local st = plan_state(n, i)
				if not st.opened then
					if action == pl.open_action and kind == pl.open then
						st.opened = true
						st.open_rs = nil
						R.pause_opens = R.pause_opens + 1
						return true
					end
					return nil
				end
			end
		end
		return nil
	end

	-- main.lua 在 MC_POST_RENDER 里 IsPaused() 翻转时调用（rs = 本逻辑帧窗口内第几次渲染）
	function R.on_pause_flip(n, paused, rs)
		local list = tbl.pauses and tbl.pauses[n]
		if not list then return end
		for i, pl in ipairs(list) do
			local st = plan_state(n, i)
			-- 引擎自己的暂停（换房、弹窗、换层）：回放里自己发生，只记下回放的暂停起止，用于解除暂停后的 tick 对齐
			if pl.engine and not st.opened and paused then
				st.opened = true
				st.open_rs = rs
				R.engine_pauses = R.engine_pauses + 1
				return
			end
			if st.opened and not st.unpaused then
				if paused then
					if not st.open_rs then st.open_rs = rs end
				else
					st.unpaused = true
					st.u_rep = rs
					R.pause_unpauses = R.pause_unpauses + 1
				end
				return
			end
		end
	end

	-- 每个渲染帧：若本帧有已打开、未解除的注入暂停且已暂停够 close_after 个渲染，返回请求标识（Python 据此发 Esc）
	function R.close_request(n, rs)
		local list = tbl.pauses and tbl.pauses[n]
		if not list then return nil end
		for i, pl in ipairs(list) do
			local st = plan_state(n, i)
			if not pl.engine and st.opened and not st.unpaused and st.open_rs and rs - st.open_rs >= (pl.close_after or 6) then
				if not st.requested then st.requested = true; R.close_requests = R.close_requests + 1 end
				return n .. ":" .. i
			end
		end
		return nil
	end

	-- 实验（replay_run --inject-extras）：录制里非玩家 12 动作的非零查询（tick 的 extra）按 (帧, tick, hook, action, who) 返回录制值
	function R.extra(n, kind, hook, action, who)
		local ex = tbl.extras and tbl.extras[n]
		local lst = ex and ex[kind]
		if not lst then return nil end
		for _, e in ipairs(lst) do
			if e[1] == hook and e[2] == action and e[3] == who then
				R.extras_injected = (R.extras_injected or 0) + 1
				if hook == 2 then return e[4] end
				return e[4] ~= 0
			end
		end
		return nil
	end

	------------------------------------------------------------------ 鼠标（phase8-mouse/FINDINGS.md）
	-- 引擎每次 Manager::Update 末尾比较光标位置：变了就把"光标闲置计数"置 300（之后每次 -1，>0 时画游戏内光标），这会改变
	-- 后续的游戏结果（实测：human2b 第 6686 帧真人把鼠标挪了 1 像素，回放不挪则第 6911 帧起分歧）。所以回放按录制的 ms 流
	-- 把光标位置/按键还原给引擎（rr_mt.dll 的 RRMT_SetMouse，直接写 GLFW 的虚拟光标；真实鼠标消息被 DLL 丢掉）。
	-- 对应规则：录制第 n 帧行里的 ms 条目 [r, sx, sy, gx, gy, b] 是"上一次 MC_POST_UPDATE 之后第 r 次渲染"时读到的值，
	-- 它在第 r 次迭代的消息泵里生效（每次迭代 = Manager::Update（泵消息 → 可能的逻辑更新 → 光标计数）+ 1 次渲染），
	-- 所以回放在"第 r-1 次渲染的 MC_POST_RENDER（采样之后）"设置；r = 1 在上一帧的 MC_POST_UPDATE 设置（同一次迭代的
	-- 光标计数在逻辑更新之后）。有菜单/引擎暂停对齐的帧，序号 r-1 按 tick 的同一映射（eff_index）换算；本帧渲染不够
	-- （回放的暂停比录制短）而没来得及设置的条目在本帧 MC_POST_UPDATE 一次补上（folded）。
	-- tbl.mouse = {[n] = {{r, sx100, sy100, b}, ...}}；tbl.mouse_init = {sx100, sy100, b}（录制第 1 帧的第一条 = 开局时的光标）。
	R.mouse_stats = {available = false, reason = "not_started", init = 0, applied = 0, folded = 0, failed = 0}
	R.mouse_done = {}   -- [n] = 已设置的条目数（按顺序）
	local mouse_dll = nil

	local function mouse_set(e, mode)
		local ok, r = pcall(mouse_dll.mouse_set, e[2] / 100, e[3] / 100, e[4] or 0, mode)
		if ok and r then return true end
		R.mouse_stats.failed = R.mouse_stats.failed + 1
		return false
	end

	-- 回放局开局（MC_POST_GAME_STARTED）：mt 模块（mt.lua）可用时启用；初始位置无声设置（不重置闲置计数、计数清 0）
	function R.mouse_start(mt)
		R.mouse_done = {}
		if not tbl.mouse then R.mouse_stats.reason = "table has no mouse stream"; return end
		if not (mt and mt.mouse_init and mt.mouse_init()) then
			R.mouse_stats.reason = (mt and mt.mouse_reason) or "mt module missing"
			return
		end
		mouse_dll = mt
		R.mouse_stats.available = true
		R.mouse_stats.reason = "ok"
		local m0 = tbl.mouse_init
		if m0 and mouse_set({0, m0[1], m0[2], m0[3]}, 2) then R.mouse_stats.init = 1 end
	end

	-- 设置第 n 帧里录制序号（映射后）<= upto 的未设置条目；fold = true 时全部设置（本帧已结束）
	local function mouse_apply(n, upto, fold)
		if not mouse_dll then return end
		local list = tbl.mouse[n]
		if not list then return end
		local k = R.mouse_done[n] or 0
		while k < #list do
			local e = list[k + 1]
			if not fold then
				local j = eff_index(n, e[1] - 1)
				if j > upto then break end
			end
			mouse_set(e, 0)
			k = k + 1
			if fold then R.mouse_stats.folded = R.mouse_stats.folded + 1 else R.mouse_stats.applied = R.mouse_stats.applied + 1 end
		end
		R.mouse_done[n] = k
	end

	-- main.lua：MC_POST_UPDATE 写完第 n 帧之后（补上第 n 帧剩下的，再设第 n+1 帧 r = 1 的）
	function R.mouse_on_update(n)
		if not mouse_dll then return end
		mouse_apply(n, 0, true)
		R.mouse_done[n] = nil
		mouse_apply(n + 1, 0, false)
	end

	-- main.lua：MC_POST_RENDER 采样之后（n = 下一个逻辑帧号，rs = 本帧窗口内第几次渲染）
	function R.mouse_on_render(n, rs)
		if mouse_dll then mouse_apply(n, rs, false) end
	end

	-- MC_PRE_ENTITY_SPAWN：返回 {t, v, s, seed} 或 nil
	function R.pin(gf, t, v, s, seed)
		if gf ~= R.occ_gf then R.occ_gf = gf; R.occ = {} end
		local tk = gf .. ":" .. t .. "." .. v .. "." .. s
		local k = (R.occ[tk] or 0) + 1
		R.occ[tk] = k
		local want = tbl.spawns[tk .. ":" .. k]
		-- 诊断：按类型统计 钉回/本来就相同/表里没有，前 400 次钉回记下 (gf, 键, 原种子, 录制种子)
		local ty = t .. "." .. v .. "." .. s
		local st = R.pinstat[ty]
		if not st then st = {0, 0, 0}; R.pinstat[ty] = st end
		if want == nil then R.unpinned = R.unpinned + 1; st[3] = st[3] + 1; return nil end
		if want ~= seed then
			R.pinned = R.pinned + 1; st[1] = st[1] + 1
			if #R.pinlog < 400 then R.pinlog[#R.pinlog + 1] = "[" .. gf .. ',"' .. tk .. ":" .. k .. '",' .. seed .. "," .. want .. "]" end
			return {t, v, s, want}
		end
		st[2] = st[2] + 1
		return nil
	end

	-- 续局回放：把每个玩家放到录制第 1 帧的位置、速度清零（只在 tbl.start_players 存在时；非续局的回放表没有这个字段）。
	-- 返回钉了几个玩家；start_pos_moved = 其中原位置与录制不同的个数
	function R.apply_start_pos()
		local sp = tbl.start_players
		if not sp then return 0 end
		local k = 0
		for i = 0, Game():GetNumPlayers() - 1 do
			local want = sp[i]
			if want then
				local p = Isaac.GetPlayer(i)
				local x, y = want.x100 / 100, want.y100 / 100
				local old = p.Position
				if math.floor(old.X * 100 + 0.5) ~= want.x100 or math.floor(old.Y * 100 + 0.5) ~= want.y100 then
					R.start_pos_moved = R.start_pos_moved + 1
				end
				p.Position = Vector(x, y)
				p.Velocity = Vector(0, 0)
				k = k + 1
			end
		end
		R.start_pos_pinned = R.start_pos_pinned + k
		return k
	end

	function R.stats_json()
		return '{"pinned":' .. R.pinned .. ',"unpinned":' .. R.unpinned .. ',"folded":' .. R.folded
			.. ',"remapped":' .. R.remapped .. ',"pause_opens":' .. R.pause_opens .. ',"close_requests":' .. R.close_requests
			.. ',"pause_unpauses":' .. R.pause_unpauses .. ',"engine_pauses":' .. R.engine_pauses
			.. ',"start_pos_pinned":' .. R.start_pos_pinned .. ',"start_pos_moved":' .. R.start_pos_moved
			.. (R.extras_injected and (',"extras_injected":' .. R.extras_injected) or '')
			.. ',"mouse":{"available":' .. tostring(R.mouse_stats.available) .. ',"reason":"' .. tostring(R.mouse_stats.reason):gsub('[%c"\\]', "?")
			.. '","init":' .. R.mouse_stats.init .. ',"applied":' .. R.mouse_stats.applied .. ',"folded":' .. R.mouse_stats.folded
			.. ',"failed":' .. R.mouse_stats.failed .. '}}'
	end

	function R.cmds_at(n) return tbl.cmds and tbl.cmds[n] end

	-- 诊断输出（main.lua 在回放局结束时写 irr_pinlog.json）：{"stat":{"t.v.s":[钉回, 相同, 表里没有]}, "log":[...]}
	function R.pinlog_json()
		local keys = {}
		for k in pairs(R.pinstat) do keys[#keys + 1] = k end
		table.sort(keys)
		local b = {}
		for _, k in ipairs(keys) do local st = R.pinstat[k]; b[#b + 1] = '"' .. k .. '":[' .. st[1] .. "," .. st[2] .. "," .. st[3] .. "]" end
		return '{"stat":{' .. table.concat(b, ",") .. '},"log":[' .. table.concat(R.pinlog, ",") .. "]}"
	end

	-- 同一进程里再开一个回放局（观战模式的进程内倒带，watch.lua）：清掉上一个回放局的全部可变状态，回放表本身不变
	function R.reset()
		R.cur, R.occ_gf, R.occ = {}, -1, {}
		R.pinned, R.unpinned, R.folded, R.fold_frame, R.fold_done = 0, 0, 0, -1, {}
		R.pause_opens, R.pause_unpauses, R.remapped, R.close_requests, R.engine_pauses = 0, 0, 0, 0, 0
		R.start_pos_pinned, R.start_pos_moved = 0, 0
		R.pinstat, R.pinlog = {}, {}
		R.mouse_stats = {available = false, reason = "not_started", init = 0, applied = 0, folded = 0, failed = 0}
		R.mouse_done = {}
		mouse_dll = nil
		pstate = {}
	end
	return R
end

return M
