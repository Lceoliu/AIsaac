-- 测试 harness（仅当 <IRR_DIR>/irr_control.json 存在时加载；人类录制会话不加载）。
-- 功能：文件 IPC（cmd-NNNNNN.json → mod，status.json 里回 ack）、按种子开局、测试机器人、
-- 在指定帧执行脚本化控制台命令（写 hcmd 行，回放时同帧重放）、帧数上限、开销基准、鼠标读取探针。
-- 这些都是测试驱动用的"非被动"操作，均被记录在 run 文件里（hcmd 行 / header.harness）。
local json = require("json")
local game = Game()

return function(ctx, cfg_text)
	local H = {ack = 0, queue = {}, pending = nil, active = nil, mouse = nil}
	local cfg = {}
	pcall(function() cfg = json.decode(cfg_text) or {} end)
	H.cfg = cfg
	local bot = nil

	-- 0.4.0：经 ctx.exec（api.lua 的 exec_raw）执行，不被第三方 mod 调用的包装记成 api 行（hcmd 行已记）
	local run_cmd = ctx.exec or Isaac.ExecuteCommand
	local function exec(command, n, tag)
		run_cmd(command)
		ctx.line('{"k":"hcmd","n":' .. (n or -1) .. ',"gf":' .. game:GetFrameCount() .. ',"tag":' .. ctx.jstr(tag or "exec")
			.. ',"cmd":' .. ctx.jstr(command) .. '}')
	end

	-- 命令：start / exec / stop 进队列在 MC_POST_UPDATE 里执行；mark / bench / mouseprobe 立即执行
	local function do_command(c)
		if c.cmd == "start" then
			-- 下一次 MC_POST_GAME_STARTED 开出的局是 harness 局（机器人/脚本命令/帧上限对它生效）
			if c.seed_num and not c.start then c.start = "seed " .. Seeds.Seed2String(c.seed_num) end
			H.pending = {start = c.start, bot = c.bot, bot_seed = c.bot_seed, frames = c.frames, cmds = c.cmds or {},
				replay = c.replay, script = c.script, pausewatch = c.pausewatch}
			if c.start then run_cmd(c.start) end
		elseif c.cmd == "exec" then
			exec(c.command, ctx.run() and ctx.run().n or -1, c.tag)
		elseif c.cmd == "stop" then
			ctx.close_run(c.reason or "stop")
		elseif c.cmd == "mark" then
			local r = ctx.run()
			ctx.line('{"k":"mark","n":' .. (r and r.n or -1) .. ',"gf":' .. game:GetFrameCount() .. ',"t":' .. Isaac.GetTime()
				.. ',"ifc":' .. Isaac.GetFrameCount() .. ',"label":' .. ctx.jstr(c.label) .. '}')
		elseif c.cmd == "bench" then
			local r = ctx.bench(c.n or 100)
			r.label = c.label
			local fh = io.open(ctx.out .. "/bench.jsonl", "a")
			if fh then fh:write(ctx.jval(r), "\n"); fh:close() end
		elseif c.cmd == "mouseprobe" then
			if c.on and not H.mouse then
				H.mouse = io.open(ctx.out .. "/mouseprobe.jsonl", "a")
				if H.mouse then H.mouse:setvbuf("full", 1 << 16) end
			elseif not c.on and H.mouse then
				H.mouse:close(); H.mouse = nil
			end
		end
	end
	local IMMEDIATE = {mark = true, bench = true, mouseprobe = true}

	function H.on_render(renders)
		-- pausewatch（暂停实验）：每个渲染帧写 pause.txt = "本局帧号 是否暂停 连续暂停渲染数 Isaac帧计数"，Python 据此按键
		local a = H.active
		if a and a.pausewatch and ctx.run() then
			if game:IsPaused() then H.prs = (H.prs or 0) + 1 else H.prs = 0 end
			H.rsu = (H.rsu or 0) + 1
			local fh = io.open(ctx.out .. "/pause.txt", "w")
			if fh then
				fh:write(ctx.run().n .. " " .. (game:IsPaused() and 1 or 0) .. " " .. H.prs .. " " .. Isaac.GetFrameCount() .. " " .. H.rsu)
				fh:close()
			end
		end
		if H.mouse then
			local ok, s = pcall(function()
				local a = Input.GetMousePosition(true)
				local b = Input.GetMousePosition(false)
				return '{"ifc":' .. Isaac.GetFrameCount() .. ',"t":' .. Isaac.GetTime()
					.. ',"b0":' .. tostring(Input.IsMouseBtnPressed(0)) .. ',"b1":' .. tostring(Input.IsMouseBtnPressed(1))
					.. ',"game_xy":[' .. ctx.fx(a.X) .. ',' .. ctx.fx(a.Y) .. '],"screen_xy":[' .. ctx.fx(b.X) .. ',' .. ctx.fx(b.Y)
					.. '],"mouse_control":' .. tostring(Options and Options.MouseControl) .. '}'
			end)
			H.mouse:write(ok and s or ('{"err":' .. ctx.jstr(s) .. '}'), "\n")
		end
		if renders % 4 ~= 0 then return end
		local want = H.ack + 1 + #H.queue
		local fh = io.open(("%s/cmd-%06d.json"):format(ctx.out, want), "r")
		if not fh then return end
		local text = fh:read("*a"); fh:close()
		local ok, c = pcall(json.decode, text)
		if not ok or type(c) ~= "table" then return end
		if IMMEDIATE[c.cmd] then
			local ok2, e = pcall(do_command, c)
			if not ok2 then ctx.fail("harness " .. tostring(c.cmd), e) end
			H.ack = c.id or want
			ctx.status()
		else
			H.queue[#H.queue + 1] = c
		end
	end

	-- 每个 MC_POST_UPDATE（无论是否在录）：执行队列里的一条命令
	function H.on_any_update()
		if #H.queue > 0 then
			local c = table.remove(H.queue, 1)
			local ok, e = pcall(do_command, c)
			if not ok then ctx.fail("harness " .. tostring(c.cmd), e) end
			H.ack = c.id or (H.ack + 1)
			ctx.status()
		end
	end

	-- irr_control.json 的 menu_run（可选）：本进程第一个开出的局（菜单 New Run / 续局）就当 harness 局，
	-- 内容同 start 命令的 {bot, bot_seed, frames, cmds, replay}（菜单开局录制、续局回放用）
	local function menu_desc()
		if cfg.menu_run and not H.menu_used then return cfg.menu_run end
		return nil
	end
	-- 下一个开出的局是否是回放局（MT reseed 要在 MC_POST_PLAYER_INIT 时就知道用哪个 salt）
	function H.next_is_replay()
		local d = H.pending or menu_desc()
		return d ~= nil and d.replay == true
	end

	-- 新局开始：若有待定的 harness 局则激活，返回写进 header 的 harness 描述
	function H.on_game_started(continued)
		if not H.pending and menu_desc() then
			H.menu_used = true
			local m = cfg.menu_run
			H.pending = {start = continued and "menu:continue" or "menu", bot = m.bot, bot_seed = m.bot_seed, frames = m.frames,
				cmds = m.cmds or {}, replay = m.replay}
		end
		if not H.pending then H.active = nil; return nil end
		H.active = H.pending
		H.pending = nil
		if H.active.bot then
			if not bot then bot = ctx.load_module("bot") end
			bot.reset(H.active.bot_seed or 12345)
		end
		return H.active
	end

	-- 录制中的每帧（帧行写完之后）
	function H.on_run_update(n)
		local a = H.active
		if not a then return end
		H.rsu = 0
		local cmds = a.cmds
		if a.replay then cmds = nil end
		if cmds then
			for _, c in ipairs(cmds) do
				if c.n == n then exec(c.command, n, "script") end
			end
		end
		if a.replay and ctx.replay_obj then
			local list = ctx.replay_obj.cmds_at(n)
			if list then for _, cmd in ipairs(list) do exec(cmd, n, "script") end end
		end
		if a.bot and bot then
			local ok, keys = pcall(bot.keys)
			if not ok then ctx.fail("bot", keys); keys = "" end
			local fh = io.open(ctx.out .. "/want.txt", "w")
			if fh then fh:write(n .. " " .. keys); fh:close() end
		end
		-- script（暂停实验）：固定按键表 {{起帧, 止帧, "字母"}, ...}，与机器人一样经 want.txt 由 Python 发真实键盘消息
		if a.script then
			local keys = ""
			for _, sc in ipairs(a.script) do
				if n >= sc[1] and n <= sc[2] then keys = keys .. sc[3] end
			end
			local fh = io.open(ctx.out .. "/want.txt", "w")
			if fh then fh:write(n .. " " .. keys); fh:close() end
		end
		if a.frames and n >= a.frames then
			H.active = nil
			ctx.close_run("frames")
		end
	end

	return H
end
