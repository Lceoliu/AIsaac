-- 测试机器人（仅 harness 模式加载）：只读游戏状态，算出想按的键（字母）由 Python 以真实键盘消息发送。
-- 不改游戏状态、不用游戏 RNG（自带 LCG）。与第一阶段 probes/mod_rr 的机器人逻辑相同。
-- 字母：l r u d = 移动；L R U D = 射击；b = 炸弹；i = 主动；p = 卡/药；x = 丢弃
local M = {}
local game = Game()

local bot = {}
function M.reset(seed)
	bot = {visited = {}, ignore = {}, stuck = 0, last = nil, wander = 0, wdir = nil, strafe = 1, tick = 0, door = nil,
		prev_room = nil, rng_state = seed or 12345, room_frames = 0}
end
M.reset()

local function brand(n)
	bot.rng_state = (bot.rng_state * 1103515245 + 12345) % 2147483648
	return bot.rng_state % n
end

local function bfs_next(room, from_idx, goal_idx)
	local w = room:GetGridWidth(); local n = room:GetGridSize()
	local prev = {[from_idx] = from_idx}; local q = {from_idx}; local h = 1
	while h <= #q do
		local c = q[h]; h = h + 1
		if c == goal_idx then break end
		local cx = c % w
		for _, d in ipairs({-1, 1, -w, w}) do
			local nb = c + d
			if nb >= 0 and nb < n and not prev[nb] and not (d == -1 and cx == 0) and not (d == 1 and cx == w - 1) then
				local col = room:GetGridCollision(nb)
				if col == GridCollisionClass.COLLISION_NONE or nb == goal_idx then prev[nb] = c; q[#q + 1] = nb end
			end
		end
	end
	if not prev[goal_idx] then return nil end
	local c = goal_idx
	while prev[c] ~= from_idx and prev[c] ~= c do c = prev[c] end
	return c
end

local function dir_letters(dx, dy, dead)
	local s = ""
	if dx < -dead then s = s .. "l" elseif dx > dead then s = s .. "r" end
	if dy < -dead then s = s .. "u" elseif dy > dead then s = s .. "d" end
	return s
end

function M.keys()
	local p = Isaac.GetPlayer(0); local room = game:GetRoom(); local level = game:GetLevel()
	local pos = p.Position; local keys = ""
	local ridx = level:GetCurrentRoomIndex()
	if ridx ~= bot.prev_room then
		bot.prev_room = ridx; bot.room_frames = 0; bot.door = nil; bot.best = nil; bot.since = 0; bot.visited[ridx] = true
	end
	bot.room_frames = bot.room_frames + 1; bot.tick = bot.tick + 1
	local enemy, ed = nil, 1e9
	local target, pickup_d = nil, 1e9
	bot.target_seed = nil
	for _, e in ipairs(Isaac.GetRoomEntities()) do
		if e:IsVulnerableEnemy() and e:IsActiveEnemy(false) and not e:IsDead() then
			local d = pos:Distance(e.Position); if d < ed then ed, enemy = d, e end
		elseif e.Type == EntityType.ENTITY_PICKUP then
			local pk = e:ToPickup()
			local want = pk.Price == 0 and not (e.Variant == 100 and e.SubType == 0)
				and e.Variant ~= 340 and e.Variant ~= 370 and e.Variant ~= 380
			if want and e.Variant == 10 and p:GetHearts() >= p:GetMaxHearts() then want = false end
			if want and bot.ignore[e.InitSeed] then want = false end
			if want then
				local d = pos:Distance(e.Position) - (e.Variant == 100 and 1000 or 0)
				if d < pickup_d then pickup_d, target, bot.target_seed = d, e.Position, e.InitSeed end
			end
		end
	end
	local goal = nil
	if enemy then
		local v = enemy.Position - pos
		if math.abs(v.X) > math.abs(v.Y) then keys = keys .. (v.X < 0 and "L" or "R") else keys = keys .. (v.Y < 0 and "U" or "D") end
		if bot.tick % 50 == 0 then bot.strafe = -bot.strafe end
		if ed < 110 then keys = keys .. dir_letters(-v.X, -v.Y, 2)
		elseif ed > 220 then goal = enemy.Position
		else keys = keys .. dir_letters(-v.Y * bot.strafe, v.X * bot.strafe, 2) end
		if ed < 90 and p:GetNumBombs() > 0 and bot.tick % 90 < 3 then keys = keys .. "b" end
		if p:GetActiveItem() ~= 0 and not p:NeedsCharge() and bot.tick % 40 < 3 then keys = keys .. "i" end
	elseif target then
		goal = target
	else
		if (p:GetCard(0) ~= 0 or p:GetPill(0) ~= 0) and bot.room_frames > 20 and bot.tick % 30 < 3 then keys = keys .. "p" end
		if bot.room_frames == 45 and p:GetNumBombs() > 0 and bot.tick % 2 == 0 then keys = keys .. "b" end
		for i = 0, room:GetGridSize() - 1 do
			local ge = room:GetGridEntity(i)
			if ge and ge:GetType() == GridEntityType.GRID_TRAPDOOR and ge.State == 1 then goal = ge.Position end
		end
		if not goal then
			if not bot.door then
				local open, fresh = {}, {}
				for slot = 0, 7 do
					local d = room:GetDoor(slot)
					if d and d:IsOpen() then
						open[#open + 1] = d
						if not bot.visited[d.TargetRoomIndex] then fresh[#fresh + 1] = d end
					end
				end
				local pool = #fresh > 0 and fresh or open
				if #pool > 0 then bot.door = pool[brand(#pool) + 1].Position end
			end
			goal = bot.door
		end
	end
	if goal then
		local from = room:GetGridIndex(pos); local to = room:GetGridIndex(goal)
		local nxt = (from >= 0 and to >= 0 and from ~= to) and bfs_next(room, from, to) or nil
		local wp = nxt and room:GetGridPosition(nxt) or goal
		if nxt == to then wp = goal end
		keys = keys .. dir_letters(wp.X - pos.X, wp.Y - pos.Y, 3)
	end
	if bot.last and pos:Distance(bot.last) < 0.5 and keys:find("[lrud]") then bot.stuck = bot.stuck + 1 else bot.stuck = 0 end
	bot.last = Vector(pos.X, pos.Y)
	if goal and not enemy then
		local d = pos:Distance(goal)
		if not bot.best or d < bot.best - 8 then bot.best = d; bot.since = 0 else bot.since = (bot.since or 0) + 1 end
		if bot.since > 60 then
			bot.stuck = 31; bot.since = 0; bot.best = nil
			if target and bot.target_seed then bot.ignore[bot.target_seed] = true end
		end
	else bot.best = nil; bot.since = 0 end
	if bot.stuck > 30 then bot.wander = 20; bot.wdir = ({"l", "r", "u", "d", "lu", "rd"})[brand(6) + 1]; bot.stuck = 0; bot.door = nil end
	if bot.wander > 0 then bot.wander = bot.wander - 1; keys = keys:gsub("[lrud]", "") .. bot.wdir end
	return keys
end

return M
