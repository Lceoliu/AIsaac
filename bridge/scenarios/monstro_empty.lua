-- Original-engine arena: fixed-seed starting room, Isaac, no collectibles.
-- Loaded explicitly through the bridge's existing console command transport.
local game = Game()
local room = game:GetRoom()
local player = Isaac.GetPlayer(0)
assert(game:GetNumPlayers() == 1, "Monstro arena requires one player")
assert(player:GetPlayerType() == PlayerType.PLAYER_ISAAC, "Expected Isaac")
assert(room:GetRoomShape() == RoomShape.ROOMSHAPE_1x1, "Expected a 1x1 starting room")
assert(player:GetCollectibleCount() == 0, "Expected no collectibles")

-- Do not silently substitute another map if the requested seed changes layout.
for i = 0, room:GetGridSize() - 1 do
    local grid = room:GetGridEntity(i)
    if grid and grid.CollisionClass ~= GridCollisionClass.COLLISION_NONE then
        local kind = grid:GetType()
        assert(kind == GridEntityType.GRID_WALL or kind == GridEntityType.GRID_DOOR,
               "Starting room has an interior obstacle")
    end
end
for _, entity in ipairs(Isaac.GetRoomEntities()) do
    if entity.Type ~= EntityType.ENTITY_PLAYER then entity:Remove() end
end
-- Rewind can restore a later player snapshot (observed: 2/6 red HP and 0 bombs).
-- Reapply the initial Isaac resource template at the episode boundary only.
player:AddHearts(player:GetMaxHearts() - player:GetHearts())
player:AddBombs(1 - player:GetNumBombs())
player:AddKeys(-player:GetNumKeys())
player:AddCoins(-player:GetNumCoins())
player.Position = Vector(320, 380)
player.Velocity = Vector(0, 0)
local boss = game:Spawn(EntityType.ENTITY_MONSTRO, 0, Vector(320, 220),
                       Vector(0, 0), nil, 0, room:GetSpawnSeed())
room:SetClear(false)
for slot = 0, 7 do
    local door = room:GetDoor(slot)
    if door then
        door:Close(true)
        door:Bar() -- ordinary closed doors can be bombed open; arena exits must stay blocked
    end
end
Isaac.DebugString("[IsaacRLScenario] monstro_empty ready; player=0 collectibles=0 spawn_seed="
                  .. tostring(boss.InitSeed) .. " hp=" .. tostring(boss.HitPoints)
                  .. " player_hearts=" .. tostring(player:GetHearts()) .. " bombs=" .. tostring(player:GetNumBombs()))
