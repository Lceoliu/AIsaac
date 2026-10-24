-- Dedicated, normal-speed human recorder. This mod never starts an RL learner.
local mod = RegisterMod("Isaac Human Recorder", 1)
local json = require("json")
local root = assert(os.getenv("ISAAC_HUMAN_DIR"), "Launch with record_human.ps1")
local game = Game()
local state = {logic_frames=0, events={}, combat={}, health={}}
local observe = include("observe")(state)
local mode, error_text = "idle", ""
local command_id, episode, frame, file = 0, 0, 0, nil
local phase, phase_ticks, pending_room = nil, 0, false
local armed, start_at, source = false, 0, "human"
local queries, probing, sampling_input = {}, false, false
local anchor, boss, last_status = nil, nil, 0
local previous_obs, episode_name, begin_ms, last_ms
local last_outcome, written = "", 0
local input_total, input_active = 0, 0
local qa_motion, qa_safe = false, false
local queued_command = nil
local recover_after_death = false
local A, H = ButtonAction, InputHook
local actions = {A.ACTION_LEFT,A.ACTION_RIGHT,A.ACTION_UP,A.ACTION_DOWN,
    A.ACTION_SHOOTLEFT,A.ACTION_SHOOTRIGHT,A.ACTION_SHOOTUP,A.ACTION_SHOOTDOWN,
    A.ACTION_BOMB,A.ACTION_ITEM,A.ACTION_PILLCARD,A.ACTION_DROP}
local allowed = {}; for _,a in ipairs(actions) do allowed[a]=true end

local function reset_events()
    state.logic_frames=0
    state.events={damage=0,tears=0,npc_deaths=0,clears=0}
    state.combat={player_damage_events=0,player_damage=0,enemy_damage_events=0,enemy_damage=0,enemy_damage_fraction=0}
    state.health={players={},npcs={},deaths={}}
end
reset_events()
local function append(handle, value)
    assert(handle:write(json.encode(value), "\n"))
    assert(handle:flush())
end
local function status()
    local out=assert(io.open(root.."/status.jsonl","a"))
    append(out,{mode=mode,error=error_text,command_id=command_id,episode=episode,
        frame=frame,file=episode_name or "",outcome=last_outcome,bytes=written,
        input_queries=input_total,active_queries=input_active,
        utc=os.time(),game_frame=game:GetFrameCount(),source=source,
        hearts=game:GetNumPlayers()>0 and Isaac.GetPlayer(0):GetHearts() or 0})
    assert(out:close());last_status=Isaac.GetTime()
end
local function finish(outcome)
    if file then
        append(file,{type="end",outcome=outcome,frames=frame,source=source,utc=os.time()})
        written=assert(file:seek());assert(file:close());file=nil
    end
    last_outcome=outcome;mode="finished";armed=false;queries={}
    recover_after_death=outcome=="death"
    if game:GetNumPlayers()>0 then Isaac.GetPlayer(0).ControlsEnabled=false end
    status()
end
local function fail(message)
    error_text=tostring(message);mode="error";armed=false;phase=nil
    Isaac.DebugString("[HumanRecorder] ERROR "..error_text)
    if file then file:close();file=nil end -- no end marker: never certify a partial file
    status()
end
local function guarded(fn)
    return function(...)
        local ok,err=pcall(fn,...)
        if not ok then fail(err) end
    end
end

-- Capture the values the game actually asks for, not late keyboard polling.
-- The recursive probe bypasses this callback; return that same native value.
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION,function(_,entity,hook,action)
    if probing or mode~="recording" or not entity or not entity:ToPlayer() or not allowed[action] then return nil end
    local p=entity:ToPlayer(); if p.Index~=Isaac.GetPlayer(0).Index then return nil end
    probing=true
    local value
    if hook==H.GET_ACTION_VALUE then value=Input.GetActionValue(action,p.ControllerIndex)
    elseif hook==H.IS_ACTION_PRESSED then value=Input.IsActionPressed(action,p.ControllerIndex)
    elseif hook==H.IS_ACTION_TRIGGERED then value=Input.IsActionTriggered(action,p.ControllerIndex) end
    probing=false
    if qa_motion then
        local down=action==A.ACTION_SHOOTUP or (frame<30 and action==A.ACTION_RIGHT)
            or (frame>=30 and frame<60 and action==A.ACTION_LEFT)
            or (frame==75 and action==A.ACTION_BOMB)
        value=hook==H.GET_ACTION_VALUE and (down and 1 or 0) or down
    end
    if value~=nil and not sampling_input then
        queries[#queries+1]={hook=hook,action=action,value=value,game_frame=game:GetFrameCount(),ms=Isaac.GetTime()}
    end
    return value
end)

-- FLAG_FREEZE alone did not stop Monstro's attacks during native acceptance.
-- Skip NPC AI only outside a recording; combat always runs original AI unchanged.
mod:AddCallback(ModCallbacks.MC_PRE_NPC_UPDATE,function(_,npc)
    if mode=="ready" or mode=="countdown" or mode=="finished" or mode=="error" or mode=="closed" then
        npc.Velocity=Vector(0,0)
        return true
    end
end)

local function clean_room()
    assert(game:GetNumPlayers()==1,"Single-player only")
    local p=Isaac.GetPlayer(0);local room=game:GetRoom()
    assert(p:GetPlayerType()==PlayerType.PLAYER_ISAAC and p:GetCollectibleCount()==0,"Expected Isaac without collectibles")
    assert(room:GetRoomShape()==RoomShape.ROOMSHAPE_1x1,"Expected 1x1 room")
    for i=0,room:GetGridSize()-1 do
        local g=room:GetGridEntity(i)
        if g and g.CollisionClass~=GridCollisionClass.COLLISION_NONE then
            assert(g:GetType()==GridEntityType.GRID_WALL or g:GetType()==GridEntityType.GRID_DOOR,"Room is not empty")
        end
    end
    for _,e in ipairs(Isaac.GetRoomEntities()) do if e.Type~=EntityType.ENTITY_PLAYER then e:Remove() end end
    p:AddHearts(p:GetMaxHearts()-p:GetHearts());p:AddBombs(1-p:GetNumBombs())
    p:AddKeys(-p:GetNumKeys());p:AddCoins(-p:GetNumCoins())
    p.Position=Vector(320,380);p.Velocity=Vector(0,0);p.ControlsEnabled=false
    for i=0,7 do local d=room:GetDoor(i);if d then d:Close(true);d:Bar() end end
    room:SetClear(false)
end
local function summon()
    assert(mode=="ready" or mode=="countdown","Prepare/reset the room first")
    if boss and boss:Exists() then boss:Remove() end
    boss=game:Spawn(EntityType.ENTITY_MONSTRO,0,Vector(320,180),Vector(0,0),nil,0,game:GetRoom():GetSpawnSeed())
    boss:AddEntityFlags(EntityFlag.FLAG_FREEZE)
    game:GetRoom():SetClear(false)
    status()
end
local function prepare()
    if file then finish("manual_reset") end
    mode="preparing";error_text="";armed=false;boss=nil
    if anchor then
        phase="ready";phase_ticks=0;pending_room=true;Isaac.ExecuteCommand("rewind")
    else
        phase="seed";phase_ticks=0;pending_room=true;Isaac.ExecuteCommand("restart 0")
    end
    status()
end
local function begin()
    assert(mode=="ready","Room is not ready")
    source="human" -- automated tests live in an explicitly labelled QA session
    if os.getenv("ISAAC_HUMAN_QA")=="1" then source="qa" end
    armed=true;mode="countdown";start_at=Isaac.GetTime()+3000;status()
end
local function start_record()
    clean_room();mode="countdown";summon()
    -- New spawn: no waiting-time AI state leaks into the recorded battle.
    boss:ClearEntityFlags(EntityFlag.FLAG_FREEZE)
    reset_events();observe.update();frame=0;queries={};input_total=0;input_active=0
    episode=episode+1;episode_name=string.format("episode-%04d.jsonl",episode)
    assert(not io.open(root.."/"..episode_name,"r"),"Refusing to overwrite an episode")
    file=assert(io.open(root.."/"..episode_name,"w"))
    begin_ms=Isaac.GetTime();last_ms=begin_ms
    local p=Isaac.GetPlayer(0);p.ControlsEnabled=true
    previous_obs=observe.snapshot()
    append(file,{type="header",schema="isaac-human-v1",mod_version="0.1.0",source=source,utc=os.time(),
        seed=game:GetSeeds():GetStartSeedString(),room_seed=game:GetRoom():GetSpawnSeed(),
        scenario="Isaac-no-items-Monstro-empty",max_logic_frames=3600,logic_hz=30,
        action_ids=actions,hooks={value=H.GET_ACTION_VALUE,pressed=H.IS_ACTION_PRESSED,triggered=H.IS_ACTION_TRIGGERED},
        input_semantics="native query values between previous obs and this obs; no 15Hz resampling",
        qa_motion=qa_motion,qa_safe=qa_safe,
        obs=previous_obs})
    mode="recording";armed=false;status()
end
local function command(name)
    if name=="prepare" or name=="reset" then prepare()
    elseif name=="summon" then summon()
    elseif name=="start" then begin()
    elseif name=="stop" then
        if file then finish("manual_stop")
        elseif mode=="countdown" then armed=false;mode="ready";status() end
    elseif name=="finish_session" then
        if file then finish("session_end") end
        armed=false;mode="closed";Isaac.GetPlayer(0).ControlsEnabled=false;status()
    elseif os.getenv("ISAAC_HUMAN_QA")=="1" and name=="qa_script" then
        qa_motion=true;qa_safe=false;begin()
    elseif os.getenv("ISAAC_HUMAN_QA")=="1" and name=="qa_timeout" then
        qa_motion=false;qa_safe=true;begin()
    elseif os.getenv("ISAAC_HUMAN_QA")=="1" and name=="qa_win" then
        assert(mode=="recording");boss:Die()
    else error("Unknown recorder command: "..tostring(name)) end
end
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM,function() pending_room=false;phase_ticks=0 end)
mod:AddCallback(ModCallbacks.MC_POST_FIRE_TEAR,function() state.events.tears=state.events.tears+1 end)
mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG,function(_,e)
    if e:ToPlayer() then state.events.damage=state.events.damage+1 end
    -- Stray projectiles/bombs can outlive a manually stopped battle. Waiting
    -- states are safe; real recordings have no damage protection.
    if e:ToPlayer() and (qa_safe or mode~="recording") then return false end
end)
mod:AddCallback(ModCallbacks.MC_POST_NPC_DEATH,function(_,e)
    state.events.npc_deaths=state.events.npc_deaths+1;state.health.deaths[observe.key(e)]=true
end)
mod:AddCallback(ModCallbacks.MC_PRE_SPAWN_CLEAN_AWARD,function() state.events.clears=state.events.clears+1 end)
mod:AddCallback(ModCallbacks.MC_POST_UPDATE,guarded(function()
    if phase then
        phase_ticks=phase_ticks+1
        assert(phase_ticks<180,"Native reset did not complete")
        if not pending_room and phase_ticks>=3 then
            if phase=="seed" then
                Isaac.ExecuteCommand("remove *");game:GetSeeds():SetStartSeed("9AM0 7PRP")
                phase="rewind";phase_ticks=0;pending_room=true;Isaac.ExecuteCommand("stage 1")
            elseif phase=="rewind" then
                phase="ready";phase_ticks=0;pending_room=true;Isaac.ExecuteCommand("rewind")
            else
                local identity=tostring(game:GetLevel():GetCurrentRoomIndex())..":"..game:GetRoom():GetSpawnSeed()
                assert(not anchor or anchor==identity,"Rewind restored a different room")
                assert(game:GetSeeds():GetStartSeedString()=="9AM0 7PRP","Unexpected seed")
                anchor=identity;phase=nil;clean_room();mode="ready";summon();status()
            end
        end
        return
    end
    if mode=="recording" then
        if game:GetFrameCount()~=previous_obs.game_frame+1 or game:GetLevel():GetCurrentRoomIndex()~=previous_obs.room.room_idx then
            finish("external_reset");mode="idle";anchor=nil
            error_text="Game changed outside recorder; prepare the room again";status();return
        end
        frame=frame+1;state.logic_frames=frame;observe.update()
        local now=Isaac.GetTime()
        -- These readbacks themselves invoke MC_INPUT_ACTION. Do not mislabel
        -- recorder-generated queries as input queries issued by the game update.
        sampling_input=true
        local obs=observe.snapshot()
        local p=Isaac.GetPlayer(0)
        local movement=p:GetMovementInput();local shooting=p:GetShootingInput()
        sampling_input=false
        for _,q in ipairs(queries) do
            input_total=input_total+1
            if q.value==true or (type(q.value)=="number" and q.value>0) then input_active=input_active+1 end
        end
        append(file,{type="frame",frame=frame,previous_game_frame=previous_obs.game_frame,
            elapsed_ms=now-begin_ms,wall_delta_ms=now-last_ms,queries=queries,
            terminal_evidence={boss_dead=not boss:Exists() or boss:IsDead(),player_dead=p:IsDead()},
            movement={movement.X,movement.Y},shooting={shooting.X,shooting.Y},obs=obs})
        written=assert(file:seek());previous_obs=obs;last_ms=now;queries={}
        if p:IsDead() then finish("death")
        elseif not boss:Exists() or boss:IsDead() then finish("win")
        elseif frame>=3600 then finish("timeout") end
    end
    -- Once the game-over menu opens there are no POST_UPDATE callbacks. Restore
    -- the safe waiting room immediately AFTER the terminal frame has been saved.
    if recover_after_death then recover_after_death=false;prepare();return end
    -- Native room transitions must not run inside MC_POST_RENDER: rewind there
    -- reproduced a Shader stack empty crash during the reset stress test.
    if queued_command then
        local c=queued_command;queued_command=nil
        if c.id then command_id=c.id end
        command(c.command)
    end
    if armed and Isaac.GetTime()>=start_at and not game:IsPaused() then start_record() end
end))
mod:AddCallback(ModCallbacks.MC_POST_RENDER,guarded(function()
    local now=Isaac.GetTime()
    if now-last_status>=500 then
        local f=io.open(root..string.format("/command-%06d.json",command_id+1),"r")
        if f then
            local c=json.decode(f:read("*a"));f:close()
            if not queued_command then queued_command={command=c.command,id=command_id+1} end
        end
        status()
    end
    if mode~="closed" then
        if not queued_command then
            if Input.IsButtonTriggered(Keyboard.KEY_F6,0) and mode=="ready" then queued_command={command="start"} end
            if Input.IsButtonTriggered(Keyboard.KEY_F7,0) and mode~="preparing" then queued_command={command="reset"} end
            if Input.IsButtonTriggered(Keyboard.KEY_F8,0) and (mode=="recording" or mode=="countdown") then queued_command={command="stop"} end
        end
    end
    local label="Recorder: "..mode.." | "..frame.." frames | F6 start / F7 reset / F8 stop"
    if mode=="countdown" then label="Recording starts in "..math.max(0,math.ceil((start_at-now)/1000)) end
    if mode=="error" then label="RECORDING FAILED - check side panel" end
    Isaac.RenderText(label,35,42,mode=="error" and 1 or 0.8,mode=="error" and 0.2 or 0.95,0.7,1)
end))
mod:AddCallback(ModCallbacks.MC_PRE_GAME_EXIT,guarded(function() if file then finish("game_exit") end end))
status()
