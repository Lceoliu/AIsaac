"""B23 (2026-10-13): pedestal start builds and forced pickups (train_tok --start-build-mode pedestal, --pickup-force A:B,
--pickup-force-imitation; all off by default, then nothing here runs).

The user's finding (2026-10-13): random start builds raise floors cleared (1.29 -> 1.54 with 3 items) but the policy
picks up pedestal items on its own in only 5% of runs: the builds went straight into the inventory (the data never held
"pedestal -> walk over it -> item -> benefit") and the pedestal id and the held id used two different embedding
tables (the second half is tok_policy's item_shared option).

Pedestal builds (worker, tok_floor): at a floor start the build's collectibles are spawned as pedestals in the starting
  room (spawn_lua, run through Bridge.lua in the episode's clone) instead of AddCollectible; the ids spawned ride along
  in ROW 'bped' of every record of the episode (the trainer's episodes.jsonl: build_pedestals).
Forced pickups (worker + actor + learner): an episode is drawn as forced with probability p (the trainer writes p into
  the workers' stats column 'pf_p' after every update: linear in the run's game-hour fraction from A to B, the
  learning rate's fraction). In a forced episode, at every record whose room is clear (ROW player column 23, the
  bridge's room:IsClear()), not a shop or another special room (minimap channels 6 / 7 of the player's room: their
  pedestals have prices, coins or hearts) and with a collectible pedestal holding an item (ent_id type 5, variant 100,
  ent_item > 0) among the targets of that room (the ids on pedestals at the room's first forced record: an active item
  swapped out onto a pedestal is not walked back to), the worker writes ROW 'pforce' = the 8-direction move toward the
  nearest such pedestal (forced_move: ent columns 0 / 1 = dx / 200, dy / 200 from the player, column 2 = distance /
  300). The actor replaces that record's sampled move with it and its shoot with 0 (no shot) before it answers and
  stores the decision (train_tok answer(): the stored action, the action under way and what the worker applies stay
  one and the same); the other heads keep the policy's choice. A room's forcing stops at a room change, gives up after
  PICKUP_FORCE_CAP forced decisions since the room's last pickup, and an EV_ITEM while forcing counts as a forced
  pickup (the counter starts again for the room's next target). The learner keeps forced decisions in the value loss
  and the memory chunks, leaves them out of PPO's policy term and the entropy (advantage 0, ratio 1, and the advantage
  normalisation over the other decisions), and adds an imitation term on the move and shoot heads:
  --pickup-force-imitation x mean over the minibatch's trained decisions of -forced x (log pi_move + log pi_shoot).
"""
import math

import numpy as np

PICKUP_FORCE_CAP = 150     # forced decisions per room approach before the worker gives up for that room
PED_MAX = 3                # ROW 'bped': pedestal ids of a start build at most
# the bridge's move codes (abplus_geometry.MOVES / tok_branch.MOVES): 0 none, 1 up, then clockwise; an angle sector k
# (atan2(dy, dx) / 45 degrees, rounded; screen y points down) -> code ((k + 2) mod 8) + 1
SECTOR_MOVE = tuple(((k + 2) % 8) + 1 for k in range(8))


def parse_pickup_force(text):
    """'A:B' -> (A, B), both in [0, 1]; '' / '0' / '0:0' -> (0.0, 0.0) (off)."""
    text = (text or '').strip()
    if not text:
        return 0.0, 0.0
    a, _, b = text.partition(':')
    a, b = float(a), float(b if b else a)
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError(f'--pickup-force {text!r}: probabilities in [0, 1] expected')
    return a, b


def pickup_force_p(force, frac):
    """The per-episode probability at run fraction frac (0 .. 1): linear from force[0] to force[1]."""
    f = min(max(float(frac), 0.0), 1.0)
    return float(force[0] + (force[1] - force[0]) * f)


def move_toward(dx, dy):
    """The 8-direction move code (1 .. 8) toward the offset (dx, dy) (screen coordinates: y down)."""
    k = int(round(math.atan2(float(dy), float(dx)) / (math.pi / 4))) % 8
    return SECTOR_MOVE[k]


def pedestal_targets(r):
    """(entity index, collectible id, distance) of the collectible pedestals holding an item in record r (a ROW element
    or a length-1 ROW view)."""
    if getattr(r, 'shape', ()) == (1,):
        r = r[0]
    n = int(r['n_ent'])
    ids = r['ent_id'][:n]
    it = r['ent_item'][:n, 0]
    sel = np.flatnonzero((ids[:, 0] == 5) & (ids[:, 1] == 100) & (it > 0))
    return [(int(j), int(it[j]), float(r['ent'][j, 2])) for j in sel]


def room_priced(r):
    """True when the player's room is a shop or another special room on the minimap (ROW map channels 3 = the player's
    room, 6 = shop, 7 = another special room: devil / angel deals, library, ... have prices)."""
    if getattr(r, 'shape', ()) == (1,):
        r = r[0]
    m = r['map']
    here = m[3] > 0
    return bool(((m[6] > 0) & here).any() or ((m[7] > 0) & here).any())


def forced_move(r, targets=None):
    """(move code 1 .. 8, collectible id) toward the nearest pedestal with an item (restricted to the ids in targets
    when given) of record r, or (0, 0) when there is none. Direction from ent columns 0 / 1 (dx / 200, dy / 200 of the
    entity from the player), nearest by column 2 (distance / 300)."""
    peds = pedestal_targets(r)
    if targets is not None:
        peds = [p for p in peds if p[1] in targets]
    if not peds:
        return 0, 0
    j, item, _ = min(peds, key=lambda q: q[2])
    if getattr(r, 'shape', ()) == (1,):
        r = r[0]
    return move_toward(r['ent'][j, 0], r['ent'][j, 1]), item


class PickupForcer:
    """The worker's per-episode state of the forced pickups (module docstring). step() is called once per record of an
    ordinary episode, after encoding; returns the move to force (0: none) and counts into `counts`
    (dict: steps, forced, free)."""

    def __init__(self, cap=PICKUP_FORCE_CAP):
        self.cap = cap
        self.on = False
        self.reset_room(None)

    def begin(self, forced):
        self.on = bool(forced)
        self.reset_room(None)

    def reset_room(self, room):
        self.room, self.n, self.gave_up, self.streak, self.targets = room, 0, False, False, None

    def step(self, r, room, ev_item, done, counts):
        if room != self.room:
            self.reset_room(room)
        if ev_item:   # (EV_ITEM shows a few decisions after the pedestal emptied: the streak lasts until then)
            if self.streak:
                counts['forced'] += 1
                self.streak, self.n = False, 0
            else:
                counts['free'] += 1
        if not self.on or self.gave_up or done:
            return 0
        rr = r[0] if getattr(r, 'shape', ()) == (1,) else r
        if rr['player'][23] < 0.5 or room_priced(rr):
            return 0
        if self.targets is None:
            ids = {p[1] for p in pedestal_targets(rr)}
            if not ids:
                return 0
            self.targets = ids
        move, _ = forced_move(rr, self.targets)
        if not move:
            return 0
        if self.n >= self.cap:
            self.gave_up, self.streak = True, False
            return 0
        self.n += 1
        self.streak = True
        counts['steps'] += 1
        return move


# Bridge.lua chunk of a pedestal build: each collectible on a free pickup position near the player (offsets tried in
# order; Room:FindFreePickupSpawnPosition(pos, 0, true) when it answers, AB+'s Lua binds it), inside the room, at least
# 60 px from the player and from the pedestals already placed. Returns "id:x:y,..." of what it spawned.
SPAWN_LUA = """
local room = Game():GetRoom()
local p = Isaac.GetPlayer(0)
local base = p.Position
local offs = {{{{-120, 0}}, {{120, 0}}, {{0, -80}}, {{0, 80}}, {{-120, -80}}, {{120, -80}}, {{-120, 80}}, {{120, 80}},
  {{-200, 0}}, {{200, 0}}, {{-80, 0}}, {{80, 0}}}}
local placed, out = {{}}, {{}}
for _, id in ipairs({{{ids}}}) do
  local pos = nil
  for _, o in ipairs(offs) do
    local want = Vector(base.X + o[1], base.Y + o[2])
    local q = want
    local ok, r = pcall(function() return room:FindFreePickupSpawnPosition(want, 0, true) end)
    if ok and r ~= nil then q = r end
    local good = room:IsPositionInRoom(q, 16) and q:Distance(base) >= 60
    if good then
      for _, s in ipairs(placed) do
        if q:Distance(s) < 60 then good = false end
      end
    end
    if good then
      pos = q
      break
    end
  end
  if pos ~= nil then
    Isaac.Spawn(5, 100, id, pos, Vector(0, 0), nil)   -- EntityType.ENTITY_PICKUP, PickupVariant.PICKUP_COLLECTIBLE
    placed[#placed + 1] = pos
    out[#out + 1] = string.format('%d:%.0f:%.0f', id, pos.X, pos.Y)
  end
end
return table.concat(out, ',')
"""


def spawn_lua(items):
    """The Lua chunk that spawns these collectibles as pedestals in the current room (SPAWN_LUA)."""
    return SPAWN_LUA.format(ids=', '.join(str(int(i)) for i in items))


# A28 (2026-10-14, eval_tok --no-pedestals 1; evaluation only): every collectible pedestal (EntityType.ENTITY_PICKUP 5,
# PickupVariant.PICKUP_COLLECTIBLE 100; with or without an item, priced or free) of the current room removed, run
# through Bridge.lua's lua command in the episode's clone by tok_floor (cfg.no_pedestals) before the step of a record
# that shows one (no_pedestal_due). Returns the number removed.
NO_PEDESTAL_LUA = """
local n = 0
for _, e in ipairs(Isaac.GetRoomEntities()) do
  if e.Type == 5 and e.Variant == 100 then
    e:Remove()
    n = n + 1
  end
end
return tostring(n)
"""


def no_pedestal_due(r):
    """(A28) True when record r (a ROW element or a length-1 ROW view) shows a collectible pedestal among its entities
    (ent_id type 5, variant 100, with or without an item), or holds the full ENT_CAP entities (the nearest ones only: a
    pedestal may be among those left out). tok_floor then runs NO_PEDESTAL_LUA before the record's step."""
    if getattr(r, 'shape', ()) == (1,):
        r = r[0]
    n = int(r['n_ent'])
    ids = r['ent_id'][:n]
    return n >= r['ent_id'].shape[0] or bool(((ids[:, 0] == 5) & (ids[:, 1] == 100)).any())


# B39 (2026-10-18, train_tok --no-active-prob p / eval_tok --no-active 1; floor / run modes): player 0's active item
# removed at an episode's start, whatever the character starts with (AB+ Isaac: The D6, collectible 105, 6 charges;
# A29: C99 never saw a start without it). Run through Bridge.lua's lua command in the episode's fork by tok_floor before
# the episode's first observation (and before a start build, whose active item then stays). RemoveCollectible clears the
# active slot (GetActiveItem 0: ROW player[21] has_active 0, active_ready 0) and the held count (the inventory block's
# rescan key holds GetActiveItem and GetCollectibleCount: no held-item token, pitem[0] 0, pinv[0..1] 0). Returns
# "removed id|active after|held count of the removed id after" (0|0|0: the character had no active item).
NO_ACTIVE_LUA = """
local p = Isaac.GetPlayer(0)
local a = p:GetActiveItem()
if a ~= 0 then p:RemoveCollectible(a) end
return tostring(a) .. "|" .. tostring(p:GetActiveItem()) .. "|" .. tostring(a ~= 0 and p:GetCollectibleNum(a) or 0)
"""


def parse_no_active(result):
    """(B39) Bridge.lua's answer to NO_ACTIVE_LUA -> (removed id, active after, held count after). ValueError when the
    answer is malformed or the player still holds an active item (tok_floor: the episode start fails and is counted
    as an error, like any other failed start)."""
    try:
        removed, after, count = (int(float(v)) for v in str(result).strip().split('|'))
    except (TypeError, ValueError):
        raise ValueError(f'no-active: unexpected answer {result!r}') from None
    if after != 0 or count != 0:
        raise ValueError(f'no-active: active item {removed} not removed (active {after}, held {count})')
    return removed, after, count


def parse_spawned(result):
    """Bridge.lua's answer to spawn_lua -> [(id, x, y), ...] of the pedestals spawned."""
    out = []
    for part in (result or '').split(','):
        bits = part.strip().split(':')
        if len(bits) == 3:
            out.append((int(bits[0]), float(bits[1]), float(bits[2])))
    return out


# ------------------------------------------------------------------ learner side (train_tok.learn, forced decisions)
def normalise_advantages(a, forced, w=None):
    """PPO's advantage normalisation with forced decisions left out: mean and std over the rows that are trained by
    PPO (weight w, memory chunks' 0 / 1, times 1 - forced), the result 0 on every other row. Returns (a_norm, std)."""
    keep = 1.0 - forced.float()
    if w is not None:
        keep = keep * w
    n = keep.sum()
    mu = (a * keep).sum() / n.clamp(min=1.0)
    std = (((a - mu) ** 2 * keep).sum() / (n - 1.0).clamp(min=1.0)).sqrt()
    return (a - mu) / (std + 1e-8) * keep, std


def move_shoot_logp(logits, actions):
    """log pi(move) + log pi(shoot) of the actions [B, >= 2] under the logits [B, >= 14] (move 9, shoot 5 first)."""
    import torch.nn.functional as F
    lg = logits.float()
    lm = F.log_softmax(lg[:, :9], -1).gather(-1, actions[:, 0:1])[:, 0]
    ls = F.log_softmax(lg[:, 9:14], -1).gather(-1, actions[:, 1:2])[:, 0]
    return lm + ls


def ppo_part_terms(logits, logp, entropy, b_logp, a, value, ret, actions, forced, clip, pm=None, pn=None, prox=None,
                   weight_max=None):
    """A minibatch part's losses with forced decisions (train_tok.learn's pickup branch): returns (pg, vf, ent, imit,
    ratio, kl, clipped). Forced rows: their behaviour log-probability is taken as the current one (ratio exactly 1: the
    stored log-probability is the sampled action's, not the forced one's) and their advantage is 0 (normalise_advantages),
    so they add exactly 0 to pg and nothing to its gradient; their entropy is left out; vf covers every row; imit = mean
    over the part of -forced x (log pi_move + log pi_shoot). Means as train_tok's pmean: over the rows, or (memory) the
    trained rows' weights pm summed to pn. prox (--decoupled): the update-start log-probabilities."""
    import torch
    f = forced.float()
    keep = 1.0 - f

    def pmean(x):
        return x.mean() if pm is None else (x * pm).sum() / pn
    lp_d = logp.detach()
    if prox is None:
        b = torch.where(forced, lp_d, b_logp)
        ratio = torch.exp(logp - b)
        pg = -pmean(torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a))
    else:
        px = torch.where(forced, lp_d, prox)
        b = torch.where(forced, lp_d, b_logp)
        ratio = torch.exp(logp - px)
        wgt = torch.exp(px - b)
        if weight_max is not None:
            wgt = wgt.clamp(max=weight_max)
        pg = -pmean(wgt * torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a))
    vf = 0.5 * pmean((value - ret) ** 2)
    ent = pmean(entropy.sum(-1) * keep)
    imit = -pmean(move_shoot_logp(logits, actions) * f)
    with torch.no_grad():
        kl = pmean((b - logp) * keep)
        clipped = pmean(((ratio - 1).abs() > clip).float() * keep)
    return pg, vf, ent, imit, ratio, kl, clipped
