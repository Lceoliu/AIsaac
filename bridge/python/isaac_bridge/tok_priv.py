"""Privileged critic (train_tok --priv-critic 1; rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 4): what the value heads may
read and the policy never does.

Way of the data. The bridge (abp_bridge.lua, `priv` command, abp-0.2.19) sends a frame "P <bytes>\\n" before each lean
observation of the one connection that asked; the lean observation itself is not touched, so ROW and everything the
actor's graphs read are the same bytes with the switch on or off, and a clone never inherits the switch. The worker
(tok_floor) decodes the frame into a PRIV record in a shared slot of its own; the trainer keeps it per worker
(arrival) and stores it next to the decision (Rollout.priv). The learner alone reads it: PrivBatch / PrivCritic here.

PRIV record:
  map [PRIV_MAP]: the floor's true layout on the 13 x 13 grid: every room, found or not. Channels: room, visited, clear,
    the player's room, boss, treasure, shop, another special room, secret, super secret;
  map_ver: changes when map does (per worker; the learner's CNN runs on the distinct maps);
  hp [NPC_K]: the room's monsters' hit points / HP_SCALE, largest first;
  scal [PRIV_S]: total monster HP / HP_SCALE (all of them, not only the first NPC_K), monsters / NPC_K, the boss's HP
    and maximum / BOSS_SCALE and its share, Curse of the Blind; the room's pickups per class / 5; pickups per class in
    the floor's other visited rooms as they were left / 10; in its unvisited rooms by their layouts' spawn lists / 10
    (the first entry of each spawn: a coarse count, random spawns and the rewards of clears are not known ahead). Visited
    rooms are remembered from the moment the connection asked: an episode started from a parked mid-floor state does not
    know what was left in the rooms before it;
  items [PED_K]: the true collectible ids of the room's pedestals under Curse of the Blind (0 otherwise: without the curse
    the policy sees them too);
  missing: 1 = no privileged data for this record (no frame; teacher and counterfactual records have zeros and this).

Pickup classes (PickupVariant, AB+ resources/scripts/enums.lua): heart, coin, key, bomb, chest, other (grab bag,
battery, shop item, a random pickup), pill or card, trinket, collectible.
"""
import struct

import numpy as np

from .tok_obs import N_COLLECTIBLE, SHAPE_CELLS

PRIV_MAP = (10, 13, 13)
NPC_K = 16
PED_K = 8
N_CLASS = 9
PRIV_S = 6 + 3 * N_CLASS
HP_SCALE = 50.0
BOSS_SCALE = 500.0
PRIV = np.dtype([('map', np.uint8, PRIV_MAP), ('map_ver', np.uint32), ('hp', np.float32, (NPC_K,)),
                 ('scal', np.float32, (PRIV_S,)), ('items', np.int16, (PED_K,)), ('missing', np.uint8)], align=True)
MAGIC = 0x56495250   # "PRIV"
_HEAD = struct.Struct('<IBBfHH9H')   # magic, flags (1: a layout block follows), blind, total HP, monsters sent, pedestals,
#                                      the room's pickups per class
_NPC = np.dtype([('hp', '<f4'), ('max', '<f4'), ('boss', 'u1')])
_ROOM = np.dtype([('idx', 'u1'), ('shape', 'u1'), ('type', 'u1'), ('flags', 'u1')])
_CLASSES = struct.Struct('<18H')     # the layout block's: other visited rooms, unvisited rooms
assert _NPC.itemsize == 9 and _ROOM.itemsize == 4
TYPE_CHANNEL = {5: 4, 4: 5, 2: 6, 7: 8, 8: 9}   # RoomType -> channel; another special room (> 1): 7; normal (1): none


def layout_map(rooms, m):
    """The channels of `m` [PRIV_MAP] from layout records (_ROOM): flags bit 0 visited, 1 clear, 2 the player's room."""
    m[:] = 0
    for r in rooms:
        kind = int(r['type'])
        ch = TYPE_CHANNEL.get(kind, 7 if kind > 1 else -1)
        for cell in SHAPE_CELLS.get(int(r['shape']), (0,)):
            cell += int(r['idx'])
            if not 0 <= cell < 169:
                continue
            y, x = divmod(cell, 13)
            f = int(r['flags'])
            m[0, y, x], m[1, y, x], m[2, y, x], m[3, y, x] = 1, f & 1, (f >> 1) & 1, (f >> 2) & 1
            if ch >= 0:
                m[ch, y, x] = 1


class PrivDecoder:
    """The frames of one connection into PRIV records. versions: an object with `.ver` (int): the worker's map version
    counter, shared over its episodes so that a map_ver stays unique within the worker."""

    def __init__(self, versions=None):
        self.versions = versions if versions is not None else type('V', (), dict(ver=0))()
        self.map = np.zeros(PRIV_MAP, np.uint8)
        self.ver = 0
        self.floor = np.zeros(2 * N_CLASS, np.float32)
        self.layout = False

    def decode(self, payload, row):
        """Write the frame into `row` (a length-1 view of a PRIV array). payload None or broken: zeros and missing."""
        if payload is None:
            row['map'], row['hp'], row['scal'], row['items'], row['map_ver'], row['missing'] = 0, 0.0, 0.0, 0, 0, 1
            return False
        h = _HEAD.unpack_from(payload, 0)
        if h[0] != MAGIC:
            raise ValueError(f'bad privileged frame magic {h[0]:#x}')
        flags, blind, total, n_npc, n_ped = h[1:6]
        off = _HEAD.size
        npc = np.frombuffer(payload, _NPC, n_npc, off)
        off += _NPC.itemsize * n_npc
        ped = np.frombuffer(payload, '<u2', n_ped, off)
        off += 2 * n_ped
        if flags & 1:
            n_r = payload[off]
            rooms = np.frombuffer(payload, _ROOM, n_r, off + 1)
            off += 1 + _ROOM.itemsize * n_r
            layout_map(rooms, self.map)
            self.floor[:] = _CLASSES.unpack_from(payload, off)
            off += _CLASSES.size
            self.versions.ver += 1
            self.ver = self.versions.ver
            self.layout = True
        if off != len(payload):
            raise ValueError(f'privileged frame has {len(payload) - off} trailing bytes')
        if not self.layout:   # (never: the first frame of a connection carries the layout)
            raise ValueError('privileged frame without a layout before it')
        row['map'][0] = self.map
        row['map_ver'] = self.ver
        hp = np.zeros(NPC_K, np.float32)
        if n_npc:
            s = np.sort(npc['hp'])[::-1][:NPC_K]
            hp[:len(s)] = s / HP_SCALE
        row['hp'][0] = hp
        boss_hp = boss_max = 0.0
        if n_npc and npc['boss'].any():
            bm = npc['boss'] != 0
            boss_hp, boss_max = float(npc['hp'][bm].sum()), float(npc['max'][bm].sum())
        s = row['scal'][0]
        s[0], s[1] = total / HP_SCALE, n_npc / NPC_K
        s[2], s[3], s[4], s[5] = boss_hp / BOSS_SCALE, boss_max / BOSS_SCALE, \
            boss_hp / boss_max if boss_max > 0 else 0.0, blind
        s[6:6 + N_CLASS] = np.array(h[6:15], np.float32) / 5.0
        s[6 + N_CLASS:] = self.floor / 10.0
        it = np.zeros(PED_K, np.int16)
        if n_ped and blind:
            k = min(n_ped, PED_K)
            it[:k] = np.minimum(ped[:k], N_COLLECTIBLE - 1)
        row['items'][0] = it
        row['missing'] = 0
        return True


class PrivBatch:
    """(the learner) the PRIV records of decisions at flat positions of a rollout, on the device: one copy of each
    distinct map (per worker and map_ver) with each record's id, the other fields per record."""

    def __init__(self, roll, flat, device):
        import torch
        w, t = flat // roll.cap, flat % roll.cap
        pv = roll.priv
        key = (w.astype(np.int64) << 33) | pv['map_ver'][w, t].astype(np.int64)
        _, first, inv = np.unique(key, return_index=True, return_inverse=True)
        self.maps = torch.from_numpy(np.ascontiguousarray(pv['map'][w[first], t[first]])).to(device)
        self.ids = torch.from_numpy(inv.reshape(-1).astype(np.int64)).to(device)
        self.hp = torch.from_numpy(np.ascontiguousarray(pv['hp'][w, t])).to(device)
        self.scal = torch.from_numpy(np.ascontiguousarray(pv['scal'][w, t])).to(device)
        self.items = torch.from_numpy(np.ascontiguousarray(pv['items'][w, t].astype(np.int64))).to(device)
        self.missing = torch.from_numpy(np.ascontiguousarray(pv['missing'][w, t])).to(device).float()
        self.missing_share = float(pv['missing'][w, t].mean()) if len(flat) else 0.0

    def __len__(self):
        return self.hp.shape[0]


def make_critic(width, p_dim=128):
    """The privileged encoder and the two value heads: p_t = encode(PRIV) (the map through a copy of the policy's Map CNN,
    the rest through an MLP), value(zr; p) and value_global(zg; p). init_from: the heads start as the policy's own
    (the new input columns zero: at the start the same values as the policy's heads, whatever p is)."""
    import torch
    from torch import nn

    class PrivCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.p_dim = int(p_dim)
            self.map = nn.Sequential(nn.Conv2d(PRIV_MAP[0], 32, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                     nn.Linear(64 * ((PRIV_MAP[1] + 1) // 2) * ((PRIV_MAP[2] + 1) // 2), p_dim))
            self.item = nn.Embedding(N_COLLECTIBLE, 16, padding_idx=0)
            self.rest = nn.Sequential(nn.Linear(NPC_K + PRIV_S + 16 + 1, p_dim), nn.GELU(), nn.Linear(p_dim, p_dim))
            self.join = nn.Sequential(nn.GELU(), nn.Linear(2 * p_dim, p_dim), nn.GELU())
            self.value = nn.Sequential(nn.Linear(width + p_dim, width), nn.GELU(), nn.Linear(width, 1))
            self.value_g = nn.Sequential(nn.Linear(width + p_dim, width), nn.GELU(), nn.Linear(width, 1))

        @torch.no_grad()
        def init_from(self, model):
            """The value heads as the policy's (TokPolicy.value, gtok.value), zero weights on the new columns."""
            for mine, old in ((self.value, model.value), (self.value_g, model.gtok.value)):
                w = old[0].weight.shape[1]
                mine[0].weight.zero_()
                mine[0].weight[:, :w] = old[0].weight
                mine[0].bias.copy_(old[0].bias)
                mine[2].weight.copy_(old[2].weight)
                mine[2].bias.copy_(old[2].bias)

        def encode(self, b, idx):
            """p_t [n, p_dim] of records idx of a PrivBatch; the CNN on their distinct maps."""
            u, inv = torch.unique(b.ids.index_select(0, idx), return_inverse=True)
            m = self.map(b.maps.index_select(0, u).float()).index_select(0, inv)
            it = self.item(b.items.index_select(0, idx)).sum(1)
            r = self.rest(torch.cat([b.hp.index_select(0, idx), b.scal.index_select(0, idx), it,
                                     b.missing.index_select(0, idx)[:, None]], -1))
            return self.join(torch.cat([m, r], -1))

        def missing(self, n, like):
            """p_t of n records without privileged data: the encoder on zeros with the missing flag (the teachers')."""
            if n == 0:
                return like.new_zeros((0, self.p_dim))
            dev = like.device
            m = self.map(torch.zeros((1,) + PRIV_MAP, device=dev))
            f = torch.zeros((1, NPC_K + PRIV_S + 16 + 1), device=dev)
            f[0, -1] = 1.0
            return self.join(torch.cat([m, self.rest(f)], -1)).expand(n, -1)

        def forward(self, zr, zg, p):
            """(value [n], value_global [n]) from the policy's zr, zg and p_t (float32)."""
            return self.value(torch.cat([zr.float(), p], -1))[:, 0], self.value_g(torch.cat([zg.float(), p], -1))[:, 0]
    return PrivCritic()
