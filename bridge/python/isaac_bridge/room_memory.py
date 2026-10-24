"""Raw room-visit writer, production transport helpers and eager reference (step 1).

One learned query compresses every observed frame in a visit to one D-wide token.
The learner recomputes those sources, including rooms before its chunk, with a graph.
Main-policy H and decision KV still use the existing rollout semantics: this module
does NOT implement the separate state-freshness change.

RawRoomState accompanies the actor/CF TemporalState; closed immutable visits travel
in Rollout's IPC metadata. RoomSourceBatch connects the learner's future PPO loss to
the raw source encoder. RawRoomReplay remains a small single-worker test reference.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class RawRoomState:
    """Host half of TemporalState: immutable raw frames for each slot's open visit.

    Closed visits belong to the rollout, not this state. Reset/CF copy operate on the
    host history together with the device state. No sampling or importance rules.
    """
    def __init__(self, slots, capacity):
        self.capacity = capacity
        self.frames = [[] for _ in range(slots)]
        self.pending = [[] for _ in range(slots)]
        self.maps = np.zeros((slots, 169), np.uint8)
        self.count = np.zeros(slots, np.int64)

    def reset(self, slots):
        for s in slots:
            self.frames[s], self.pending[s] = [], []
        self.maps[slots] = 0
        self.count[slots] = 0

    def copy(self, src, dst, other):
        # Copy all sources before assigning: overlapping slot permutations are legal.
        frames = [list(other.frames[s]) for s in src]
        pending = [list(other.pending[s]) for s in src]
        maps, count = other.maps[src].copy(), other.count[src].copy()
        for i, d in enumerate(dst):
            self.frames[d], self.pending[d] = frames[i], pending[i]
        self.maps[dst], self.count[dst] = maps, count

    def observe(self, rows, pending, slots):
        from .tok_obs import EV_EXIT, EV_NEW_ROOM
        closed = [None] * len(rows)
        for i, s in enumerate(slots):
            first = bool(rows['first'][i])
            m = rows['map'][i, 3].reshape(-1)
            shown = bool(m.any())
            moved = (shown and not np.array_equal(m, self.maps[s])) or bool(
                int(rows['events'][i]) & (EV_EXIT | EV_NEW_ROOM))
            if first:
                self.reset([s])
            elif moved and self.frames[s]:
                closed[i] = (np.concatenate(self.frames[s]), np.concatenate(self.pending[s]))
                for a in closed[i]:
                    a.flags.writeable = False
                self.count[s] += 1
                self.frames[s], self.pending[s] = [], []
            if first or shown:
                self.maps[s] = m
            # Once the existing room cap is reached no further writes are possible.
            if self.count[s] < self.capacity:
                r, p = rows[i:i + 1].copy(), pending[i:i + 1].copy()
                r.flags.writeable = p.flags.writeable = False
                self.frames[s].append(r)
                self.pending[s].append(p)
        return closed


def encode_visits(model, visits, micro=256):
    """Raw observations -> current Spatial -> learned visit token, in bounded batches.

    Checkpoint the Spatial pass when differentiating: a long room does not retain
    thousands of Spatial activations. Its observation sequence is never truncated.
    """
    from .tok_policy import to_batch
    from torch.utils.checkpoint import checkpoint
    dev = next(model.parameters()).device
    rows = np.concatenate([v[0] for v in visits])
    pending = np.concatenate([v[1] for v in visits])
    features = []
    for lo in range(0, len(rows), micro):
        b = to_batch(rows[lo:lo + micro], pending[lo:lo + micro], dev)
        features.append(checkpoint(model.encode_room_sources, b, use_reentrant=False)
                        if torch.is_grad_enabled() else model.encode_room_sources(b))
    return model.room_writer(torch.cat(features), [len(v[0]) for v in visits])


@torch.no_grad()
def prepare_room_step(model, state, rows, pending, slots):
    """Run outside a CUDA graph; its fixed-size source tensor is the graph input."""
    visits = state.raw_rooms.observe(rows, pending, slots)
    out = state.prev_o.new_zeros((len(rows), state.D))
    ids = [i for i, v in enumerate(visits) if v is not None]
    if ids:
        out[ids] = encode_visits(model, [visits[i] for i in ids])
    return out, visits


class RoomSourceBatch:
    """Exact gradient cache for ONE optimiser step (never across weight updates).

    All required causal visit prefixes are encoded with the current weights. PPO
    parts accumulate dL/dsource in a leaf; backward() replays the raw encoder/writer
    and applies that VJP before clipping/Adam. This bounds memory without detaching
    the writer's learning signal or re-encoding every visit for every PPO part.
    """
    def __init__(self, model, td, cids, micro=256):
        self.model, self.micro = model, micro
        needed = set()
        for c in cids:
            needed.update(td['raw_needed'][int(c)])
        self.groups, group, frames = [], [], 0
        for i in sorted(needed):
            n = len(td['raw_visits'][i][0])
            if group and frames + n > micro:
                self.groups.append(group)
                group, frames = [], 0
            group.append(i)
            frames += n
        if group:
            self.groups.append(group)
        self.visits = td['raw_visits']
        dev = next(model.parameters()).device
        source = torch.zeros((len(self.visits) + 1, model.tatt.D), device=dev)
        with torch.no_grad():
            for ids in self.groups:
                source[ids] = encode_visits(model, [self.visits[i] for i in ids], micro)
        self.source = source.requires_grad_()
        self.td = dict(td, o=self.source, r_src=torch.arange(len(source), device=dev), raw_room_memory=True)

    def backward(self):
        if not self.groups:
            return
        for ids in self.groups:
            values = encode_visits(self.model, [self.visits[i] for i in ids], self.micro)
            values.backward(self.source.grad[ids])


class RoomSequenceWriter(nn.Module):
    """Learned causal-visit pooling; no future query, event scores or hidden game state."""

    def __init__(self, width, heads):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(self, features, lengths):
        if not lengths or min(lengths) < 1 or sum(lengths) != len(features):
            raise ValueError('room sequences must be nonempty and partition all source features')
        x = pad_sequence(features.split(lengths), batch_first=True)
        n, t, d = x.shape
        pos = torch.arange(t, device=x.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, d, 2, device=x.device).float() * (-math.log(10000.0) / d))
        angles = pos * freq
        time = torch.stack((angles.sin(), angles.cos()), -1).flatten(1)[:, :d]
        x = x + time.to(x.dtype)[None]
        mask = torch.arange(t, device=x.device)[None] >= torch.tensor(lengths, device=x.device)[:, None]
        q = self.query.expand(n, -1, -1).to(x.dtype)
        pooled, _ = self.attention(q, x, x, key_padding_mask=mask, need_weights=False)
        return self.norm((q + pooled)[:, 0].float())


class RawRoomReplay:
    """Owns one worker's causal raw history, room-write events and legacy H/KV snapshots.

    append() acts on one ROW with the real TokPolicy. chunk() is a differentiable
    learner forward over any recorded [start, stop), including pre-chunk room sources.
    Use one instance per worker; fork() isolates counterfactual histories and state.
    An observation at a transition belongs to the NEW room, never the closing visit.
    """

    def __init__(self, model):
        from .tok_policy import TemporalState
        if model.room_writer is None:
            raise ValueError('RawRoomReplay requires TokPolicy(room_memory=True)')
        if model.memory:
            raise ValueError('RawRoomReplay reference currently supports the no-GRU model only')
        self.model = model
        self.state = TemporalState(model, 1, kv_dtype=torch.float32)
        self.rows, self.pending, self.episodes = [], [], []
        self.previous, self.has_previous, self.kv = [], [], []
        self.rooms = []  # (start, stop, available_at, episode, room_index, stage)
        self.episode, self.visit_start = -1, 0
        self.last_map = np.zeros(169, np.uint8)

    def fork(self):
        other = RawRoomReplay(self.model)
        other.state.copy_slot_(0, 0, self.state)
        for name in ('rows', 'pending', 'episodes', 'previous', 'has_previous', 'kv', 'rooms'):
            setattr(other, name, list(getattr(self, name)))
        other.episode, other.visit_start = self.episode, self.visit_start
        other.last_map = self.last_map.copy()
        return other

    def _sources(self, model, visits):
        from .tok_policy import to_batch
        device = next(model.parameters()).device
        if not visits:
            return torch.empty((0, model.tatt.D), device=device)
        rows = np.concatenate([np.concatenate(self.rows[a:b]) for a, b, *_ in visits])
        pending = np.concatenate([np.concatenate(self.pending[a:b]) for a, b, *_ in visits])
        return model.write_room_sequences(to_batch(rows, pending, device), [b - a for a, b, *_ in visits])

    @torch.no_grad()
    def append(self, rows, pending):
        from .tok_policy import to_batch
        from .tok_obs import EV_EXIT, EV_NEW_ROOM
        if len(rows) != 1 or len(pending) != 1:
            raise ValueError('RawRoomReplay.append takes exactly one worker observation')
        first = bool(rows['first'][0])
        if not self.rows and not first:
            raise ValueError('raw history must begin at episode reset, not an unknown hidden state')
        at = len(self.rows)
        current_map = rows['map'][0, 3].reshape(-1)
        shown = bool(current_map.any())
        moved = (shown and not np.array_equal(current_map, self.last_map)) or bool(
            int(rows['events'][0]) & (EV_EXIT | EV_NEW_ROOM))
        if first:
            self.episode += 1
            self.visit_start = at
        source = torch.zeros((1, self.model.tatt.D), device=self.state.prev_o.device)
        if not first and moved:
            room_index = int(self.state.nroom[0])
            if room_index < self.model.tatt.R:
                visit = (self.visit_start, at, at, self.episode, room_index, int(self.state.prev_stage[0]))
                source = self._sources(self.model, [visit])
                self.rooms.append(visit)
            self.visit_start = at
        if first or shown:
            self.last_map = current_map.copy()
        raw, pend = rows.copy(), pending.copy()
        raw.flags.writeable = pend.flags.writeable = False
        self.rows.append(raw)
        self.pending.append(pend)
        self.episodes.append(self.episode)
        self.previous.append(self.state.prev_o[0].detach().clone())
        self.has_previous.append(bool(self.state.steps[0]) and not first)
        td = dict(mode='step', state=self.state, slot=torch.zeros(1, dtype=torch.long, device=source.device),
                  room_source=source)
        out = self.model(to_batch(rows, pending, source.device), temporal=td,
                         need_vg=self.model.gtok is not None)
        self.kv.append(td['out']['kv'][0].detach().clone())
        return out

    def chunk(self, start, stop, model=None):
        """Recompute raw room sources on EVERY learner forward; H/decision KV stay stored.

        No room with available_at >= stop is even encoded. Earlier room tables are
        causal across visits and carry gradients through the current source encoder.
        """
        from .tok_policy import to_batch
        if not 0 <= start < stop <= len(self.rows):
            raise ValueError('chunk must be a nonempty recorded interval')
        model = self.model if model is None else model
        ta = model.tatt
        dev = next(model.parameters()).device
        visits = [r for r in self.rooms if r[2] < stop]
        encoded = self._sources(model, visits)
        sources = torch.cat((encoded, encoded.new_zeros((1, ta.D))), 0)
        count = len(visits)
        groups = []
        for ep in sorted({r[3] for r in visits}):
            groups.append([i for i, r in enumerate(visits) if r[3] == ep])
        groups.append([count])  # dummy source has its own causal group
        width = max(map(len, groups))
        grp = torch.full((len(groups), width), -1, dtype=torch.long, device=dev)
        gpos = torch.empty(count + 1, dtype=torch.long, device=dev)
        for g, ids in enumerate(groups):
            grp[g, :len(ids)] = torch.tensor(ids, device=dev)
            gpos[ids] = g * width + torch.arange(len(ids), device=dev)
        keys = torch.tensor([r[4] for r in visits] + [0], device=dev)
        stages = torch.tensor([r[5] for r in visits] + [0], device=dev)
        table_plan = dict(r_src=torch.arange(count + 1, device=dev), r_k=keys, r_stage=stages,
                          r_grp=grp, r_gpos=gpos)
        table = ta.room_table(sources, table_plan, track_grad=True)

        pre = [i for i, r in enumerate(visits) if r[2] < start and r[3] == self.episodes[start]]
        inside = [i for i, r in enumerate(visits) if start <= r[2] < stop]
        pre_ids, in_ids = pre or [count], inside or [count]
        n = stop - start
        context = range(start - ta.W + 1, start)
        zero_kv = sources.new_zeros((ta.L, 2, ta.D))
        ctx_kv = torch.stack([self.kv[i].to(dev) if i >= 0 else zero_kv for i in context])[None]
        plan = dict(
            raw_room_memory=True, imap=torch.arange(n, device=dev), omap=torch.arange(n, device=dev),
            qep=torch.tensor(self.episodes[start:stop], device=dev)[None],
            ctx_kv=ctx_kv,
            ctx_ep=torch.tensor([self.episodes[i] if i >= 0 else -3 for i in context], device=dev)[None],
            pre_valid=torch.tensor([[True] * len(pre) if pre else [False]], device=dev),
            pre_kv=[(k[pre_ids][None], v[pre_ids][None]) for k, v in table],
            in_valid=torch.tensor([[True] * len(inside) if inside else [False]], device=dev),
            in_o=sources[in_ids][None], in_k=keys[in_ids][None], in_stage=stages[in_ids][None],
            in_at=torch.tensor([[visits[i][2] - start for i in inside] if inside else [0]], device=dev),
            in_ep=torch.tensor([[visits[i][3] for i in inside] if inside else [-3]], device=dev))
        temporal = dict(mode='chunk', plan=plan, o_prev=torch.stack(self.previous[start:stop]).to(dev),
                        has=torch.tensor(self.has_previous[start:stop], device=dev))
        batch = to_batch(np.concatenate(self.rows[start:stop]), np.concatenate(self.pending[start:stop]), dev)
        return model(batch, temporal=temporal, need_vg=model.gtok is not None)
