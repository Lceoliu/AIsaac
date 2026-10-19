"""Raw room-visit writer and eager replay reference (step 1 only).

One learned query compresses every observed frame in a visit to one D-wide token.
The learner recomputes those sources, including rooms before its chunk, with a graph.
Main-policy H and decision KV still use the existing rollout semantics: this module
does NOT implement the separate state-freshness change.

RawRoomReplay is a single-worker, CPU-archive/eager-device reference, not the existing
CUDA GraphActor / multiprocessing transport. Keep it alive across rollout boundaries.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


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
