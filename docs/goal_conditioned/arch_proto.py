"""Prototype of the recommended goal-conditioned CombatTransformer (architecture review, CPU only, read-only repo).

New modules (all gated by gate = goal_task[..., 1:].sum(-1), 0 for COMBAT):
  goal_encoder  MLP(G -> 64 -> 64) + LN, random init
  goal_fusion   Linear(64 -> 256), ZERO, added to fusion[0] output before the LayerNorm
  goal_query    Linear(64 -> 128), ZERO, added to the entity-attention queries
  goal_conv     Conv2d(1 -> 16, 3, bias=False), ZERO, added to map_cnn[0] output before its GELU
  room_cnn      Conv(8->16) GELU Conv(16->32,s2) GELU Conv(32->32,s2) GELU FC(896->128) GELU, random init
  room_fusion   Linear(128 -> 256), ZERO, added before the fusion LayerNorm
Checks: COMBAT bitwise identity (both encode paths, mixed batches, full temporal windows), gradient flow,
stage-1 freezing keeps COMBAT bitwise, goal visibility in entity-free frames, parameter counts, Adam name mapping.
"""
import copy
exec(open('count_params.py').read().split("net = CombatTransformer(space())")[0])
torch.set_num_threads(4)
G = 10          # task one-hot 4, dx, dy, d, sin, cos, in_window
ROOM_C = 8      # inside, walkable, hazard, destructible, door, player, goal, window


class GoalCombatTransformer(CombatTransformer):
    def __init__(self, observation_space, room_branch=True, **kw):
        super().__init__(observation_space, **kw)
        self.goal_encoder = nn.Sequential(nn.Linear(G, 64), nn.GELU(), nn.Linear(64, 64), nn.LayerNorm(64))
        self.goal_fusion = nn.Linear(64, self.model_dim)
        self.goal_query = nn.Linear(64, self.queries.shape[1])
        self.goal_conv = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.room_branch = room_branch
        if room_branch:
            self.room_cnn = nn.Sequential(nn.Conv2d(ROOM_C, 16, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                          nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
                                          nn.Flatten(), nn.Linear(32 * 4 * 7, 128), nn.GELU())
            self.room_fusion = nn.Linear(128, self.model_dim)
        self.zero_injections()

    ZERO = ('goal_fusion', 'goal_query', 'goal_conv', 'room_fusion')

    def zero_injections(self):
        # After SB3's ortho_init (policies.py:612-633 re-initialises every Linear/Conv2d of the extractor).
        with torch.no_grad():
            for name in self.ZERO:
                m = getattr(self, name, None)
                if m is None:
                    continue
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()

    def _inject(self, o, player, entities, padding, n):
        gate = o['goal'][:, 1:4].sum(-1)                         # (n,), 0 = COMBAT
        h = self.goal_encoder(o['goal'])
        queries = (self.queries.unsqueeze(0) + self.player_query(player).unsqueeze(1)
                   + (gate[:, None] * self.goal_query(h)).unsqueeze(1))
        summary, _ = self.entity_attention(queries, entities, entities, key_padding_mask=padding, need_weights=False)
        c1 = self.map_cnn[0](o['terrain']) + gate[:, None, None, None] * self.goal_conv(o['goal_map'])
        terrain = c1
        for layer in list(self.map_cnn)[1:]:
            terrain = layer(terrain)
        return gate, h, summary, terrain

    def _fuse(self, o, gate, h, parts):
        pre = self.fusion[0](torch.cat(parts, -1)) + gate[:, None] * self.goal_fusion(h)
        if self.room_branch:
            bits = o['room_bits'].long()                         # (n, 16, 28) packed channels
            room = ((bits[:, None] >> torch.arange(ROOM_C)[None, :, None, None]) & 1).float()
            pre = pre + gate[:, None] * self.room_fusion(self.room_cnn(room))
        return self.fusion[2](self.fusion[1](pre))

    def encode_frame_static(self, o):
        n = o['player'].shape[0]
        inputs = [o['player'], self.animation_embedding(o['player_anim']),
                  self.active_item(o['active_kind'].long().squeeze(-1))]
        if self.has_deadline:
            inputs.append(o['remaining_time'].unsqueeze(-1))
        if self.combat_dim:
            inputs.append(o['combat'])
        player = self.player(torch.cat(inputs, -1))
        kinds = o['entity_kind'].long()
        features = torch.cat([o['entities'], o['entity_flags']], -1) if self.flag_dim else o['entities']
        encoded = self.entity(torch.cat([features, self.entity_type(kinds[..., 0]), self.variant(kinds[..., 1]),
                                         self.subtype(kinds[..., 2]), self.animation_embedding(o['entity_anim'])], -1))
        entities = torch.cat([self.empty_entity.expand(n, 1, -1), encoded], 1)
        padding = torch.cat([torch.zeros((n, 1), dtype=torch.bool), ~o['entity_mask'].bool()], 1)
        gate, h, summary, terrain = self._inject(o, player, entities, padding, n)
        actions = self.previous_action_features(o['previous_action'])
        fused = self._fuse(o, gate, h, [player, summary.flatten(1), terrain, actions])
        if self.value_extra:
            fused = torch.cat([fused, o['fire_distance'].unsqueeze(-1)], -1)
        return fused * o['history_mask'].unsqueeze(-1)


def goal_space(terrain=(7, 9, 15)):
    s = space(terrain)
    s.spaces['goal'] = Box((H, G)); s.spaces['goal_map'] = Box((H, 1, 9, 15)); s.spaces['room_bits'] = Box((H, 16, 28))
    return s


def frames(n, entities, task, seed=1):
    g = torch.Generator().manual_seed(seed)
    o = {}
    for k, b in goal_space().spaces.items():
        shape = (n,) + tuple(b.shape[1:])
        if k in ('player_anim', 'active_kind', 'entity_kind', 'entity_anim'):
            o[k] = torch.randint(0, 64, shape, generator=g, dtype=torch.int32)
        else:
            o[k] = torch.randn(shape, generator=g) * 0.5
    o['entity_mask'] = (torch.arange(256)[None, :] < torch.tensor(entities)[:, None]).float()
    o['terrain'] = (torch.rand(o['terrain'].shape, generator=g) > 0.6).float()
    o['previous_action'] = torch.zeros(n, 18); o['previous_action'][:, 0] = 1
    o['history_mask'] = torch.ones(n)
    o['goal_map'] = (torch.rand(o['goal_map'].shape, generator=g) > 0.97).float()
    o['room_bits'] = torch.randint(0, 256, (n, 16, 28), generator=g, dtype=torch.int32)
    o['goal'][:, :4] = 0
    o['goal'][torch.arange(n), torch.as_tensor(task)] = 1          # random goal numbers even for COMBAT rows
    return o


torch.manual_seed(0)
old = CombatTransformer(space()).eval()
with torch.no_grad():   # non-trivial LayerNorm affine params, as in a trained checkpoint
    for m in old.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.normal_(1, 0.2); m.bias.normal_(0, 0.2)
new = GoalCombatTransformer(goal_space()).eval()

# ---- non-strict migration: only the listed new prefixes may be missing, nothing else may differ ----
NEW = ('goal_encoder.', 'goal_fusion.', 'goal_query.', 'goal_conv.', 'room_cnn.', 'room_fusion.')
src, dst = old.state_dict(), new.state_dict()
missing = [k for k in dst if k not in src]
assert all(k.startswith(NEW) for k in missing), missing
assert all(k in dst and dst[k].shape == v.shape for k, v in src.items())
res = new.load_state_dict(src, strict=False)
assert not res.unexpected_keys and all(k.startswith(NEW) for k in res.missing_keys)
new.zero_injections()
print('new keys', len(missing), 'new tensors', len(missing))

n = 64
ents = [0, 1, 5, 17, 64, 256] * 10 + [3, 3, 3, 3]
combat = frames(n, ents, [0] * n)
mixed_task = [0, 1, 2, 3] * 16
mixed = frames(n, ents, mixed_task, seed=2)
with torch.no_grad():
    r_old = old.encode_frame_static(combat); r_new = new.encode_frame_static(combat)
    print('COMBAT static  bitwise', torch.equal(r_old, r_new))
    m_old = old.encode_frame_static(mixed); m_new = new.encode_frame_static(mixed)
    c = torch.tensor(mixed_task) == 0
    print('mixed batch: COMBAT rows bitwise', torch.equal(m_old[c], m_new[c]),
          ' GOTO rows (zero-init) bitwise', torch.equal(m_old[~c], m_new[~c]))

# ---- full windows through encode_frames + temporal Transformer (eager path) ----
def windows(b, l, task, seed):
    o = frames(b * l, [0, 3, 17, 64] * (b * l // 4), [task] * (b * l), seed)
    o = {k: v.reshape(b, l, *v.shape[1:]) for k, v in o.items()}
    lengths = torch.tensor([l, l // 2, 1, l - 3])[:b]
    o['history_mask'] = (torch.arange(l)[None, :] < lengths[:, None]).float()   # right padding
    o['time'] = torch.arange(l)[None, :].float().expand(b, l) * (4 / 30)
    return o

# encode_frames for the new class = reference implementation patched with the same injection (compact path):
def encode_frames_new(self, obs):
    valid = obs['history_mask'].bool(); b, l = valid.shape
    flat = {k: v[valid] for k, v in obs.items() if k not in ('history_mask', 'time')}
    flat['history_mask'] = torch.ones(len(flat['player']))
    out = self.encode_frame_static(flat)
    seq = out.new_zeros(b, l, self.features_dim); seq[valid] = out
    return seq

with torch.no_grad():
    w = windows(4, 64, 0, 5)
    seq_old = old.temporal_features(old.encode_frames(w), w['time'], w['history_mask'])
    seq_new = new.temporal_features(encode_frames_new(new, w), w['time'], w['history_mask'])
    ref_static = old.temporal_features(encode_frames_new(old, w), w['time'], w['history_mask'])
    print('COMBAT windows: new(static per frame) vs old(static per frame) bitwise', torch.equal(seq_new, ref_static),
          ' vs old compact encode_frames maxdiff %.2e' % (seq_old - seq_new).abs().max().item())

# ---- gradient flow on GOTO frames ----
goto = frames(n, [0] * 32 + [5] * 32, [1] * n, seed=3)
new.train()
def grads(model, o):
    model.zero_grad()
    out = model.encode_frame_static(o)
    (out.square().mean() + out[:, 7].mean()).backward()
    return {name: sum(p.grad.abs().sum().item() for p in getattr(model, name).parameters() if p.grad is not None)
            for name in ('goal_encoder', 'goal_fusion', 'goal_query', 'goal_conv', 'room_cnn', 'room_fusion')}
print('step-0 grads', {k: f'{v:.2e}' for k, v in grads(new, goto).items()})

# ---- stage 1: freeze every old parameter, train only the new modules on GOTO frames ----
for name, p in new.named_parameters():
    p.requires_grad_(name.startswith(NEW))
opt = torch.optim.Adam([p for p in new.parameters() if p.requires_grad], lr=1e-3)
target = torch.randn(n, 257)
for _ in range(20):
    opt.zero_grad()
    out = new.encode_frame_static(goto)
    ((out - target) ** 2).mean().backward()
    opt.step()
print('after 20 stage-1 steps grads', {k: f'{v:.2e}' for k, v in grads(new, goto).items()})
new.eval()
with torch.no_grad():
    print('stage 1: COMBAT still bitwise', torch.equal(old.encode_frame_static(combat), new.encode_frame_static(combat)),
          ' mixed COMBAT rows bitwise', torch.equal(old.encode_frame_static(mixed)[c], new.encode_frame_static(mixed)[c]),
          ' GOTO rows changed', not torch.equal(old.encode_frame_static(mixed)[~c], new.encode_frame_static(mixed)[~c]))
    # goal visibility in entity-free GOTO frames: change only the goal vector
    e0 = frames(8, [0] * 8, [1] * 8, seed=4); e1 = {k: v.clone() for k, v in e0.items()}
    e1['goal'][:, 4:] = torch.randn(8, G - 4)
    d_full = (new.encode_frame_static(e0) - new.encode_frame_static(e1)).abs().max().item()
    q_only = copy.deepcopy(new); q_only.goal_fusion.weight.zero_(); q_only.goal_fusion.bias.zero_()
    q_only.room_branch = False; q_only.goal_conv.weight.zero_()
    d_q = (q_only.encode_frame_static(e0) - q_only.encode_frame_static(e1)).abs().max().item()
    print('entity-free GOTO, goal vector changed: output diff with direct path %.3e, query-only %.3e' % (d_full, d_q))

# ---- parameter counts ----
def count(m): return sum(p.numel() for p in m.parameters())
for name in ('goal_encoder', 'goal_fusion', 'goal_query', 'goal_conv', 'room_cnn', 'room_fusion'):
    print(f'  {name:13s} {count(getattr(new, name)):>9,d}')
print('old extractor', f'{count(old):,d}', ' new extractor', f'{count(new):,d}', ' delta', f'{count(new) - count(old):,d}',
      ' old tensors', len(list(old.parameters())), ' new tensors', len(list(new.parameters())))
lite = GoalCombatTransformer(goal_space(), room_branch=False)
print('without room branch: delta', f'{count(lite) - count(old):,d}')
# MACs per frame of the room branch (hand formula from the layer shapes)
macs = 16 * 28 * 16 * ROOM_C * 9 + 8 * 14 * 32 * 16 * 9 + 4 * 7 * 32 * 32 * 9 + 896 * 128 + 128 * 256
goal_macs = G * 64 + 64 * 64 + 64 * 256 + 64 * 128 + 9 * 15 * 16 * 9
print('room branch MAC/frame', f'{macs:,d}', ' goal paths MAC/frame', f'{goal_macs:,d}')

# ---- shared submodule under three attribute names (SB3 share_features_extractor=True) ----
class P(nn.Module):
    def __init__(s):
        super().__init__(); s.features_extractor = nn.Linear(2, 2)
        s.pi_features_extractor = s.features_extractor; s.vf_features_extractor = s.features_extractor
print('state_dict keys with a shared extractor:', list(P().state_dict().keys()), ' parameters():', len(list(P().parameters())))

# ---- Adam state by name: old optimizer (index order of old.parameters()) -> new optimizer ----
old_opt = torch.optim.Adam(old.parameters(), lr=1e-4)
old.zero_grad(); old.encode_frame_static(combat).square().mean().backward(); old_opt.step()
old_names = [k for k, _ in old.named_parameters()]
new_names = [k for k, _ in new.named_parameters()]
for p in new.parameters(): p.requires_grad_(True)
new_opt = torch.optim.Adam(new.parameters(), lr=1e-4)
sd_old = old_opt.state_dict(); sd_new = new_opt.state_dict()
state = {new_names.index(nm): sd_old['state'][i] for i, nm in enumerate(old_names) if i in sd_old['state']}
new_opt.load_state_dict({'state': state, 'param_groups': sd_new['param_groups']})
new.zero_grad(); new.encode_frame_static(mixed).square().mean().backward(); new_opt.step()
print('Adam by name: mapped', len(state), 'of', len(new_names), 'tensors; step OK; new tensors got fresh state',
      all(new_opt.state[p]['step'].item() == 1 for nm, p in new.named_parameters() if nm.startswith(NEW)))
