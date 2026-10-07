"""Second set of CPU checks: the compact encode_frames path with the injections, the per-task value head and a
gated LoRA on the actor's hidden layer, all against the unchanged CombatTransformer / heads."""
import copy
exec(open('arch_proto.py').read().split("torch.manual_seed(0)\nold = ")[0])


class GoalCompact(GoalCombatTransformer):
    def encode_frames(self, observations):   # the repo's encode_frames with the same three injections
        valid_time = observations['history_mask'].bool()
        batch, length = valid_time.shape
        p = observations['player'][valid_time]
        animation = self.animation_embedding(observations['player_anim'][valid_time])
        active = self.active_item(observations['active_kind'][valid_time].long().squeeze(-1))
        inputs = [p, animation, active]
        if self.has_deadline: inputs.append(observations['remaining_time'][valid_time].unsqueeze(-1))
        if self.combat_dim: inputs.append(observations['combat'][valid_time])
        player = self.player(torch.cat(inputs, -1))
        mask = observations['entity_mask'][valid_time].bool()
        features = observations['entities'][valid_time][mask]
        if self.flag_dim:
            features = torch.cat([features, observations['entity_flags'][valid_time][mask]], -1)
        kinds = observations['entity_kind'][valid_time][mask].long()
        anim = self.animation_embedding(observations['entity_anim'][valid_time][mask])
        encoded = self.entity(torch.cat([features, self.entity_type(kinds[:, 0]),
                                         self.variant(kinds[:, 1]), self.subtype(kinds[:, 2]), anim], -1))
        counts = mask.sum(-1)
        maximum = int(counts.max().item())
        entities = self.empty_entity.expand(len(player), maximum + 1, -1).clone()
        padding = torch.ones((len(player), maximum + 1), dtype=torch.bool)
        padding[:, 0] = False
        rows, _ = mask.nonzero(as_tuple=True)
        ranks = (mask.long().cumsum(-1))[mask]
        entities[rows, ranks] = encoded
        padding[rows, ranks] = False
        o = {k: observations[k][valid_time] for k in ('goal', 'goal_map', 'room_bits', 'terrain')}
        gate, h, summary, terrain = self._inject(o, player, entities, padding, len(player))
        actions = self.previous_action_features(observations['previous_action'][valid_time])
        fused = self._fuse(o, gate, h, [player, summary.flatten(1), terrain, actions])
        if self.value_extra:
            fused = torch.cat([fused, observations['fire_distance'][valid_time].unsqueeze(-1)], -1)
        sequence = fused.new_zeros(batch, length, self.features_dim)
        sequence[valid_time] = fused
        return sequence


def windows(b, l, tasks, seed):
    o = frames(b * l, [0, 3, 17, 64] * (b * l // 4), tasks, seed)
    o = {k: v.reshape(b, l, *v.shape[1:]) for k, v in o.items()}
    lengths = torch.tensor([l, l // 2, 1, l - 3])[:b]
    o['history_mask'] = (torch.arange(l)[None, :] < lengths[:, None]).float()
    o['time'] = torch.arange(l)[None, :].float().expand(b, l) * (4 / 30)
    return o


torch.manual_seed(0)
old = CombatTransformer(space()).eval()
with torch.no_grad():
    for m in old.modules():
        if isinstance(m, nn.LayerNorm):
            m.weight.normal_(1, 0.2); m.bias.normal_(0, 0.2)
new = GoalCompact(goal_space()).eval()
new.load_state_dict(old.state_dict(), strict=False); new.zero_injections()
# pretend stage 1 already trained the new modules (non-zero injections)
with torch.no_grad():
    for name in GoalCombatTransformer.ZERO:
        for p in getattr(new, name).parameters():
            p.normal_(0, 0.05)

l = 64
w_c = windows(4, l, [0] * (4 * l), 7)
w_m = windows(4, l, ([0] * l + [1] * l) * 2, 8)      # windows 0,2 COMBAT, 1,3 GOTO (per-window task: history reset)
with torch.no_grad():
    so = old.temporal_features(old.encode_frames(w_c), w_c['time'], w_c['history_mask'])
    sn = new.temporal_features(new.encode_frames(w_c), w_c['time'], w_c['history_mask'])
    print('compact path, COMBAT windows, trained injections: bitwise', torch.equal(so, sn))
    mo = old.temporal_features(old.encode_frames(w_m), w_m['time'], w_m['history_mask'])
    mn = new.temporal_features(new.encode_frames(w_m), w_m['time'], w_m['history_mask'])
    print('compact path, batch of COMBAT + GOTO windows: COMBAT windows bitwise', torch.equal(mo[[0, 2]], mn[[0, 2]]),
          ' GOTO windows differ', not torch.equal(mo[[1, 3]], mn[[1, 3]]))
    fs = new.encode_frame_static({k: v[:, 0] for k, v in w_m.items() if k != 'time'})
    fe = new.encode_frames({k: v[:, :1] for k, v in w_m.items()})[:, 0]
    print('static vs compact, new model incl. GOTO rows: maxdiff %.2e (test_graph_sampler bound 1e-4)' % (fs - fe).abs().max())


# ---- per-task value head and gated LoRA on the actor hidden layer ----
EXTRA = 3            # critic-only extras appended to the features: fire_distance, gate, d_geo / 40
policy_net = nn.Sequential(nn.Linear(256, 256), nn.Tanh())
value_mlp = nn.Sequential(nn.Linear(257, 256), nn.Tanh())
value_net = nn.Linear(256, 1)


class TaskValue(nn.Module):
    """COMBAT: the old value_net on the old value MLP; GOTO: its own MLP on [transformer latent, d_geo]."""
    def __init__(s):
        super().__init__()
        s.goto = nn.Sequential(nn.Linear(257, 256), nn.Tanh(), nn.Linear(256, 1))

    def forward(s, features):
        v_c = value_net(value_mlp(features[:, :257])).flatten()
        v_g = s.goto(torch.cat([features[:, :256], features[:, 258:259]], -1)).flatten()
        return torch.where(features[:, 257] > 0, v_g, v_c)


class GatedLoRA(nn.Module):
    def __init__(s, rank=16):
        super().__init__(); s.a = nn.Linear(256, rank, bias=False); s.b = nn.Linear(rank, 256, bias=False)
        nn.init.zeros_(s.b.weight)

    def forward(s, features):
        x = features[:, :256]
        return torch.tanh(policy_net[0](x) + features[:, 257:258] * s.b(s.a(x)))


f = torch.randn(512, 256 + EXTRA)
f[:, 257] = (torch.arange(512) % 2).float()          # gate
with torch.no_grad():
    tv, lora = TaskValue(), GatedLoRA()
    for p in list(tv.parameters()) + list(lora.parameters()): p.normal_(0, 0.1)   # trained
    c = f[:, 257] == 0
    print('TaskValue COMBAT rows == old value head bitwise', torch.equal(tv(f)[c], value_net(value_mlp(f[:, :257])).flatten()[c]))
    print('gated LoRA COMBAT rows == old policy_net bitwise', torch.equal(lora(f)[c], policy_net(f[:, :256])[c]))
print('TaskValue.goto params', sum(p.numel() for p in tv.goto.parameters()), ' LoRA r16 params',
      sum(p.numel() for p in lora.parameters()))
