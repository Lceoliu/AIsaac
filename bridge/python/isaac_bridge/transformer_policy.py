"""Entity-set attention, spatial CNN, and a finite causal Transformer for PPO.

History contains raw observations, not stale embeddings from an older policy.
Every minibatch re-encodes the complete window with current weights. No RNN,
future frames, or cross-worker KV state are used.

combat-v5 observations (VisibleHistory(geometry=True, factored_actions=True)) add the per-NPC
lineage/blocking flags to the entity tokens and give the previous action as the factored heads'
one-hot. Their fire_distance (d_fire / 40) is the critic's input for the alignment potential: it
bypasses the Transformer and is appended to the features as the last dimension, which
SplitMlpExtractor gives to the value branch only, so the actor and the auxiliary geometry head
(GeometryPolicy) must find the firing geometry themselves.

Size (user plan 2026-09-28, a larger distillation student): features_dim is the model width (fusion output
and Transformer), layers / heads the temporal Transformer, entity_queries the learned queries of the
entity-set attention (each gives one entity_dim summary of the frame's entities; fusion takes all of them),
entity_dim the entity token width. The defaults are the architecture of every run up to C37.

Goal-conditioned line (rl/docs/GOAL_CONDITIONED_DESIGN.md, user decisions 2026-09-30), active only when the observations have
'goal' (VisibleHistory(goal=True)); every other checkpoint builds and computes exactly as before. gate = the frame's task is not
COMBAT (goal one-hot slots 1..7). The goal (GOAL_FIELDS) is embedded by goal_encoder (h_g, 64) and added, only where gate is
set (torch.where, so a COMBAT row takes the unchanged value bit for bit), in three places: to the entity-attention queries
(goal_query), to the terrain CNN's first convolution (goal_conv over goal_map: the goal cell and the open doors) and before
the fusion LayerNorm (goal_fusion, the main path). The three are zero-initialised (GeometryPolicy re-zeroes them after SB3's
orthogonal init), so a policy migrated from a checkpoint without them computes that checkpoint's function on every input.
Past the Transformer the features carry the frame's goal and goal_distance (critic only) next to fire_distance. The actor's
logits get a gated residual, l = l_base(z) + 1_GOTO * dl_nav(z, goal) (NavActionNet, last layer zero), and the value a GOTO
head reading the temporal latent, the goal and the walking distance directly (TaskValueNet), next to the unchanged COMBAT
value.

C44 (goal-hp2, design 2.2 item 5): with 'room_bits' in the observations a full-room branch (room_cnn over the whole room
on the 16 x 28 canvas, 8 bit planes) adds room_fusion(...) to the fusion pre-activation of every frame; room_fusion starts
at zero (re-zeroed after SB3's init as the goal injections), so a checkpoint migrated into it computes its own function.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .transformer_obs import ANIMATION_BYTES, ENTITY_FIELDS, GOAL_SLOTS, PLAYER_FIELDS


# The goal line's modules (after SB3's features_extractor. / pi_features_extractor. / vf_features_extractor. prefixes): the
# parameters a checkpoint from before them lacks (training_session.warm_start_model) and C44's 'goal' optimizer group.
GOAL_LINE_MODULES = ('goal_encoder.', 'goal_query.', 'goal_conv.', 'goal_fusion.', 'action_net.nav.', 'value_net.goto.',
                     'room_cnn.', 'room_fusion.')
ROOM_MODULES = ('room_cnn.', 'room_fusion.')
_PREFIXES = ('features_extractor.', 'pi_features_extractor.', 'vf_features_extractor.', '')


def module_name(key, modules):
    """True when a parameter / state key belongs to one of `modules` (GOAL_LINE_MODULES, ROOM_MODULES)."""
    return any(key.startswith(prefix) and key[len(prefix):].startswith(modules) for prefix in _PREFIXES)


def goal_gate(goal):
    """True where a goal vector's task is not COMBAT (one-hot slots 1..GOAL_SLOTS-1)."""
    return goal[..., 1:GOAL_SLOTS].sum(-1) > 0


class CombatTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, layers=4, heads=8, entity_queries=4, entity_dim=128):
        spaces = observation_space.spaces
        # features_dim is the model width; fire_distance (combat-v5) rides along as one more output, and in the goal line
        # the frame's goal and goal_distance after it (goal_dim + 1 more).
        self.fire_extra = int('fire_distance' in spaces)
        self.goal_dim = spaces['goal'].shape[-1] if 'goal' in spaces else 0
        value_extra = self.fire_extra + (self.goal_dim + 1 if self.goal_dim else 0)
        super().__init__(observation_space, features_dim + value_extra)
        self.value_extra = value_extra
        # what the COMBAT critic MLP reads: the model width and fire_distance, as every checkpoint before the goal line
        self.critic_dims = features_dim + self.fire_extra
        self.model_dim = self.actor_dims = features_dim
        self.flag_dim = spaces['entity_flags'].shape[-1] if 'entity_flags' in spaces else 0
        self.factored_previous = spaces['previous_action'].shape[-1] != 4
        self.character = nn.Embedding(256, 4, padding_idx=0)
        self.animation = nn.Linear(ANIMATION_BYTES*4, 32)
        self.entity_type = nn.Embedding(1024, 32)
        self.variant = nn.Embedding(8192, 16)
        self.subtype = nn.Embedding(4096, 8)
        self.active_item = nn.Embedding(4096, 16)
        self.has_deadline='remaining_time' in observation_space.spaces
        # combat-v3 room state (transformer_obs.COMBAT_FIELDS), part of the player token.
        self.combat_dim=observation_space.spaces['combat'].shape[-1] if 'combat' in observation_space.spaces else 0
        self.player = nn.Sequential(nn.Linear(len(PLAYER_FIELDS)+32+16+int(self.has_deadline)+self.combat_dim, 64), nn.GELU(),
                                    nn.Linear(64, 64), nn.LayerNorm(64))
        self.entity = nn.Sequential(nn.Linear(len(ENTITY_FIELDS)+self.flag_dim+32+16+8+32, entity_dim), nn.GELU(),
                                    nn.Linear(entity_dim, entity_dim), nn.LayerNorm(entity_dim))
        self.queries = nn.Parameter(torch.randn(entity_queries, entity_dim)*0.02)
        self.player_query = nn.Linear(64, entity_dim)
        self.empty_entity = nn.Parameter(torch.zeros(1, entity_dim))
        self.entity_attention = nn.MultiheadAttention(entity_dim, 4, dropout=0, batch_first=True)
        # The terrain canvas (rows, cols): 9x15 (5x8 after the stride), C41's 16x28 (8x14).
        rows, cols = spaces['terrain'].shape[-2:]
        self.map_cnn = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                     nn.Flatten(), nn.Linear(32*((rows+1)//2)*((cols+1)//2), 128), nn.GELU())
        if self.factored_previous:
            # One-hot of the factored heads (all zero before the first action).
            self.action = nn.Linear(spaces['previous_action'].shape[-1], 32)
        else:
            self.joint_action = nn.Embedding(45, 16)
            self.bomb_action = nn.Embedding(2, 4)
            self.item_action = nn.Embedding(2, 4)
            self.action = nn.Linear(25, 32)
        self.fusion = nn.Sequential(nn.Linear(64+entity_queries*entity_dim+128+32, features_dim),
                                    nn.LayerNorm(features_dim), nn.GELU())
        # Construct independently, rather than cloning one identically initialized layer.
        # Dropout=0 keeps rollout and PPO likelihood evaluation consistent.
        self.temporal = nn.ModuleList([
            nn.TransformerEncoderLayer(features_dim, heads, features_dim*4,
                                       dropout=0, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.final_norm = nn.LayerNorm(features_dim)
        frequency = torch.exp(torch.arange(0, features_dim, 2)*(-math.log(10000)/features_dim))
        self.register_buffer('frequency', frequency)
        if self.goal_dim:
            # Registered after every existing module: the existing parameters keep their order.
            self.goal_encoder = nn.Sequential(nn.Linear(self.goal_dim, 64), nn.GELU(), nn.Linear(64, 64), nn.LayerNorm(64))
            self.goal_query = nn.Linear(64, entity_dim)
            self.goal_conv = nn.Conv2d(2, 16, 3, padding=1, bias=False)
            self.goal_fusion = nn.Linear(64, features_dim)
            self.zero_goal_injections()
        # C44 (goal-hp2, design 2.2 item 5): the full-room branch over room_bits (ROOM_CANVAS, ROOM_BITS unpacked to 8
        # channels): 3 convolutions (16 x 28 -> 8 x 14 -> 4 x 7), FC 896 -> 128, then room_fusion (zero at the start)
        # added to the fusion's pre-activation for every frame: a migrated checkpoint computes its own function until
        # the branch learns. Registered last: every existing parameter keeps its order.
        self.room_dim = int('room_bits' in spaces)
        if self.room_dim:
            rows, cols = spaces['room_bits'].shape[-2:]
            self.room_cnn = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.GELU(),
                                          nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                          nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
                                          nn.Flatten(), nn.Linear(32 * ((rows + 3) // 4) * ((cols + 3) // 4), 128), nn.GELU())
            self.room_fusion = nn.Linear(128, features_dim)
            self.register_buffer('room_shifts', torch.arange(8), persistent=False)
            self.zero_room_injection()

    GOAL_INJECTIONS = ('goal_query', 'goal_conv', 'goal_fusion')

    def zero_goal_injections(self):
        """The layers adding the goal to the existing computation start at zero (the migrated function is unchanged)."""
        with torch.no_grad():
            for name in self.GOAL_INJECTIONS:
                for p in getattr(self, name).parameters():
                    p.zero_()

    def goal_injections_zero(self):
        return all(not p.any() for name in self.GOAL_INJECTIONS for p in getattr(self, name).parameters())

    def zero_room_injection(self):
        with torch.no_grad():
            for p in self.room_fusion.parameters():
                p.zero_()

    def room_injection_zero(self):
        return not self.room_dim or all(not p.any() for p in self.room_fusion.parameters())

    def _room(self, bits):
        """room_fusion(room_cnn(bits)) of per-frame room_bits (n, rows, cols), or None without the branch."""
        if not self.room_dim:
            return None
        planes = ((bits.long().unsqueeze(1) >> self.room_shifts.view(1, -1, 1, 1)) & 1).to(self.room_fusion.weight.dtype)
        return self.room_fusion(self.room_cnn(planes))

    def _goal_parts(self, goal):
        """(gate (n,), h_g (n, 64)) of per-frame goal vectors, or (None, None) outside the goal line."""
        if not self.goal_dim:
            return None, None
        return goal_gate(goal), self.goal_encoder(goal)

    def _queries(self, player, gate, h_goal):
        queries = self.queries.unsqueeze(0) + self.player_query(player).unsqueeze(1)
        if gate is None:
            return queries
        return torch.where(gate[:, None, None], queries + self.goal_query(h_goal).unsqueeze(1), queries)

    def _terrain(self, terrain, goal_map, gate):
        if gate is None:
            return self.map_cnn(terrain)
        first = self.map_cnn[0](terrain)
        first = torch.where(gate[:, None, None, None], first + self.goal_conv(goal_map), first)
        return self.map_cnn[1:](first)

    def _fuse(self, inputs, gate, h_goal, room=None):
        if gate is None and room is None:
            return self.fusion(inputs)
        pre = self.fusion[0](inputs)
        if gate is not None:
            pre = torch.where(gate[:, None], pre + self.goal_fusion(h_goal), pre)
        if room is not None:
            pre = pre + room
        return self.fusion[2](self.fusion[1](pre))

    def _extras(self, fused, o, select=None):
        """Per-frame critic-only / head inputs appended after the model width (they bypass the Transformer)."""
        parts = [fused]
        pick = (lambda v: v[select]) if select is not None else (lambda v: v)
        if self.fire_extra:
            parts.append(pick(o['fire_distance']).unsqueeze(-1))
        if self.goal_dim:
            parts += [pick(o['goal']), pick(o['goal_distance']).unsqueeze(-1)]
        return torch.cat(parts, -1) if len(parts) > 1 else fused

    def animation_embedding(self, values):
        return self.animation(self.character(values.long()).flatten(-2))

    def previous_action_features(self, old):
        if self.factored_previous:
            return self.action(old)
        return self.action(torch.cat([self.joint_action(old[:, 0].long()), self.bomb_action(old[:, 1].long()),
                                      self.item_action(old[:, 2].long()), old[:, 3:4]], -1)) * old[:, 3:4]

    def encode_frames(self, observations):
        valid_time = observations['history_mask'].bool()
        batch, length = valid_time.shape
        # Do not run the CNN or entity encoder on padded times/entities.
        p = observations['player'][valid_time]
        animation = self.animation_embedding(observations['player_anim'][valid_time])
        active = self.active_item(observations['active_kind'][valid_time].long().squeeze(-1))
        inputs=[p,animation,active]
        if self.has_deadline:inputs.append(observations['remaining_time'][valid_time].unsqueeze(-1))
        if self.combat_dim:inputs.append(observations['combat'][valid_time])
        player = self.player(torch.cat(inputs, -1))
        mask = observations['entity_mask'][valid_time].bool()
        features = observations['entities'][valid_time][mask]
        if self.flag_dim:
            features = torch.cat([features, observations['entity_flags'][valid_time][mask]], -1)
        kinds = observations['entity_kind'][valid_time][mask].long()
        anim = self.animation_embedding(observations['entity_anim'][valid_time][mask])
        encoded = self.entity(torch.cat([features, self.entity_type(kinds[:, 0]),
                                         self.variant(kinds[:, 1]), self.subtype(kinds[:, 2]), anim], -1))
        gate, h_goal = self._goal_parts(observations['goal'][valid_time]) if self.goal_dim else (None, None)
        counts = mask.sum(-1)
        maximum = int(counts.max().item())
        # A learned null entity permits valid attention when the room has no entities.
        entities = self.empty_entity.expand(len(player), maximum+1, -1).clone()
        padding = torch.ones((len(player), maximum+1), dtype=torch.bool, device=player.device)
        padding[:, 0] = False
        rows, _ = mask.nonzero(as_tuple=True)
        ranks = (mask.long().cumsum(-1))[mask]
        entities[rows, ranks] = encoded
        padding[rows, ranks] = False
        queries = self._queries(player, gate, h_goal)
        summary, _ = self.entity_attention(queries, entities, entities, key_padding_mask=padding,
                                           need_weights=False)
        terrain = self._terrain(observations['terrain'][valid_time],
                                observations['goal_map'][valid_time] if self.goal_dim else None, gate)
        actions = self.previous_action_features(observations['previous_action'][valid_time])
        room = self._room(observations['room_bits'][valid_time]) if self.room_dim else None
        fused = self._fuse(torch.cat([player, summary.flatten(1), terrain, actions], -1), gate, h_goal, room)
        if self.value_extra:
            fused = self._extras(fused, observations, valid_time)
        sequence = fused.new_zeros(batch, length, self.features_dim)
        sequence[valid_time] = fused
        return sequence

    def encode_frame_static(self, o):
        """encode_frames for one frame per row, (n, ...) inputs, without data-dependent shapes.

        All 256 entity slots go through the entity MLP and the padding ones are masked out of the
        attention instead of being compacted away, so no host synchronisation is needed and the
        computation can be captured in a CUDA graph (graph_sampler.py). Same result as
        encode_frames up to float summation order; rows whose history_mask is 0 give zeros.
        """
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
        padding = torch.cat([torch.zeros((n, 1), dtype=torch.bool, device=player.device), ~o['entity_mask'].bool()], 1)
        gate, h_goal = self._goal_parts(o['goal']) if self.goal_dim else (None, None)
        queries = self._queries(player, gate, h_goal)
        summary, _ = self.entity_attention(queries, entities, entities, key_padding_mask=padding, need_weights=False)
        terrain = self._terrain(o['terrain'], o['goal_map'] if self.goal_dim else None, gate)
        actions = self.previous_action_features(o['previous_action'])
        room = self._room(o['room_bits']) if self.room_dim else None
        fused = self._fuse(torch.cat([player, summary.flatten(1), terrain, actions], -1), gate, h_goal, room)
        if self.value_extra:
            fused = self._extras(fused, o)
        return fused * o['history_mask'].unsqueeze(-1)

    def sequence_features(self, observations):
        sequence = self.encode_frames(observations)
        return self.temporal_features(sequence, observations['time'], observations['history_mask'])

    def temporal_features(self, sequence, elapsed, history_mask):
        """Temporal stage shared by full-window training and frozen-frame sampling."""
        # combat-v5: the per-frame critic input (last dimension) skips the Transformer.
        extra = sequence[..., self.model_dim:]
        sequence = sequence[..., :self.model_dim]
        # Relative elapsed time, no global engine frame/seed as a memorization cue.
        time = (elapsed-elapsed[:, :1])*15
        angles = time.unsqueeze(-1) * self.frequency
        position = torch.stack([angles.sin(), angles.cos()], -1).flatten(-2)
        sequence = sequence + position
        length = sequence.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=sequence.device).triu(1)
        padding = ~history_mask.bool()
        for layer in self.temporal:
            sequence = layer(sequence, src_mask=causal, src_key_padding_mask=padding)
        sequence = self.final_norm(sequence)
        return torch.cat([sequence, extra], -1) if self.value_extra else sequence

    def forward(self, observations):
        sequence = self.sequence_features(observations)
        last = observations['history_mask'].long().sum(-1)-1
        return sequence[torch.arange(sequence.shape[0], device=sequence.device), last]


class SplitMlpExtractor(nn.Module):
    """SB3 MlpExtractor whose actor branch sees only the first actor_dims features (the value branch
    sees all, including CombatTransformer's critic-only fire_distance).

    Goal line (goal_dims > 0): the value MLP still reads the first critic_dims features (model width + fire_distance, as
    before); both latents are extended for the task heads: actor [policy_net(z), z, goal], critic [value_net(...), z, goal,
    goal_distance] (NavActionNet / TaskValueNet split them; latent_dim_pi / latent_dim_vf stay the MLP widths)."""

    def __init__(self, feature_dim, net_arch, activation_fn, device, actor_dims, critic_dims=None, goal_dims=0):
        super().__init__()
        self.critic_dims = feature_dim if critic_dims is None else critic_dims
        self.goal_dims = goal_dims
        pi, vf = (net_arch.get('pi', []), net_arch.get('vf', [])) if isinstance(net_arch, dict) else (net_arch, net_arch)

        def mlp(inputs, sizes):
            layers = []
            for size in sizes:
                layers += [nn.Linear(inputs, size), activation_fn()]
                inputs = size
            return nn.Sequential(*layers).to(device), inputs

        self.actor_dims = actor_dims
        self.policy_net, self.latent_dim_pi = mlp(actor_dims, pi)
        self.value_net, self.latent_dim_vf = mlp(self.critic_dims, vf)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def _goal(self, features):
        start = self.critic_dims
        return features[..., start:start + self.goal_dims], features[..., start + self.goal_dims:start + self.goal_dims + 1]

    def forward_actor(self, features):
        latent = self.policy_net(features[..., :self.actor_dims])
        if not self.goal_dims:
            return latent
        goal, _ = self._goal(features)
        return torch.cat([latent, features[..., :self.actor_dims], goal], -1)

    def forward_critic(self, features):
        if not self.goal_dims:
            return self.value_net(features)
        goal, distance = self._goal(features)
        return torch.cat([self.value_net(features[..., :self.critic_dims]), features[..., :self.actor_dims], goal, distance],
                         -1)


class NavActionNet(nn.Linear):
    """The action logits of SB3's action_net (the same weight / bias, so checkpoints load), plus on an extended latent
    [latent_pi, z, goal] the gated navigation residual: base(latent_pi) + dl_nav(z, goal) where goal's task is not COMBAT
    (torch.where: a COMBAT row is base bit for bit). The residual's last layer starts at zero."""

    def __init__(self, in_features, out_features, model_dim, goal_dim, hidden=256):
        super().__init__(in_features, out_features)
        self.model_dim, self.goal_dim = model_dim, goal_dim
        self.nav = nn.Sequential(nn.Linear(model_dim + goal_dim, hidden), nn.GELU(), nn.Linear(hidden, out_features))
        self.zero_residual()

    def zero_residual(self):
        with torch.no_grad():
            self.nav[-1].weight.zero_()
            self.nav[-1].bias.zero_()

    def forward(self, latent):
        if latent.shape[-1] == self.in_features:
            return super().forward(latent)
        base = F.linear(latent[..., :self.in_features], self.weight, self.bias)
        rest = latent[..., self.in_features:]
        goal = rest[..., self.model_dim:self.model_dim + self.goal_dim]
        delta = self.nav(rest[..., :self.model_dim + self.goal_dim])
        return torch.where(goal_gate(goal)[..., None], base + delta, base)


class TaskValueNet(nn.Linear):
    """SB3's value_net (the same weight / bias: the COMBAT value) plus a GOTO value head on an extended latent
    [latent_vf, z, goal, goal_distance]: where the goal's task is not COMBAT, goto([z, goal, goal_distance]) reads the
    temporal latent and the goal directly (user decision 2026-09-30); a COMBAT row is the unchanged value."""

    def __init__(self, in_features, model_dim, goal_dim, hidden=256):
        super().__init__(in_features, 1)
        self.model_dim, self.goal_dim = model_dim, goal_dim
        self.goto = nn.Sequential(nn.Linear(model_dim + goal_dim + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, latent):
        if latent.shape[-1] == self.in_features:
            return super().forward(latent)
        combat = F.linear(latent[..., :self.in_features], self.weight, self.bias)
        rest = latent[..., self.in_features:]
        goal = rest[..., self.model_dim:self.model_dim + self.goal_dim]
        return torch.where(goal_gate(goal)[..., None], self.goto(rest), combat)


# Auxiliary geometry head (combat-v5): logits of the shoot action that hits an aligned target (5,
# 0 = none) and d_fire / 40, both from the actor's features (abplus_geometry labels).
AUX_OUTPUTS = 6


class GeometryPolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable policy with SplitMlpExtractor and the auxiliary geometry head (aux_head). In the goal line (the features
    extractor has goal_dim) action_net / value_net are NavActionNet / TaskValueNet (module docstring)."""

    def __init__(self, *args, lr_groups=False, **kwargs):
        # C44 (design 4.8): the optimizer gets two named parameter groups, 'shared' (every parameter from before the goal
        # line) and 'goal' (GOAL_LINE_MODULES), each with its own learning rate (gpu_ppo.GroupLearningRate). A policy
        # argument, so a checkpoint rebuilds the same optimizer and its state loads.
        self.lr_groups = bool(lr_groups)
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        actor_dims = getattr(self.features_extractor, 'actor_dims', self.features_dim)
        goal_dim = getattr(self.features_extractor, 'goal_dim', 0)
        self.mlp_extractor = SplitMlpExtractor(self.features_dim, self.net_arch, self.activation_fn, self.device,
                                               actor_dims, getattr(self.features_extractor, 'critic_dims', None) if goal_dim
                                               else None, goal_dim)
        self.aux_head = nn.Sequential(nn.Linear(actor_dims, 128), nn.GELU(), nn.Linear(128, AUX_OUTPUTS))

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        fe = self.features_extractor
        if not getattr(fe, 'goal_dim', 0):
            return
        # The goal-line heads replace SB3's (same weight names, so every existing key loads), after its orthogonal init;
        # the injections are zeroed again (the init overwrote them) and the optimizer is rebuilt over every parameter.
        base, value = self.action_net, self.value_net
        self.action_net = NavActionNet(base.in_features, base.out_features, fe.model_dim, fe.goal_dim).to(base.weight.device)
        self.value_net = TaskValueNet(value.in_features, fe.model_dim, fe.goal_dim).to(value.weight.device)
        with torch.no_grad():
            self.action_net.weight.copy_(base.weight)
            self.action_net.bias.copy_(base.bias)
            self.value_net.weight.copy_(value.weight)
            self.value_net.bias.copy_(value.bias)
        fe.zero_goal_injections()
        if getattr(fe, 'room_dim', 0):
            fe.zero_room_injection()   # C44: the full-room branch joins at zero too
        self.action_net.zero_residual()
        if self.lr_groups:
            named = list(self.named_parameters())
            groups = [dict(params=[q for n, q in named if not module_name(n, GOAL_LINE_MODULES)], name='shared'),
                      dict(params=[q for n, q in named if module_name(n, GOAL_LINE_MODULES)], name='goal')]
            self.optimizer = self.optimizer_class(groups, lr=lr_schedule(1), **self.optimizer_kwargs)
        else:
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def aux_outputs(self, features):
        """(aim logits (n, 5), d_fire / 40 prediction (n,)) from full features (critic input dropped)."""
        out = self.aux_head(features[..., :self.mlp_extractor.actor_dims])
        return out[..., :5], out[..., 5]
