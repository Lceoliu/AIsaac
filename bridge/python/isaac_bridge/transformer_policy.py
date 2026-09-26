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
"""
import math

import torch
from torch import nn
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .transformer_obs import ANIMATION_BYTES, ENTITY_FIELDS, PLAYER_FIELDS


class CombatTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, layers=4, heads=8):
        spaces = observation_space.spaces
        # features_dim is the model width; fire_distance (combat-v5) rides along as one more output.
        value_extra = int('fire_distance' in spaces)
        super().__init__(observation_space, features_dim + value_extra)
        self.value_extra = value_extra
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
        self.entity = nn.Sequential(nn.Linear(len(ENTITY_FIELDS)+self.flag_dim+32+16+8+32, 128), nn.GELU(),
                                    nn.Linear(128, 128), nn.LayerNorm(128))
        self.queries = nn.Parameter(torch.randn(4, 128)*0.02)
        self.player_query = nn.Linear(64, 128)
        self.empty_entity = nn.Parameter(torch.zeros(1, 128))
        self.entity_attention = nn.MultiheadAttention(128, 4, dropout=0, batch_first=True)
        self.map_cnn = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                     nn.Flatten(), nn.Linear(32*5*8, 128), nn.GELU())
        if self.factored_previous:
            # One-hot of the factored heads (all zero before the first action).
            self.action = nn.Linear(spaces['previous_action'].shape[-1], 32)
        else:
            self.joint_action = nn.Embedding(45, 16)
            self.bomb_action = nn.Embedding(2, 4)
            self.item_action = nn.Embedding(2, 4)
            self.action = nn.Linear(25, 32)
        self.fusion = nn.Sequential(nn.Linear(64+512+128+32, features_dim),
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
        queries = self.queries.unsqueeze(0) + self.player_query(player).unsqueeze(1)
        summary, _ = self.entity_attention(queries, entities, entities, key_padding_mask=padding,
                                           need_weights=False)
        terrain = self.map_cnn(observations['terrain'][valid_time])
        actions = self.previous_action_features(observations['previous_action'][valid_time])
        fused = self.fusion(torch.cat([player, summary.flatten(1), terrain, actions], -1))
        if self.value_extra:
            fused = torch.cat([fused, observations['fire_distance'][valid_time].unsqueeze(-1)], -1)
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
        queries = self.queries.unsqueeze(0) + self.player_query(player).unsqueeze(1)
        summary, _ = self.entity_attention(queries, entities, entities, key_padding_mask=padding, need_weights=False)
        terrain = self.map_cnn(o['terrain'])
        actions = self.previous_action_features(o['previous_action'])
        fused = self.fusion(torch.cat([player, summary.flatten(1), terrain, actions], -1))
        if self.value_extra:
            fused = torch.cat([fused, o['fire_distance'].unsqueeze(-1)], -1)
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
    sees all, including CombatTransformer's critic-only fire_distance)."""

    def __init__(self, feature_dim, net_arch, activation_fn, device, actor_dims):
        super().__init__()
        pi, vf = (net_arch.get('pi', []), net_arch.get('vf', [])) if isinstance(net_arch, dict) else (net_arch, net_arch)

        def mlp(inputs, sizes):
            layers = []
            for size in sizes:
                layers += [nn.Linear(inputs, size), activation_fn()]
                inputs = size
            return nn.Sequential(*layers).to(device), inputs

        self.actor_dims = actor_dims
        self.policy_net, self.latent_dim_pi = mlp(actor_dims, pi)
        self.value_net, self.latent_dim_vf = mlp(feature_dim, vf)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_dims])

    def forward_critic(self, features):
        return self.value_net(features)


# Auxiliary geometry head (combat-v5): logits of the shoot action that hits an aligned target (5,
# 0 = none) and d_fire / 40, both from the actor's features (abplus_geometry labels).
AUX_OUTPUTS = 6


class GeometryPolicy(MaskableMultiInputActorCriticPolicy):
    """Maskable policy with SplitMlpExtractor and the auxiliary geometry head (aux_head)."""

    def _build_mlp_extractor(self):
        actor_dims = getattr(self.features_extractor, 'actor_dims', self.features_dim)
        self.mlp_extractor = SplitMlpExtractor(self.features_dim, self.net_arch, self.activation_fn, self.device,
                                               actor_dims)
        self.aux_head = nn.Sequential(nn.Linear(actor_dims, 128), nn.GELU(), nn.Linear(128, AUX_OUTPUTS))

    def aux_outputs(self, features):
        """(aim logits (n, 5), d_fire / 40 prediction (n,)) from full features (critic input dropped)."""
        out = self.aux_head(features[..., :self.mlp_extractor.actor_dims])
        return out[..., :5], out[..., 5]
