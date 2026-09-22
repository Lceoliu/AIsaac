"""Entity-set attention, spatial CNN, and a finite causal Transformer for PPO.

History contains raw observations, not stale embeddings from an older policy.
Every minibatch re-encodes the complete window with current weights. No RNN,
future frames, privileged critic inputs, or cross-worker KV state are used.
"""
import math

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .transformer_obs import ANIMATION_BYTES, ENTITY_FIELDS, PLAYER_FIELDS


class CombatTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, layers=4, heads=8):
        super().__init__(observation_space, features_dim)
        self.character = nn.Embedding(256, 4, padding_idx=0)
        self.animation = nn.Linear(ANIMATION_BYTES*4, 32)
        self.entity_type = nn.Embedding(1024, 32)
        self.variant = nn.Embedding(8192, 16)
        self.subtype = nn.Embedding(4096, 8)
        self.active_item = nn.Embedding(4096, 16)
        self.player = nn.Sequential(nn.Linear(len(PLAYER_FIELDS)+32+16, 64), nn.GELU(),
                                    nn.Linear(64, 64), nn.LayerNorm(64))
        self.entity = nn.Sequential(nn.Linear(len(ENTITY_FIELDS)+32+16+8+32, 128), nn.GELU(),
                                    nn.Linear(128, 128), nn.LayerNorm(128))
        self.queries = nn.Parameter(torch.randn(4, 128)*0.02)
        self.player_query = nn.Linear(64, 128)
        self.empty_entity = nn.Parameter(torch.zeros(1, 128))
        self.entity_attention = nn.MultiheadAttention(128, 4, dropout=0, batch_first=True)
        self.map_cnn = nn.Sequential(nn.Conv2d(7, 16, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
                                     nn.Flatten(), nn.Linear(32*5*8, 128), nn.GELU())
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

    def encode_frames(self, observations):
        valid_time = observations['history_mask'].bool()
        batch, length = valid_time.shape
        # Do not run the CNN or entity encoder on padded times/entities.
        p = observations['player'][valid_time]
        animation = self.animation_embedding(observations['player_anim'][valid_time])
        active = self.active_item(observations['active_kind'][valid_time].long().squeeze(-1))
        player = self.player(torch.cat([p, animation, active], -1))
        mask = observations['entity_mask'][valid_time].bool()
        features = observations['entities'][valid_time][mask]
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
        old_action = observations['previous_action'][valid_time]
        actions = self.action(torch.cat([self.joint_action(old_action[:, 0].long()),
            self.bomb_action(old_action[:, 1].long()), self.item_action(old_action[:, 2].long()),
            old_action[:, 3:4]], -1))
        actions = actions * old_action[:, 3:4]
        fused = self.fusion(torch.cat([player, summary.flatten(1), terrain, actions], -1))
        sequence = fused.new_zeros(batch, length, self.features_dim)
        sequence[valid_time] = fused
        return sequence

    def sequence_features(self, observations):
        sequence = self.encode_frames(observations)
        return self.temporal_features(sequence, observations['time'], observations['history_mask'])

    def temporal_features(self, sequence, elapsed, history_mask):
        """Temporal stage shared by full-window training and frozen-frame sampling."""
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
        return self.final_norm(sequence)

    def forward(self, observations):
        sequence = self.sequence_features(observations)
        last = observations['history_mask'].long().sum(-1)-1
        return sequence[torch.arange(sequence.shape[0], device=sequence.device), last]
