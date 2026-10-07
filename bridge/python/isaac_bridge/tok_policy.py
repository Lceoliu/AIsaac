"""The token policy (EXPERIMENTS.md C55 on): one Transformer over the player, the room, the doors and every visible
entity; move, shoot and bomb heads and a value head read the player's token.

Tokens (tok_obs.ROW): the player (its state, the action already under way, and a small CNN over the 9 x 9 cell patch
around it), the room (a CNN over the whole grid), the minimap (a CNN over the 13 x 13 level grid; empty outside
whole-floor episodes), up to 8 doors, up to 64 entities (their features plus embeddings of type, variant and
subtype). Positions are relative to the player, so "what is close and where is it going" is a
property of single tokens and attention relates them; nothing is pooled before the Transformer.

Actions take effect one decision late (the sampler's rule): the action under way, `pending`, is part of the state.

Items (2026-10-06, Phase A; TokPolicy(items=True), stored in the checkpoint's config): two more heads, "use the active
item" (2) and "use the pill / card" (2), masked like the bomb head (use only when the active item is ready, only with a
pill or card in pocket slot 0) and biased to rarely act at the start; a pedestal's collectible id and a trinket's id
are embedded into its entity token; the player token also gets the held collectibles as one multi-hot vector (the sum
of their learned embeddings: no extra tokens, so the Transformer's cost and the CUDA graphs' shapes are unchanged),
the active item's, the trinkets', the pill's (color and, once identified, effect) and the card's embeddings, ROW
'pinv' and the two new heads' action under way. Every new input enters a first Linear as appended columns: a model
loaded from an items-free checkpoint with load_compatible (new columns zero) computes the same move, shoot, bomb
and value outputs as that checkpoint until training changes the new weights.

Charge (2026-10-07; TokPolicy(charge=True), stored in the checkpoint's config as 'charge'): ROW 'pcharge' (the charged
weapon's counter and the weapon-type bits, tok_obs.PCHARGE_F) as the player token's last inputs; a checkpoint without
them loads with load_compatible (new columns zero) and computes the same outputs. Off by default: checkpoints without
the key keep their shapes (train_tok --resume loads them strictly).
"""
import os

# One hardware work queue per process for the CUDA streams (train_tok sets the same before importing torch; it only
# takes effect if no CUDA context exists yet): it makes the GPU-sharing crash of replayed GraphActor graphs (illegal
# memory access, Xid 31 GCC FAULT_PDE at an address of the other process; see train_tok.main) about ten times rarer.
# It does not remove it: do not share the GPU with another CUDA process while GraphActor runs. An explicit
# CUDA_DEVICE_MAX_CONNECTIONS in the environment wins.
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tok_obs import (DOOR_CAP, DOOR_F, ENT_F, GRID, MAP, N_CARD, N_COLLECTIBLE, N_PILL_COLOR, N_PILL_EFFECT,
                      N_TRINKET, PATCH, PCHARGE_F, PINV_F, PLAYER_F, ROW)

N_MOVE, N_SHOOT, N_BOMB = 9, 5, 2
HEADS = (N_MOVE, N_SHOOT, N_BOMB)
NOOP = (0, 0, 0)
N_ITEM, N_PILL = 2, 2                      # items (2026-10-06): use the active item, use the pill / card
ITEM_HEADS = HEADS + (N_ITEM, N_PILL)
USE_BIAS = -4.0                            # a fresh item / pill head says "use" in about 2% of the decisions it may
ITEM_EMB, TRINKET_EMB, PILL_EMB, EFFECT_EMB, CARD_EMB = 16, 8, 4, 8, 8
# player-token inputs the items add: the two heads' action under way, pinv, held (multi-hot), active, two trinkets, pill
# color, pill effect, card
PLAYER_ITEMS_F = N_ITEM + N_PILL + PINV_F + ITEM_EMB + ITEM_EMB + 2 * TRINKET_EMB + PILL_EMB + EFFECT_EMB + CARD_EMB


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix64(z):
    """splitmix64's finaliser on a uint64 array (wrapping arithmetic)."""
    with np.errstate(over='ignore'):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def crn_uniforms(keys, t, columns):
    """Common random numbers for the policy's sampling (Phase B2, 2026-10-07): float32 [n, columns] uniforms that are a
    deterministic function of (key, decision index t, column), the column being the slot of the sampling uniforms (one
    per logit of every head, GraphActor.u's layout). Counter-based: element (key, t, c) is splitmix64's output number
    t * 256 + c + 1 of the stream seeded with mix(key); its top 23 bits k give (k + 0.5) / 2^23, exactly representable
    and inside (0, 1) (torch.rand's resolution is 2^-24). Different keys give independent streams; for a fixed key the
    values over (t, c) are i.i.d. uniform for every purpose of a sampler (abplus_probe_crn.py tests uniformity and
    independence)."""
    keys = np.asarray(keys, np.int64).astype(np.uint64)
    t = np.asarray(t, np.int64).astype(np.uint64)
    with np.errstate(over='ignore'):
        seed = _mix64(keys * _GOLDEN + np.uint64(0x632BE59BD9B4E019))
        ctr = (t[:, None] << np.uint64(8)) + np.arange(columns, dtype=np.uint64)[None, :] + np.uint64(1)
        z = _mix64(seed[:, None] + ctr * _GOLDEN)
    return (((z >> np.uint64(41)).astype(np.float64) + 0.5) / float(1 << 23)).astype(np.float32)


def gumbel_sample(logits_list, u):
    """Gumbel-max over each head's log-softmax with the uniforms u [n, sum of head sizes] (GraphActor's rule): the
    actions [n, heads] and their log-probability [n]."""
    actions, logp, at = [], 0.0, 0
    for head in logits_list:
        k = head.shape[1]
        logs = F.log_softmax(head, -1)
        gumbel = -torch.log(-torch.log(u[:, at:at + k].clamp(1e-10, 1 - 1e-7)))
        a = torch.argmax(logs + gumbel, -1)
        actions.append(a)
        logp = logp + logs.gather(-1, a[:, None])[:, 0]
        at += k
    return torch.stack(actions, 1), logp


def heads_of(model_or_config):
    """The action heads of a TokPolicy (or of its config dict)."""
    cfg = model_or_config if isinstance(model_or_config, dict) else model_or_config.config
    return ITEM_HEADS if cfg.get('items') else HEADS


def mlp(n_in, hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, n_out))


def _autocast_on():
    return torch.is_autocast_enabled()


def embed(seq, x):
    """An input mlp on raw float features. Under autocast (train_tok --amp) it runs in float32: the features
    (positions, velocities, ...) are not rounded to bfloat16's 8 significant bits, and with only its first Linear in
    float32 the PPO gradient of some minibatches deviated much more (cosine to float32 down to 0.93 instead of
    0.996). Without autocast it is seq(x), unchanged."""
    if not _autocast_on():
        return seq(x)
    with torch.autocast('cuda', enabled=False):
        return seq(x.float())


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.ff = mlp(width, 4 * width, width)

    def forward(self, x, bias):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.n2(x))

    def forward_packed(self, x, bias, keep, shape):
        """forward() on the real tokens only: x [m, d] holds the tokens at the flat positions `keep` of a [b, t]
        layout (shape (b, t)). The per-token parts (layer norms, qkv, projection, feed-forward) run on the m real
        tokens; attention runs in the padded layout with the same mask (padded keys get weight exactly 0, padded
        queries are dropped). The real tokens' values are those of forward() up to summation order."""
        b, t = shape
        m, d = x.shape
        qkv = self.qkv(self.n1(x))
        full = qkv.new_zeros((b * t, 3 * d)).index_copy(0, keep, qkv)
        q, k, v = full.view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.proj(a.transpose(1, 2).reshape(b * t, d).index_select(0, keep))
        return x + self.ff(self.n2(x))


CHOICE_EMB, CHOICE_HIDDEN = 16, 64
OPT_SKIP, OPT_TAKE, OPT_DOOR = 0, 1, 2


class ChoiceHeads(nn.Module):
    """Choice-value heads (Phase B2, 2026-10-07; TokPolicy(choice=K), config key 'choice'): K small MLPs g_k(h, e) on
    the decision record's player token h (the Transformer's output after the final norm, the input of the policy and
    value heads) and an option embedding e. Formulation: g_k is a per-option value up to a state-dependent constant, and
    only differences are trained: the prediction for a pair of options (a, b) is g_k(h, e_a) - g_k(h, e_b) and its label
    is the measured score difference of the pair's branches (tok_branch: discounted partial return + bootstrap). For the
    item kinds the pair is always (take, skip): e_take = take + item(collectible id), e_skip = skip, so the label is
    D = score(take) - score(skip) whichever option the episode itself chose (item: take was its own; item_left: skip
    was). Door options (e = door + a projection of the door's token) are defined but not trained yet (open issue).
    The K heads see different bootstrap weights of the records; their spread is the uncertainty the sampler's point
    priority uses (train_tok --branch-uncert). Trained on detached features with an optimiser of their own: the policy,
    the value and their optimiser are unchanged by them."""

    def __init__(self, width, k=4):
        super().__init__()
        self.k = int(k)
        self.item = nn.Embedding(N_COLLECTIBLE, CHOICE_EMB)
        self.opt = nn.Embedding(4, CHOICE_EMB)
        self.door = nn.Linear(width, CHOICE_EMB)
        self.heads = nn.ModuleList(mlp(width + CHOICE_EMB, CHOICE_HIDDEN, 1) for _ in range(self.k))

    def values(self, h, e):
        """[B, K] per-option values of option embeddings e [B, E] at player tokens h [B, W]."""
        x = torch.cat([h, e], -1)
        return torch.cat([m(x) for m in self.heads], -1)

    def take_minus_skip(self, h, item):
        """[B, K]: predicted score(take) - score(skip) of collectible `item` [B] (long) at h."""
        e_take = self.opt.weight[OPT_TAKE][None] + self.item(item)
        e_skip = self.opt.weight[OPT_SKIP][None].expand(h.shape[0], -1)
        return self.values(h, e_take) - self.values(h, e_skip)

    @staticmethod
    def nearest_pedestal(b):
        """(collectible id [B] long, has [B] bool) of the nearest pedestal holding an item in a batch dict (entity
        column 2: distance / 300, as tok_branch.pedestals)."""
        ids, it, ent = b['ent_id'], b['ent_item'], b['ent']
        e = ids.shape[1]
        real = torch.arange(e, device=ids.device)[None] < b['n_ent'][:, None]
        mask = real & (ids[..., 0] == 5) & (ids[..., 1] == 100) & (it[..., 0] > 0)
        dist = torch.where(mask, ent[..., 2].float(), torch.full_like(ent[..., 2].float(), 1e9))
        j = dist.argmin(1)
        item = it[..., 0].long().gather(1, j[:, None])[:, 0]
        return item, mask.any(1)


class TokPolicy(nn.Module):
    def __init__(self, width=192, layers=4, heads=4, items=False, charge=False, choice=0):
        super().__init__()
        self.config = dict(width=width, layers=layers, heads=heads)
        if items:   # (the key only when set: items-free checkpoints keep their config)
            self.config['items'] = True
        if charge:   # 2026-10-07: ROW 'pcharge' as the player token's last inputs (the key only when set, as items)
            self.config['charge'] = True
        if choice:   # 2026-10-07 (Phase B2): K choice-value heads (ChoiceHeads; the key only when set)
            self.config['choice'] = int(choice)
        self.items = bool(items)
        self.charge = bool(charge)
        self.heads = ITEM_HEADS if items else HEADS
        self.type_emb, self.variant_emb, self.subtype_emb = nn.Embedding(1024, 24), nn.Embedding(1024, 12), \
            nn.Embedding(256, 4)
        if items:   # id 0 = none: a zero vector
            self.item_emb = nn.Embedding(N_COLLECTIBLE, ITEM_EMB, padding_idx=0)
            self.trinket_emb = nn.Embedding(N_TRINKET, TRINKET_EMB, padding_idx=0)
            self.pill_emb = nn.Embedding(N_PILL_COLOR, PILL_EMB, padding_idx=0)
            self.effect_emb = nn.Embedding(N_PILL_EFFECT, EFFECT_EMB, padding_idx=0)
            self.card_emb = nn.Embedding(N_CARD, CARD_EMB, padding_idx=0)
        self.entity = mlp(ENT_F + 40 + ((ITEM_EMB + TRINKET_EMB) if items else 0), width, width)
        self.patch = nn.Sequential(nn.Conv2d(GRID[0], 32, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.Flatten(),
                                   nn.Linear(32 * PATCH * PATCH, 128), nn.GELU())
        self.player = mlp(PLAYER_F + sum(HEADS) + 128 + (PLAYER_ITEMS_F if items else 0) + (PCHARGE_F if charge else 0),
                          width, width)
        self.room = nn.Sequential(nn.Conv2d(GRID[0], 32, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                  nn.Linear(64 * ((GRID[1] + 3) // 4) * ((GRID[2] + 3) // 4), width))
        self.door = mlp(DOOR_F, width, width)
        self.map = nn.Sequential(nn.Conv2d(MAP[0], 32, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                 nn.Linear(64 * ((MAP[1] + 1) // 2) * ((MAP[2] + 1) // 2), width))
        self.kind = nn.Parameter(torch.zeros(5, width))   # player, room, door, entity, map
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.pi = nn.Linear(width, sum(self.heads))
        self.value = mlp(width, width, 1)
        self.danger = nn.Linear(width, N_MOVE)   # auxiliary: which moves get the player hurt (the search teacher)
        nn.init.normal_(self.kind, std=0.02)
        nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.pi.bias)
        with torch.no_grad():
            self.pi.bias[N_MOVE + N_SHOOT + 1] = -3.0   # a bomb is rare at the start: about 5% of decisions
            if items:   # so are an item's and a pill's use
                self.pi.bias[sum(HEADS) + 1] = USE_BIAS
                self.pi.bias[sum(HEADS) + N_ITEM + 1] = USE_BIAS
        nn.init.zeros_(self.danger.weight)
        nn.init.zeros_(self.danger.bias)
        # registered last: the parameters before it keep their order (an optimiser state of a run without it loads)
        self.choice = ChoiceHeads(width, choice) if choice else None

    def main_parameters(self):
        """The parameters PPO's optimiser owns: all but the choice heads' (in the order of parameters())."""
        return [p for n, p in self.named_parameters() if not n.startswith('choice.')]

    def forward(self, b, entities=None, packed=False, need_h=False):
        """b: the dict of to_batch. Returns (logits [B, 16], value [B], danger logits [B, 9]).
        packed: the Transformer's per-token layers on the real tokens only (Block.forward_packed; the padded door and
        entity tokens are skipped). Same values up to summation order, less work when many tokens are padding (the
        learner); not for CUDA-graph capture (the number of real tokens varies)."""
        n = b['player'].shape[0]
        used = entities or b['ent'].shape[1]
        ids = b['ent_id'][:, :used].long()
        emb = [self.type_emb(ids[..., 0]), self.variant_emb(ids[..., 1]), self.subtype_emb(ids[..., 2])]
        if self.items:
            it = b['ent_item'][:, :used].long()
            emb += [self.item_emb(it[..., 0]), self.trinket_emb(it[..., 1])]
        emb = torch.cat(emb, -1)
        entity = embed(self.entity, torch.cat([b['ent'][:, :used], emb], -1)) + self.kind[3]
        pending = b['pending']
        under_way = torch.cat([F.one_hot(pending[:, i], k) for i, k in enumerate(HEADS)], -1).float()
        patch = self.patch(b['patch'].float()).float()
        parts = [b['player'], under_way, patch]
        if self.items:   # appended: load_compatible pads these columns with zeros
            inv = b['inv'].long()
            pit = b['pitem'].long()
            parts += [F.one_hot(pending[:, 3], N_ITEM).float(), F.one_hot(pending[:, 4], N_PILL).float(),
                      b['pinv'], self.item_emb(inv[..., 0]).sum(1), self.item_emb(pit[:, 0]),
                      self.trinket_emb(pit[:, 1]), self.trinket_emb(pit[:, 2]), self.pill_emb(pit[:, 3]),
                      self.effect_emb(pit[:, 4]), self.card_emb(pit[:, 5])]
        if self.charge:   # appended last: load_compatible pads these columns with zeros
            parts.append(b['pcharge'])
        player = embed(self.player, torch.cat(parts, -1)) + self.kind[0]
        room = self.room(b['grid'].float()) + self.kind[1]
        door = embed(self.door, b['doors']) + self.kind[2]
        level = self.map(b['map'].float()) + self.kind[4]
        x = torch.cat([player[:, None], room[:, None], level[:, None], door, entity], 1)
        dev = x.device
        pad = torch.cat([torch.zeros((n, 3), dtype=torch.bool, device=dev),
                         torch.arange(DOOR_CAP, device=dev)[None] >= b['n_doors'][:, None],
                         torch.arange(used, device=dev)[None] >= b['n_ent'][:, None]], 1)
        bias = torch.zeros((n, 1, 1, x.shape[1]), device=dev, dtype=x.dtype).masked_fill(pad[:, None, None], -1e9)
        if packed:
            t = x.shape[1]
            if 'keep' in b:   # the real tokens' positions computed on the host (packed_index: no wait for the GPU)
                keep, starts = b['keep'], b['starts']
            else:
                real = ~pad
                keep = real.reshape(-1).nonzero().squeeze(1)
                counts = real.sum(1)
                starts = counts.cumsum(0) - counts
            xp = x.reshape(n * t, -1).index_select(0, keep)
            for block in self.blocks:
                xp = block.forward_packed(xp, bias, keep, (n, t))
            x0 = xp.index_select(0, starts)   # the player's token: the first of each record
        else:
            for block in self.blocks:
                x = block(x, bias)
            x0 = x[:, 0]
        if _autocast_on():   # the heads in float32 (logits and value not rounded to bfloat16)
            with torch.autocast('cuda', enabled=False):
                h = self.norm(x0.float())
                if need_h:   # (2026-10-07, Phase B2: the player token for the choice heads)
                    return self.pi(h), self.value(h)[:, 0], self.danger(h), h
                return self.pi(h), self.value(h)[:, 0], self.danger(h)
        h = self.norm(x0)
        if need_h:
            return self.pi(h), self.value(h)[:, 0], self.danger(h), h
        return self.pi(h), self.value(h)[:, 0], self.danger(h)

    @staticmethod
    def gate(b):
        """[B, 3] bool from a batch: may the bomb head say "bomb" (a bomb held), the item head "use" (the active item
        ready: ROW player column 22), the pill head "use" (a pill or a card in pocket slot 0: pinv columns 2, 4)."""
        bombs = b['bombs'] > 0
        if 'pinv' not in b:
            return torch.stack([bombs, bombs, bombs], -1)
        ready = b['player'][:, 22] > 0.5
        pocket = (b['pinv'][:, 2] > 0) | (b['pinv'][:, 4] > 0)
        return torch.stack([bombs, ready, pocket], -1)

    def split(self, logits, gate):
        """The heads' logits; without a bomb the bomb head can only say no (gate: [B, 3] from TokPolicy.gate, or the
        bombs [B] as before); with items the item head says "use" only when the active item is ready, the pill head
        only with a pill or card held."""
        if gate.dim() == 1:
            gate = (gate > 0)[:, None].expand(-1, 3)
        out = list(logits.split(self.heads, -1))
        for j, i in ((2, 0), (3, 1), (4, 2)):
            if j < len(out):
                h = out[j]
                out[j] = torch.stack([h[:, 0], h[:, 1].masked_fill(~gate[:, i], -1e9)], -1)
        return out

    def act(self, b, greedy=False, entities=None, u=None):
        """Actions [B, heads], their log-probability [B] and the value [B]. u (2026-10-07, Phase B2): sampling
        uniforms [B, sum of head sizes] for Gumbel-max (GraphActor's rule; the common random numbers of branch
        records); None: torch.multinomial as before."""
        logits, value, _ = self.forward(b, entities)
        if u is not None and not greedy:
            actions, logp = gumbel_sample(self.split(logits.float(), self.gate(b)), u)
            return actions, logp, value.float()
        actions, logp = [], 0.0
        for head in self.split(logits.float(), self.gate(b)):
            logs = F.log_softmax(head, -1)
            a = logs.argmax(-1) if greedy else torch.multinomial(logs.exp(), 1)[:, 0]
            actions.append(a)
            logp = logp + logs.gather(-1, a[:, None])[:, 0]
        return torch.stack(actions, 1), logp, value.float()

    def evaluate(self, b, actions, entities=None, packed=False):
        """Log-probability of the actions [B], entropy per head [B, heads], value [B], danger logits [B, 9]."""
        logits, value, danger = self.forward(b, entities, packed)
        logp, entropy = self.log_prob(logits, self.gate(b), actions)
        return logp, entropy, value.float(), danger.float()

    def log_prob(self, logits, gate, actions):
        """(log-probability of the actions [B], entropy per head [B, heads]) from forward()'s logits (evaluate's);
        gate: TokPolicy.gate of the batch (or its bombs [B])."""
        logp, entropy = 0.0, []
        for i, head in enumerate(self.split(logits.float(), gate)):
            logs = F.log_softmax(head, -1)
            logp = logp + logs.gather(-1, actions[:, i:i + 1])[:, 0]
            entropy.append(-(logs.exp() * logs.clamp(min=-50)).sum(-1))
        return logp, torch.stack(entropy, 1)


ENTITY_FIELDS = ('ent', 'ent_id', 'ent_item')   # per-entity batch fields (cut to the entity tokens used)


def used_entities(n_ent):
    """How many entity tokens a batch needs: the largest count, in steps of 8."""
    return max(8, -(-int(n_ent.max()) // 8) * 8)


def to_batch(rows, pending, device):
    """ROW records (a numpy structured array) and the actions under way [n, 3] as tensors on the device."""
    def t(field):
        return torch.from_numpy(np.ascontiguousarray(rows[field])).to(device, non_blocking=True)

    used = min(rows['ent'].shape[1], used_entities(rows['n_ent']))
    b = dict(player=t('player'), doors=t('doors'), patch=t('patch'), grid=t('grid'), map=t('map'), n_ent=t('n_ent'),
             n_doors=t('n_doors'), bombs=t('bombs'), inv=t('inv'), pitem=t('pitem'), pinv=t('pinv'),
             pcharge=t('pcharge'))
    for k in ENTITY_FIELDS:
        b[k] = torch.from_numpy(np.ascontiguousarray(rows[k][:, :used])).to(device, non_blocking=True)
    b['pending'] = torch.from_numpy(np.ascontiguousarray(pending)).to(device, non_blocking=True).long()
    return b


def take(b, index):
    """The rows `index` (a tensor) of a batch dict, with only the entity tokens they need."""
    out = {k: v[index] for k, v in b.items()}
    used = min(out['ent'].shape[1], used_entities(out['n_ent']))
    for k in ENTITY_FIELDS:
        if k in out:
            out[k] = out[k][:, :used]
    return out


def packed_index(n_ent, n_doors, entities):
    """forward(packed=True)'s token positions from the host's entity and door counts (numpy, one per record) for a
    batch with `entities` entity tokens: (keep, starts) int64, the flat positions of the real tokens in the [n, t]
    layout (t = 3 + DOOR_CAP + entities) and each record's first real token among them. Put into the batch dict as
    'keep' and 'starts' (tensors on its device) they spare forward() the wait for the GPU that finding them costs."""
    n = len(n_ent)
    real = np.ones((n, 3 + DOOR_CAP + entities), bool)
    real[:, 3:3 + DOOR_CAP] = np.arange(DOOR_CAP)[None] < np.asarray(n_doors)[:, None]
    real[:, 3 + DOOR_CAP:] = np.arange(entities)[None] < np.asarray(n_ent)[:, None]
    counts = real.sum(1)
    return np.flatnonzero(real), np.cumsum(counts) - counts


def cat_batch(a, b):
    """Two batch dicts as one, a's records first (the entity tokens padded to the wider of the two)."""
    e = max(a['ent'].shape[1], b['ent'].shape[1])
    out = {}
    for k, x in a.items():
        y = b[k]
        if k in ENTITY_FIELDS:
            x = F.pad(x, (0, 0, 0, e - x.shape[1])) if x.shape[1] < e else x
            y = F.pad(y, (0, 0, 0, e - y.shape[1])) if y.shape[1] < e else y
        out[k] = torch.cat([x, y.to(x.dtype)])
    return out


ROW_BYTES = ROW.itemsize          # a multiple of 8 (the record holds int64 fields and is aligned)
_INPUTS = ('player', 'ent', 'ent_id', 'doors', 'patch', 'grid', 'map', 'n_ent', 'n_doors', 'bombs', 'ent_item', 'inv',
           'pitem', 'pinv', 'pcharge')
_TORCH = {np.dtype(np.float32): torch.float32, np.dtype(np.int16): torch.int16, np.dtype(np.int32): torch.int32,
          np.dtype(np.uint8): torch.uint8}


def decode_rows(raw, pending=None, entities=None, n_heads=3):
    """The policy's inputs as views into ROW records held as bytes on the device: raw uint8 [n, >= ROW_BYTES]. The
    values are the records' own bits (what to_batch copies field by field). pending: the actions under way [n, heads]
    (a tensor), or None when raw carries them as n_heads int64 right after the record (GraphActor's layout).
    entities: entity tokens to keep (default: what the batch needs, as to_batch)."""
    n = raw.shape[0]
    b = {}
    for name in _INPUTS:
        dt, off = ROW.fields[name][:2]
        v = raw[:, off:off + dt.itemsize].view(_TORCH[dt.base])
        b[name] = v.view(n, *dt.shape) if dt.shape else v[:, 0]
    if pending is None:
        pending = raw[:, ROW_BYTES:ROW_BYTES + 8 * n_heads].view(torch.int64)
    b['pending'] = pending
    if entities is None:
        entities = min(b['ent'].shape[1], max(8, -(-int(b['n_ent'].max()) // 8) * 8))
    for k in ENTITY_FIELDS:
        b[k] = b[k][:, :entities]
    return b


def rows_to_device(rows, index, pending, device):
    """to_batch for a large set of records without numpy copies: rows a ROW array (any shape, C-contiguous), index the
    flat positions to take (numpy int64), pending [len(index), heads] int64. The gather runs in torch (no GIL held), one
    copy goes to the device and the fields are views of it (decode_rows). Same values as
    to_batch(rows.reshape(-1)[index], pending, device)."""
    raw = torch.from_numpy(rows.reshape(-1).view(np.uint8).reshape(-1, ROW_BYTES))
    picked = raw.index_select(0, torch.from_numpy(np.ascontiguousarray(index, np.int64)))
    dev_raw = picked.to(device, non_blocking=False)
    pend = torch.from_numpy(np.ascontiguousarray(pending, np.int64)).to(device)
    return decode_rows(dev_raw, pend)


class GraphActor:
    """TokPolicy.act (sampling) as captured CUDA graphs on fixed shapes: `rows_max` records with `entities` entity
    tokens each, and a second graph with all ENT_CAP tokens for the records with more entities. Eager PyTorch spends
    about 3 ms per call on launching the network's kernels one by one whatever the batch (EXPERIMENTS.md B10); a
    replayed graph launches them at once. The graphs read the model's parameters in place, so optimiser steps (or a
    copy of new weights into them) need no new capture. Sampling is Gumbel-max on uniforms drawn into a fixed tensor
    before each replay, which is the categorical distribution of the logits.
    One call moves the records as raw bytes: they are copied into a pinned staging buffer (with the actions under
    way behind each record), sent to the device in one transfer and decoded there inside the graph (decode_rows: the
    same bits as to_batch); the outputs come back through a pinned buffer. act() returns None only for more than
    rows_max records (the caller then uses the eager path).
    submit() / result() split a call so that the caller can work (or run another GraphActor on another stream) while
    the GPU computes; `stream`: the CUDA stream of the calls (None: the current stream at each call).
    """

    def __init__(self, model, rows_max, entities=32, stream=None, crn=True):
        from .tok_obs import ENT_CAP
        dev = next(model.parameters()).device
        self.model, self.n, self.stream, self._n = model, rows_max, stream, 0
        n = rows_max
        heads = model.heads
        nh = self.nh = len(heads)
        width = ROW_BYTES + 8 * nh
        self.stage = torch.zeros((n, width), dtype=torch.uint8).pin_memory()
        self.stage_np = self.stage.numpy()
        self.raw = torch.zeros((n, width), dtype=torch.uint8, device=dev)
        self.u = torch.rand((n, sum(heads)), device=dev)
        # 2026-10-07 (Phase B2): common random numbers. Records whose ROW 'crn' is not 0 (branch episodes) get the
        # uniforms crn_uniforms(crn, t, slot) instead of fresh ones: computed on the host, copied through a pinned
        # buffer and written into their rows of u after u.uniform_() and before the replay (u.uniform_ still fills
        # every row: the generator's draws, and so every other record's uniforms, are what they are without it).
        # crn=False: ignored (fresh uniforms for every record, as before).
        self.crn = bool(crn)
        self.crn_stage = torch.zeros((n, sum(heads)), dtype=torch.float32).pin_memory()
        self.crn_stage_np = self.crn_stage.numpy()
        self.crn_idx_stage = torch.zeros(n, dtype=torch.int64).pin_memory()
        self.crn_idx_np = self.crn_idx_stage.numpy()
        self.crn_u = torch.zeros((n, sum(heads)), dtype=torch.float32, device=dev)
        self.crn_idx = torch.zeros(n, dtype=torch.int64, device=dev)
        self.crn_rows = 0   # records sampled with common random numbers (all calls)
        # 2026-10-07 (Phase B2): with choice heads two more output columns per record: the heads' mean and spread
        # (standard deviation over the K heads) of score(take) - score(skip) for the nearest pedestal holding an item
        # (0, 0 without one): self.extra after result()
        self.choice = getattr(model, 'choice', None) is not None
        self.n_extra = 2 if self.choice else 0
        self.extra = None
        self.out_host = torch.zeros((n, nh + 2 + self.n_extra), dtype=torch.float32).pin_memory()
        self.out_np = self.out_host.numpy()
        self.done = torch.cuda.Event()
        self.sizes = sorted({min(int(entities), ENT_CAP), ENT_CAP})
        self.entities = self.sizes[0]

        def run(tokens):
            b = decode_rows(self.raw, entities=tokens, n_heads=nh)
            if self.choice:
                logits, value, _, h = model(b, tokens, need_h=True)
            else:
                logits, value, _ = model(b, tokens)
            actions, logp, at = [], 0.0, 0
            for head in model.split(logits.float(), model.gate(b)):
                k = head.shape[1]
                logs = F.log_softmax(head, -1)
                gumbel = -torch.log(-torch.log(self.u[:, at:at + k].clamp(1e-10, 1 - 1e-7)))
                a = torch.argmax(logs + gumbel, -1)
                actions.append(a)
                logp = logp + logs.gather(-1, a[:, None])[:, 0]
                at += k
            cols = [torch.stack(actions, 1).float(), logp[:, None], value.float()[:, None]]
            if self.choice:
                item, has = ChoiceHeads.nearest_pedestal(b)
                d = model.choice.take_minus_skip(h.float(), item).float()
                keep = has.float()[:, None]
                cols += [d.mean(1, keepdim=True) * keep, d.std(1, keepdim=True) * keep]
            return torch.cat(cols, 1)

        self.graphs, self.outs = [], []
        pool = None
        # captured on a stream of this actor's own: cuBLAS keeps one workspace per stream and a graph keeps the one of
        # the stream it was captured on. On torch.cuda.graph's default capture stream (shared by every graph) two
        # actors replayed at once (train_tok --actor-slots 2) shared one workspace and corrupted each other's
        # split-K products: the value head's output (seen as wrong stored values in about 9% of the decisions)
        capture = torch.cuda.Stream()
        with torch.inference_mode():
            for tokens in self.sizes:
                for _ in range(3):   # allocations and algorithm choices happen before the capture
                    run(tokens)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=capture):   # never replayed at once: one memory pool
                    out = run(tokens)
                pool = graph.pool()
                self.graphs.append(graph)
                self.outs.append(out)
        torch.cuda.synchronize()

    def submit(self, rows, pending):
        """Start a call (returns at once; False if it does not fit). rows: a C-contiguous ROW array, pending [n, heads]
        int64. The next submit may only follow this call's result()."""
        n = len(rows)
        if n > self.n:
            return False
        most = int(rows['n_ent'].max())
        g = 0 if most <= self.sizes[0] else len(self.sizes) - 1
        self.stage_np[:n, :ROW_BYTES] = np.ascontiguousarray(rows).view(np.uint8).reshape(n, ROW_BYTES)
        self.stage_np[:n, ROW_BYTES:] = np.ascontiguousarray(pending, np.int64).view(np.uint8).reshape(n, 8 * self.nh)
        stream = self.stream if self.stream is not None else torch.cuda.current_stream()
        sel = None
        if self.crn and 'crn' in rows.dtype.names:
            crn = rows['crn']
            if crn.any():
                sel = np.flatnonzero(crn)
                k = len(sel)
                self.crn_stage_np[:k] = crn_uniforms(crn[sel], rows['t'][sel], self.u.shape[1])
                self.crn_idx_np[:k] = sel
                self.crn_rows += k
        with torch.cuda.stream(stream):
            self.raw[:n].copy_(self.stage[:n], non_blocking=True)
            self.u.uniform_()
            if sel is not None:
                k = len(sel)
                self.crn_u[:k].copy_(self.crn_stage[:k], non_blocking=True)
                self.crn_idx[:k].copy_(self.crn_idx_stage[:k], non_blocking=True)
                self.u.index_copy_(0, self.crn_idx[:k], self.crn_u[:k])
            self.graphs[g].replay()
            self.out_host[:n].copy_(self.outs[g][:n], non_blocking=True)
            self.done.record()
        self._n = n
        return True

    def result(self):
        """(actions [n, heads] int64, log-probabilities [n], values [n]) of the submitted call, as numpy arrays."""
        self.done.synchronize()
        out = self.out_np[:self._n].copy()
        nh = self.nh
        self.extra = out[:, nh + 2:] if self.n_extra else None
        return out[:, :nh].astype(np.int64), out[:, nh], out[:, nh + 1]

    def act(self, rows, pending):
        """submit + result: (actions, log-probabilities, values), or None if it does not fit."""
        return self.result() if self.submit(rows, pending) else None


def load_compatible(model, state):
    """Load a checkpoint's weights into a model whose input layers may have grown (more door features, ...): a
    tensor of another shape is copied into the leading part of the model's (the rest keeps its fresh values, zero
    for added input columns). Returns the names that did not match exactly."""
    own = model.state_dict()
    changed = []
    for name, value in state.items():
        if name not in own:
            changed.append(name + ' (dropped)')
            continue
        if own[name].shape == value.shape:
            own[name] = value
            continue
        target = own[name].clone()
        if target.dim() == value.dim() and all(a >= b for a, b in zip(target.shape, value.shape)):
            if target.dim() == 2:   # a Linear's weight: new input columns start at zero
                target[:, value.shape[1]:] = 0
            target[tuple(slice(0, k) for k in value.shape)] = value
            own[name] = target
            changed.append(name + ' (padded)')
        else:
            changed.append(name + ' (kept fresh)')
    model.load_state_dict(own)
    return changed
