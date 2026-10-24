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

Entity extension (2026-10-08; TokPolicy(ent_ext=True), config key 'ent_ext'): the entity columns tok_obs.ENT_F0 ..
ENT_F (the flag columns of tears / projectiles / lasers and a laser's geometry, abp-0.2.16) as the entity mlp's last
inputs, after the embeddings; without it the entity token reads columns 0 .. ENT_F0 - 1 only, as before. A checkpoint
without them loads with load_compatible (new columns zero) and computes the same outputs whatever the new columns hold.

Character (2026-10-08, character randomisation; TokPolicy(pchar=True), config key 'pchar'): ROW 'pchar' (player 0's
PlayerType, bridge abp-0.2.17) selects a row of a learned table of tok_obs.N_CHAR vectors (char_emb, zero at the start)
that is added to the player mlp's first Linear output, before its GELU: the same function as N_CHAR one-hot input
columns of that Linear. A checkpoint without it loads with load_compatible (char_emb stays zero) and computes exactly
the same outputs whatever character the records hold (one-hot columns padded into the Linear gave the same values only
up to summation order, about 3e-5 on the logits: abplus_probe_charge_compat.py). char_emb is registered after every
other parameter (their order, and an optimiser state of a run without it, unchanged).

Memory (2026-10-11, rl/docs/SCALING_THESIS.md S5b "记忆"; TokPolicy(memory=D), config key 'memory'): a GRUCell(width, D)
reads the player token after the final norm (z, the heads' input) and the hidden state h of the step before; the heads
then read z + mem_out(h_new) (mem_out: Linear(D, width), zero at the start). That is the same function as
concatenating h_new to every head's input with zero-initialised new columns, but one projection serves the policy,
value, danger and choice heads and their shapes stay as they are: a model resumed from a memory-free checkpoint (the GRU
fresh, mem_out zero) computes exactly that checkpoint's outputs until training moves mem_out (and through it the GRU).
h is per environment (worker) and carried by the caller: forward / act / evaluate take mem = h [B, D] (one GRU step per
record; h_new is returned last by forward / act), or for the learner a dict of chunks (h0 [C, D] the stored state at
each chunk's first record, T, reset [C, T] episode starts inside the chunk, optional tail [R, D] for R more records
after the C x T chunk records, one step each): truncated BPTT over the chunk. mem=None on a memory model = h of zeros
(an episode's start) and the output arity of a memory-free model. The memory modules are registered last (after
char_emb): the parameters before them keep their order. Without memory nothing here changes.

Extra tokens (2026-10-12, B21 item tokens / B22 global token; B24 one attention; B33 no gates; TokPolicy(item_tokens=
True) needs items=True, TokPolicy(global_token=True); config keys 'item_tokens', 'global_token', only when set). A
fixed number of ordinary tokens per record (static shapes for the CUDA graphs) appended to the old ones (player, room,
map, doors, entities): [..., items, global] (_forward_fused). Every block runs ONE attention over all of them (one
softmax; the same LayerNorm, q / k / v projections, output projection and feed-forward as every token):
  B21 (ItemTokens): one per held collectible (ROW 'inv', newest first, the newest item_slots of them: a collectible
  table of its own (dim 32), the count and count > 1, a slot embedding), one for the active item (its id in the same
  table, 'pinv' columns 0-1: charge share, max charge / 12), one per trinket (2), one for the pocket (pill colour, pill
  effect, card); each token = mlp(those inputs) + a type embedding (held, active, trinket, pocket). Absent ones are
  masked keys (-1e9 in the attention bias, as absent entities). The old multi-hot / pinv inputs of the player token
  stay.
  B22 (GlobalToken, 1): a learned vector; a second value head `value_global` reads it (after a norm of its own; with
  memory + mem_out(h_new) as the other heads); with memory the GRU's input is z + mem_in(norm(global token)) (mem_in
  zero at the start, as mem_out).
The heads read the player token. The learner's packed layouts treat the present item tokens like entities
(ItemTokens(compact=True) moves them to the front, item_count() gives their number on the host, packed_index /
learner_plan / plan_tokens take it). item_slots N (TokPolicy(item_slots=N), config key 'item_slots' when N != INV_CAP;
needs item tokens): the newest N held collectibles get tokens (slot embeddings 0 .. N-1 of the INV_CAP table: the
parameters keep their shapes). The modules are registered after everything else (memory included): an optimiser state
without them is remapped by parameter names (remap_optimizer_state). B33 (2026-10-16): B21 / B22's separate gated
terms (Block._ext), B24's per-layer gates (value multipliers) and the conversion between the two layouts are gone;
checkpoints with gates are run from sandboxes at commit b0a89b9 or earlier.

Shared collectible table (2026-10-13, B23; TokPolicy(item_shared=True), needs item_tokens, config key 'item_shared'):
ItemTokens reads the policy's item_emb (ITEM_EMB = 16 columns, the pedestal ids' and the multi-hot's table) instead of
a table of its own (IT_COLL = 32): one representation of a collectible whether it lies on a pedestal or is held. The
table is not registered a second time (one parameter, one state-dict key). A B21 checkpoint is converted by
share_item_table (its own table dropped, the item-token mlp's first Linear keeps its 44 other columns and gets fresh
collectible columns); off: the B21 module (tests/test_pickup_smoke.py).

Temporal attention (2026-10-14, B30 / B31, B33 no gates; rl/docs/TEMPORAL_ATTENTION.md; TokPolicy(temporal=True,
temporal_width=256, ...), config keys 'temporal', 'temporal_width', 'temporal_layers', 'temporal_heads',
'temporal_window', 'temporal_rooms' only when on). TemporalAttention (registered after everything else: an optimiser
state without it is remapped by name) is a pre-LN transformer of width D over one token per stored decision:
s_t = W_s [z_t ; norm_g(yg_t)] (z_t = norm(player token), yg_t the global token after the blocks, without one W_s z_t).
A step token attends to the room tokens created so far in its episode and to the window's step tokens <= t (W steps
including itself, a learned per-head bias over the step distance 0 .. W-1; rel[:, W] is the bias of a room key); a room
token r_k = W_r o_exit + E_room[k] + E_stage[stage] (o_exit: o of the last step in the room left, k: the episode's k-th
room, < temporal_rooms; past the cap no more are made) attends to the earlier room tokens and itself only, so its K / V
per layer are computed once, when it is made. Output o_t = norm(x_t). Readout: the heads read z (+ mem_out(h)) + P(o_t),
value_global norm_g(yg) (+ mem_out(h)) + P_g(o_t). Feedback (B31): a history token u_t = Q(o_{t-1}) + kind_hist is one
more ordinary token of the one attention, [..., items, history, global] (a temporal model without item / global tokens
uses the same layout for it), its key masked where there is no previous step (an episode's first step). P, P_g and Q
are ordinary Linear layers (torch's default initialisation): the temporal side is part of the network from the first
step. Room changes: a record whose map shows a current room (ROW map channel 3) other than
the last one shown in the episode, or with EV_EXIT / EV_NEW_ROOM (TemporalAttention.step; never at an episode's first
step; whole-floor / run modes only: room-mode records have no map).
Actor (TemporalState, per worker slot, plus a scratch slot for padded graph rows): one K / V tensor
[layers, 2, slots + 1, heads, rooms + window, dh] (rooms first, then the step ring) in kv_dtype (bfloat16 on the GPU),
step counts, room counts, the previous o / stage / map; reset by an episode's first record. Per call (step): the room
query (made only where a room changes) and the step query of each record go through the layers together; each layer
writes their K / V into the cache in place and ONE scaled_dot_product_attention over all slots reads the cache tensor
directly (queries of the other slots are zeros, their rows discarded): no window is gathered or concatenated. The step
runs outside torch.compile (a graph break) and is captured in GraphActor's CUDA graphs like the rest.
Learner (sequence): chunks of up to temporal_bptt consecutive stored decisions of one worker; per chunk the stored K / V
of the window - 1 decisions before it (stop-grad), the episode's room tokens made before the chunk (stop-grad: B31,
their per-layer K / V computed once per PPO epoch for every room of the rollout, room_table, and gathered per chunk)
and those made inside the chunk (recomputed WITH gradient through W_r, the embeddings and the layers, from the stored
o_{t-1}: v1 approximation, see TEMPORAL_ATTENTION.md); block-causal attention with the same masks and relative bias as
the actor. The history token uses the stored o_{t-1} (stop-grad).
Storage (B31): the rollout's per-decision K / V, s and o stay on the GPU (train_tok.Rollout's device tensors, written
by GraphActor / the eager step with index_copy_, gathered by temporal_inputs / temporal_part); only counters cross the
bus (per record: its rollout row in, create / nroom / steps / stage_prev out).
Without the GRU (B32, 2026-10-15; TokPolicy(memory=0, temporal=True)): the same model minus mem_out(h) and mem_in;
the learner's chunks are the temporal chunks alone (temporal_chunks(..., T_mem=None)); drop_memory converts a
memory + temporal checkpoint (the GRU's tensors dropped, every other one exact).

Observation v2 (2026-10-16, B35, bridge abp-0.2.18; train_tok --obs-v2 1 sets all three; config keys only when on, a
checkpoint without them builds the model of before, bit for bit):
  obs_prices (key 'obs_prices'): tok_obs.price_columns(ROW 'ent_price') (coin price / 15, heart containers 0 / 1 / 2,
    three soul hearts 0 / 1, for sale) as the entity mlp's last PRICE_F inputs (after the ent_ext columns);
  obs_hearts (key 'obs_hearts'): ROW 'phearts' (black half hearts / 12, eternal, bone hearts / 6) as the player mlp's
    last PHEART_F inputs (after pcharge);
  obs_room_type (key 'obs_room_type'): a learned table of N_ROOM_TYPE vectors (room_type_emb, zero at the start,
    registered after everything else) indexed by ROW 'room_type' and added to the room token.
A checkpoint without them loads with load_compatible (new Linear columns zero, room_type_emb zero) and computes the
same outputs up to the first Linears' summation order (tests/test_obs_v2_smoke.py).
"""
import contextlib
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

from .tok_obs import (DOOR_CAP, DOOR_F, ENT_F, ENT_F0, EV_EXIT, EV_NEW_ROOM, GRID, INV_CAP, MAP, N_CARD, N_CHAR,
                      N_COLLECTIBLE, N_PILL_COLOR, N_PILL_EFFECT, N_ROOM_TYPE, N_TRINKET, PATCH, PCHARGE_F, PHEART_F,
                      PINV_F, PLAYER_F, PRICE_F, ROW, price_columns)

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
# 2026-10-12 (B21 item tokens): an item token's inputs: collectible (held or active), trinket, pill colour, pill effect,
# card embeddings, then count, count > 1, charge share, max charge / 12 (zero where a kind has none)
IT_COLL, IT_TRINKET, IT_POCKET = 32, 16, 8
IT_F = IT_COLL + IT_TRINKET + 3 * IT_POCKET + 4
N_ITEM_TOK = INV_CAP + 4   # held collectibles, the active item, two trinkets, the pocket


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


@contextlib.contextmanager
def _strict_fp32():
    """cuBLAS float32 products without TF32 inside (the flag is process-wide: set and restored around the calls)."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class _MathAttention(torch.autograd.Function):
    """softmax(q k^T / sqrt(d) + bias) v as batched matrix products in strict float32, forward and backward.

    2026-10-10 (B18, the learner's fast path): F.scaled_dot_product_attention on float32 inputs with a mask runs
    PyTorch's memory-efficient kernel (CUTLASS, float32 SIMT); on the learner's minibatches (4,096 records x 4 heads x
    ~50-70 tokens, head size 48) its forward and backward were a quarter of the GPU time of an update (H20). The same
    function as plain cuBLAS products (TF32 off, so float32 like that kernel) and PyTorch's softmax is 1.3x cheaper at
    51 tokens and 2x at 67 (past 64 that kernel computes a second block); with the length buckets far more. The saved
    softmax costs [b * heads, t, t] floats of memory per layer. Same values up to summation order
    (abplus_bench_learner.py --check, abplus_check_learner_fp64.py)."""

    @staticmethod
    def forward(ctx, q, k, v, bias):
        b, h, t, dh = q.shape
        scale = dh ** -0.5
        q_, k_, v_ = (x.reshape(b * h, t, dh) for x in (q, k, v))
        mask = bias.expand(b, h, 1, t).reshape(b * h, 1, t)
        with _strict_fp32():
            p = torch.softmax(torch.baddbmm(mask, q_, k_.transpose(1, 2), alpha=scale), -1)
            out = torch.bmm(p, v_)
        ctx.save_for_backward(q_, k_, v_, p)
        ctx.scale, ctx.shape = scale, (b, h, t, dh)
        return out.view(b, h, t, dh)

    @staticmethod
    def backward(ctx, grad):
        q_, k_, v_, p = ctx.saved_tensors
        b, h, t, dh = ctx.shape
        g = grad.reshape(b * h, t, dh)
        with _strict_fp32():
            dv = torch.bmm(p.transpose(1, 2), g)
            ds = torch._softmax_backward_data(torch.bmm(g, v_.transpose(1, 2)), p, -1, p.dtype)
            dq = torch.bmm(ds, k_).mul_(ctx.scale)
            dk = torch.bmm(ds.transpose(1, 2), q_).mul_(ctx.scale)
        return dq.view(b, h, t, dh), dk.view(b, h, t, dh), dv.view(b, h, t, dh), None


class _LayerNorm(torch.autograd.Function):
    """F.layer_norm over the last dimension of x [m, d], with the weight and bias gradients as plain column sums.

    2026-10-10 (B18, the learner's fast path): PyTorch's layer-norm backward computes them in a kernel of its own
    (GammaBetaBackward) that took ~3 ms per learner step on the packed tokens (~50,000 x 192; H20), several times its
    memory traffic. Same forward kernel, same input gradient kernel; the weight / bias gradients sum the same terms
    (gy * normalised x, gy) in another order."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        y, mean, rstd = torch.native_layer_norm(x, (x.shape[-1],), weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, gy):
        x, weight, mean, rstd = ctx.saved_tensors
        gx = torch.ops.aten.native_layer_norm_backward(gy, x, (x.shape[-1],), mean, rstd, weight, None,
                                                       [True, False, False])[0]
        return gx, (gy * ((x - mean) * rstd)).sum(0), gy.sum(0), None


def layer_norm(norm, x, fast=False):
    """norm(x) (an nn.LayerNorm); fast (B18, float32 2-D input, not under autocast): _LayerNorm, the same function."""
    if fast and x.dim() == 2 and x.dtype == torch.float32 and not _autocast_on():
        return _LayerNorm.apply(x, norm.weight, norm.bias, norm.eps)
    return norm(x)


class _BucketAttention(torch.autograd.Function):
    """_MathAttention for every length bucket of a layer in one autograd node (B18, the learner's fast path).

    full [P, 3 * width]: the layer's q, k, v rows in the buckets' layout (Block.forward_packed with learner_plan's
    gpos); buckets [(records, t, key mask [records, 1, 1, t])]. Returns the attention outputs [P, width] in the same
    layout (rows of padded positions: unused). The same products and softmax per bucket as _MathAttention, without the
    autograd nodes of the per-bucket views and copies (host time per step)."""

    @staticmethod
    def forward(ctx, full, heads, buckets):
        rows, d3 = full.shape
        d = d3 // 3
        dh = d // heads
        scale = dh ** -0.5
        out = full.new_empty((rows, d))
        saved, sizes, at = [], [], 0
        with _strict_fp32():
            for nb, t, mask in buckets:
                blk = full[at:at + nb * t].view(nb, t, 3, heads, dh)
                q_, k_, v_ = (blk[:, :, i].transpose(1, 2).reshape(nb * heads, t, dh) for i in range(3))
                mask_ = mask.expand(nb, heads, 1, t).reshape(nb * heads, 1, t)
                p = torch.softmax(torch.baddbmm(mask_, q_, k_.transpose(1, 2), alpha=scale), -1)
                out[at:at + nb * t].view(nb, t, heads, dh).copy_(
                    torch.bmm(p, v_).view(nb, heads, t, dh).transpose(1, 2))
                saved += [q_, k_, v_, p]
                sizes.append((nb, t))
                at += nb * t
        ctx.save_for_backward(*saved)
        ctx.sizes, ctx.heads, ctx.scale = sizes, heads, scale
        return out

    @staticmethod
    def backward(ctx, grad):
        saved, heads, scale = ctx.saved_tensors, ctx.heads, ctx.scale
        grad = grad.contiguous()
        rows, d = grad.shape
        dh = d // heads
        out = grad.new_zeros((rows, 3 * d))
        at = 0
        with _strict_fp32():
            for j, (nb, t) in enumerate(ctx.sizes):
                q_, k_, v_, p = saved[4 * j:4 * j + 4]
                g = grad[at:at + nb * t].view(nb, t, heads, dh).transpose(1, 2).reshape(nb * heads, t, dh)
                dv = torch.bmm(p.transpose(1, 2), g)
                ds = torch._softmax_backward_data(torch.bmm(g, v_.transpose(1, 2)), p, -1, p.dtype)
                dq = torch.bmm(ds, k_).mul_(scale)
                dk = torch.bmm(ds.transpose(1, 2), q_).mul_(scale)
                blk = out[at:at + nb * t].view(nb, t, 3, heads, dh)
                for i, x in enumerate((dq, dk, dv)):
                    blk[:, :, i].copy_(x.view(nb, heads, t, dh).transpose(1, 2))
                at += nb * t
        return out, None, None


def attention(q, k, v, bias, math=False):
    """F.scaled_dot_product_attention(q, k, v, attn_mask=bias); math (B18, float32 only, not under autocast):
    _MathAttention, the same function."""
    if math and q.dtype == torch.float32 and not _autocast_on():
        return _MathAttention.apply(q, k, v, bias)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.ff = mlp(width, 4 * width, width)
        self.source_pointwise_recompute = False
        self.math = False      # 2026-10-10 (B18): attention() as _MathAttention (TokPolicy.set_math_attention)
        self.ln_fast = False   # (B18) the packed layer norms as _LayerNorm

    def forward(self, x, bias):
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = attention(q, k, v, bias, self.math)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.feed_forward(self.n2(x))

    def feed_forward(self, x):
        if self.source_pointwise_recompute:
            from .source_rematerialize import source_ff
            return source_ff(self.ff, x)
        return self.ff(x)

    def forward_packed(self, x, bias, keep, shape):
        """forward() on the real tokens only: x [m, d] holds the tokens at the flat positions `keep` of a [b, t]
        layout (shape (b, t)). The per-token parts (layer norms, qkv, projection, feed-forward) run on the m real
        tokens; attention runs in the padded layout with the same mask (padded keys get weight exactly 0, padded
        queries are dropped). The real tokens' values are those of forward() up to summation order.
        shape a dict (2026-10-10, B18, the learner's fast path; TokPolicy.forward with 'buckets'): attention runs per
        bucket of records with similar real-token counts, each record's real tokens compacted to the front of a
        [nb, t_bucket] layout (keys past its count masked): the same attention per record (a padded key's weight is
        exactly 0 either way) on far fewer padded positions; its cost grows with t^2 and jumps past 64 tokens."""
        m, d = x.shape
        qkv = self.qkv(layer_norm(self.n1, x, self.ln_fast))
        if isinstance(shape,dict) and 'cu_seqlens' in shape:
            # Native PyTorch 2.7 varlen FlashAttention has registered autograd.
            # Real tokens are already packed by record: no bucket padding or
            # key mask, and no replay/custom backward implementation.
            q,k,v=qkv.view(m,3,self.heads,d//self.heads).unbind(1)
            cu,maximum=shape['cu_seqlens'],shape['max_seqlen']
            a=torch.ops.aten._flash_attention_forward(q,k,v,cu,cu,maximum,maximum,0.,False,False)[0]
            x=x+self.proj(a.reshape(m,d))
            return x+self.feed_forward(layer_norm(self.n2,x,self.ln_fast))
        if isinstance(shape, dict):   # 2026-10-10 (B18): attention per length bucket (learner_plan)
            gpos = shape['gpos']
            full = qkv.new_zeros((shape['total'], 3 * d)).index_copy(0, gpos, qkv)
            if self.math and full.dtype == torch.float32 and not _autocast_on():   # (one node for all buckets)
                a = _BucketAttention.apply(full, self.heads, shape['buckets']).index_select(0, gpos)
                x = x + self.proj(a)
                return x + self.feed_forward(layer_norm(self.n2, x, self.ln_fast))
            outs, at = [], 0
            for nb, t, bias_k in shape['buckets']:
                q, k, v = full[at:at + nb * t].view(nb, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
                outs.append(attention(q, k, v, bias_k, self.math).transpose(1, 2).reshape(nb * t, d))
                at += nb * t
            a = (torch.cat(outs) if len(outs) > 1 else outs[0]).index_select(0, gpos)
            x = x + self.proj(a)
            return x + self.feed_forward(layer_norm(self.n2, x, self.ln_fast))
        b, t = shape
        full = qkv.new_zeros((b * t, 3 * d)).index_copy(0, keep, qkv)
        q, k, v = full.view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = attention(q, k, v, bias, self.math)
        x = x + self.proj(a.transpose(1, 2).reshape(b * t, d).index_select(0, keep))
        return x + self.feed_forward(layer_norm(self.n2, x, self.ln_fast))


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


class ItemTokens(nn.Module):
    """B21 (2026-10-12; TokPolicy(item_tokens=True)): N_ITEM_TOK tokens per record from ROW 'inv', 'pitem', 'pinv' (the
    module docstring) and their key mask.
    B23 (2026-10-13; TokPolicy(item_shared=True)): coll = the policy's item_emb (ITEM_EMB = 16 columns, the table of the
    pedestals' ids and the multi-hot), passed in and not registered here (one parameter, one state-dict key
    'item_emb.weight'); the mlp's first Linear then reads ITEM_EMB collectible columns instead of IT_COLL."""

    def __init__(self, width, coll=None, slots=INV_CAP):
        super().__init__()
        if not 1 <= int(slots) <= INV_CAP:
            raise ValueError(f'ItemTokens: slots must be in 1 .. {INV_CAP}, not {slots}')
        self.slots = int(slots)            # B24: held-collectible token slots (newest first; INV_CAP = as before)
        self.n_tok = self.slots + 4        # + the active item, two trinkets, the pocket
        self.shared = coll is not None
        if self.shared:   # (not a submodule: nn.Module.__setattr__ would register it a second time)
            self.__dict__['_shared_coll'] = coll
        else:
            self.coll = nn.Embedding(N_COLLECTIBLE, IT_COLL, padding_idx=0)
        self.dc = coll.embedding_dim if self.shared else IT_COLL   # the collectible columns of an item token's input
        self.trinket = nn.Embedding(N_TRINKET, IT_TRINKET, padding_idx=0)
        self.pill = nn.Embedding(N_PILL_COLOR, IT_POCKET, padding_idx=0)
        self.effect = nn.Embedding(N_PILL_EFFECT, IT_POCKET, padding_idx=0)
        self.card = nn.Embedding(N_CARD, IT_POCKET, padding_idx=0)
        self.mlp = mlp(IT_F - IT_COLL + self.dc, width, width)
        self.slot = nn.Embedding(INV_CAP, width)   # a held collectible's position in 'inv' (newest first)
        self.kind = nn.Embedding(4, width)         # held, active, trinket, pocket
        nn.init.normal_(self.slot.weight, std=0.02)
        nn.init.normal_(self.kind.weight, std=0.02)

    @property
    def coll(self):
        """The collectible table: its own (B21) or the policy's item_emb (B23 item_shared)."""
        own = self._modules.get('coll')
        return own if own is not None else self.__dict__['_shared_coll']

    def forward(self, b, compact=False):
        """(tokens [n, n_tok, width], present [n, n_tok] bool). No host synchronisation, static shapes. n_tok =
        N_ITEM_TOK unless slots (B24 --item-slots) keeps fewer held collectibles (the newest ones: ROW 'inv' is newest
        first). compact (B24): the present tokens moved to the front of each record, in their order (the absent
        ones after them; the attention does not depend on the order of its keys): the real item tokens are the first
        item_count() of a record, as the real entities are the first n_ent (the packed layouts plan them from that
        count on the host)."""
        inv = b['inv'] if self.slots == INV_CAP else b['inv'][:, :self.slots]
        inv, pit, pinv = inv.long(), b['pitem'].long(), b['pinv'].float()
        n, S = inv.shape[0], inv.shape[1]
        hid, cnt = inv[..., 0].clamp(0, N_COLLECTIBLE - 1), inv[..., 1]
        coll, dc = self.coll, self.dc

        def zf(k, w):
            return pinv.new_zeros((n, k, w))
        cf = cnt.float()[..., None]
        held = torch.cat([coll(hid), zf(S, IT_TRINKET + 3 * IT_POCKET), cf, (cf > 1).float(),
                          zf(S, 2)], -1)
        active = torch.cat([coll(pit[:, 0].clamp(0, N_COLLECTIBLE - 1))[:, None],
                            zf(1, IT_TRINKET + 3 * IT_POCKET + 2), pinv[:, None, 0:2]], -1)
        trink = torch.cat([zf(2, dc), self.trinket(pit[:, 1:3].clamp(0, N_TRINKET - 1)),
                           zf(2, 3 * IT_POCKET + 4)], -1)
        pocket = torch.cat([zf(1, dc + IT_TRINKET), self.pill(pit[:, 3].clamp(0, N_PILL_COLOR - 1))[:, None],
                            self.effect(pit[:, 4].clamp(0, N_PILL_EFFECT - 1))[:, None],
                            self.card(pit[:, 5].clamp(0, N_CARD - 1))[:, None], zf(1, 4)], -1)
        x = embed(self.mlp, torch.cat([held, active, trink, pocket], 1))
        kw = self.kind.weight   # (slices, no index tensors: nothing is copied from the host inside a CUDA graph)
        sw = self.slot.weight if S == INV_CAP else self.slot.weight[:S]
        pos = torch.cat([sw + kw[0:1], kw[1:2], kw[2:3], kw[2:3], kw[3:4]])
        present = torch.cat([(hid > 0) & (cnt > 0), pit[:, 0:1] > 0, pit[:, 1:3] > 0,
                             ((pit[:, 3] > 0) | (pit[:, 5] > 0))[:, None]], 1)
        if not compact:
            return x + pos, present
        # (B24) rank of each token: present ones 0 .. c-1 in order, absent ones c .. n_tok-1 (a permutation per record)
        c = present.long()
        rank = torch.where(present, c.cumsum(1) - 1, c.sum(1, keepdim=True) + (1 - c).cumsum(1) - 1)
        y = x + pos
        y = torch.zeros_like(y).scatter(1, rank[..., None].expand(-1, -1, y.shape[-1]), y)
        return y, torch.arange(present.shape[1], device=present.device)[None] < c.sum(1, keepdim=True)


def item_count(rows, slots=INV_CAP):
    """(B24) The present item tokens of ROW records (numpy, host): ItemTokens' present mask summed per record (held
    collectibles among the first `slots` of 'inv', the active item, two trinkets, the pocket). int64 [n]."""
    inv, pit = rows['inv'][:, :slots], rows['pitem']
    held = ((inv[..., 0] > 0) & (inv[..., 1] > 0)).sum(1)
    pocket = (pit[:, 3] > 0) | (pit[:, 5] > 0)
    return (held + (pit[:, 0] > 0) + (pit[:, 1] > 0) + (pit[:, 2] > 0) + pocket).astype(np.int64)


class GlobalToken(nn.Module):
    """B22 (2026-10-12; TokPolicy(global_token=True)): a learned token, the second value head value_global on it (own
    norm) and, with memory, mem_in (zero at the start: the GRU's input is z + mem_in(norm(global token)))."""

    def __init__(self, width, memory=0):
        super().__init__()
        self.tok = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.tok, std=0.02)
        self.norm = nn.LayerNorm(width)
        self.value = mlp(width, width, 1)
        self.mem_in = nn.Linear(width, width) if memory else None
        if memory:
            nn.init.zeros_(self.mem_in.weight)
            nn.init.zeros_(self.mem_in.bias)


T_NEG = -1e9          # (B30) a masked key's additive bias (finite: every query keeps at least one real key)
N_STAGE_EMB = 16      # (B30) stage embeddings of the room tokens (ROW 'stage' clamped to 0 .. 15)


def _no_compile(fn):
    """(B30) fn left out of torch.compile (a graph break: its in-place state writes run as plain kernels, captured in
    GraphActor's CUDA graphs like the compiled parts around them)."""
    dis = getattr(getattr(torch, 'compiler', None), 'disable', None)
    return dis(fn) if dis is not None else fn


@contextlib.contextmanager
def _sdpa_cache_backends():
    """(B30) the actor's cache attention on PyTorch's memory-efficient kernel (reads the bfloat16 cache in place,
    float32 accumulation and softmax inside), math where it cannot run (CPU); also inside --actor-sdpa-math's context
    (the math backend would materialise the logits of the whole cache)."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        ctx = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    except (ImportError, TypeError):
        ctx = contextlib.nullcontext()
    with ctx:
        yield


def _attend(q, k, v, bias):
    """(B30, learner) softmax(q k^T / sqrt(dh) + bias) v with plain autograd (the bias, and through it the relative
    position table, gets its gradient). q [C, Tq, H, dh], k / v [C, Tk, H, dh], bias [C, H or 1, Tq, Tk];
    returns [C, Tq, H * dh]."""
    C, Tq, H, dh = q.shape
    q_, k_, v_ = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    p = torch.softmax(torch.matmul(q_, k_.transpose(-1, -2)) * dh ** -0.5 + bias, -1)
    return torch.matmul(p, v_).transpose(1, 2).reshape(C, Tq, H * dh)


class TemporalLayer(nn.Module):
    """(B30) One pre-LN layer of the temporal transformer (attention + feed-forward)."""

    def __init__(self, width, heads, ffn):
        super().__init__()
        self.heads = heads
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.ff = mlp(width, ffn, width)

    def qkv_of(self, x):
        """q, k, v [..., heads, dh] of tokens x [..., D]."""
        D = x.shape[-1]
        return self.qkv(self.n1(x)).view(*x.shape[:-1], 3, self.heads, D // self.heads).unbind(-3)

    def finish(self, x, att):
        """The residual after the attention output att [..., D], then the feed-forward."""
        x = x + self.proj(att)
        return x + self.ff(self.n2(x))


class TemporalAttention(nn.Module):
    """B30 (2026-10-14; TokPolicy(temporal=True)): the temporal transformer (module docstring, rl/docs/
    TEMPORAL_ATTENTION.md). width: the spatial width d; glob: the policy has a global token (s_t reads it, value_global's
    readout P_g exists); D, layers, heads, window W, rooms R."""

    def __init__(self, width, glob, D=256, layers=4, heads=4, window=256, rooms=128):
        super().__init__()
        if D % heads or window < 2 or rooms < 1:
            raise ValueError(f'TemporalAttention: width {D} / heads {heads} / window {window} / rooms {rooms}')
        self.D, self.L, self.H, self.W, self.R = int(D), int(layers), int(heads), int(window), int(rooms)
        self.inp = nn.Linear(2 * width if glob else width, D)   # W_s
        self.layers = nn.ModuleList(TemporalLayer(D, heads, 4 * D) for _ in range(layers))
        self.norm = nn.LayerNorm(D)
        self.rel = nn.Parameter(torch.zeros(heads, window + 1))   # [h, distance 0 .. W-1], [h, W]: a room key
        self.room_in = nn.Linear(D, D)                            # W_r
        self.room_emb = nn.Embedding(rooms, D)
        self.stage_emb = nn.Embedding(N_STAGE_EMB, D)
        self.read = nn.Linear(D, width)                           # P
        self.read_g = nn.Linear(D, width) if glob else None       # P_g
        self.feed = nn.Linear(D, width)                           # Q
        self.hist_kind = nn.Parameter(torch.zeros(width))         # the history token's kind embedding
        nn.init.normal_(self.room_emb.weight, std=0.02)
        nn.init.normal_(self.stage_emb.weight, std=0.02)
        nn.init.normal_(self.hist_kind, std=0.02)

    def embed(self, z, gn):
        """s_t = W_s [z ; norm_g(yg)] (float32)."""
        return self.inp(torch.cat([z, gn], -1) if gn is not None else z)

    def room_token(self, o_src, k, stage):
        """r_k = W_r o_exit + E_room[k] + E_stage[stage]."""
        return self.room_in(o_src.float()) + self.room_emb(k.long().clamp(0, self.R - 1)) + \
            self.stage_emb(stage.long().clamp(0, N_STAGE_EMB - 1))

    @_no_compile
    def step(self, st, s, slot, rec, room_source=None):
        """The actor's / evaluator's step for n records, in place on TemporalState st (graph-safe: static shapes, no
        host synchronisation). s [n, D] (float32), slot [n] long (distinct, except the scratch slot), rec: dict(first
        [n], map [n, 8, 13, 13] uint8, events [n], stage [n]). Returns dict(o [n, D], s, kv [n, L, 2, D] (the step
        token's K / V per layer, st.dtype), create [n] bool (a room token made), nroom [n] (rooms after it), steps [n]
        (the step's index in its episode), stage_prev [n] (the made room's stage)."""
        dt = st.kv.dtype
        n, D, H, W, R = s.shape[0], self.D, self.H, self.W, self.R
        dh = D // H
        dev = s.device
        first = rec['first'] != 0
        zero = torch.zeros_like(slot)
        steps0 = torch.where(first, zero, st.steps.index_select(0, slot))
        nroom0 = torch.where(first, zero, st.nroom.index_select(0, slot))
        map3 = rec['map'][:, 3].reshape(n, -1)
        pmap = st.prev_map.index_select(0, slot)
        shown = map3.any(1).bool()   # (a record without a current room on its map neither moves nor replaces the last one)
        moved = (shown & (map3 != pmap).any(1)) | ((rec['events'].long() & (EV_EXIT | EV_NEW_ROOM)) != 0)
        create = (~first) & (steps0 > 0) & moved & (nroom0 < R)
        kroom = nroom0.clamp(max=R - 1)
        stage_prev = st.prev_stage.index_select(0, slot)
        src = st.prev_o.index_select(0, slot) if room_source is None else room_source
        r = self.room_token(src, kroom, stage_prev)
        x = torch.stack([r, s.float()], 1)   # [n, 2 (room query, step query), D]
        pos = R + torch.remainder(steps0, W)
        nroom1 = nroom0 + create.long()
        # masks of the call's rows [n, H, 2, R + W]: rooms first, then the step ring (distance d of ring entry j)
        jr = torch.arange(R, device=dev)
        d = torch.remainder(steps0[:, None] - torch.arange(W, device=dev)[None], W)
        rel = self.rel.float()
        neg = torch.full((), T_NEG, device=dev)
        m_room_q = torch.cat([torch.where(jr[None] <= kroom[:, None], 0.0, neg)[:, None].expand(n, H, R),
                              neg.expand(n, H, W)], -1)
        m_step_q = torch.cat([torch.where((jr[None] < nroom1[:, None])[:, None], rel[:, W][None, :, None], neg),
                              torch.where((d <= steps0[:, None])[:, None], rel[:, d].transpose(0, 1), neg)], -1)
        S1 = st.kv.shape[2]
        base = torch.full((S1, H, 2, R + W), T_NEG, device=dev)
        base[..., 0] = 0.0   # (rows of slots outside the call keep one finite key: no NaN anywhere)
        mask = base.index_copy(0, slot, torch.stack([m_room_q, m_step_q], 2)).to(dt)
        kv_out = []
        for l_, lay in enumerate(self.layers):
            q, k, v = lay.qkv_of(x)   # [n, 2, H, dh]
            kc, vc = st.kv[l_, 0], st.kv[l_, 1]   # [S1, H, R + W, dh] views of the cache
            kc[slot, :, pos] = k[:, 1].to(dt)
            vc[slot, :, pos] = v[:, 1].to(dt)
            kc[slot, :, kroom] = torch.where(create[:, None, None], k[:, 0].to(dt), kc[slot, :, kroom])
            vc[slot, :, kroom] = torch.where(create[:, None, None], v[:, 0].to(dt), vc[slot, :, kroom])
            kv_out.append(torch.stack([k[:, 1], v[:, 1]], 1).reshape(n, 2, D))
            q_all = torch.zeros((S1, H, 2, dh), device=dev, dtype=dt).index_copy(0, slot, q.transpose(1, 2).to(dt))
            with _sdpa_cache_backends():
                att = F.scaled_dot_product_attention(q_all, kc, vc, attn_mask=mask)
            att = att.index_select(0, slot).float().transpose(1, 2).reshape(n, 2, D)
            x = lay.finish(x, att)
        o = self.norm(x[:, 1])
        st.steps.index_copy_(0, slot, steps0 + 1)
        st.nroom.index_copy_(0, slot, nroom1)
        st.prev_o.index_copy_(0, slot, o)
        st.prev_stage.index_copy_(0, slot, rec['stage'].long())
        st.prev_map.index_copy_(0, slot, torch.where((shown | first)[:, None], map3, pmap))
        return dict(o=o, s=s, kv=torch.stack(kv_out, 1).to(dt), create=create, nroom=nroom1, steps=steps0,
                    stage_prev=stage_prev)

    def room_table(self, o_store, td, track_grad=False):
        """(B31, the learner; once per PPO epoch, at its start) every room token of the rollout through the layers
        without gradient, with the current parameters: per (worker, episode) group of temporal_inputs' plan td the room
        tokens r = W_r o_exit + E_room[k] + E_stage[stage] (o_exit: rows r_src of o_store, the rollout's o) in their
        order, each attending to the earlier ones of its group and itself (the actor's room query). Returns
        [(K, V) per layer]: [rooms + 1, heads, dh] each (the last row: the plan's dummy room, the padding target of
        temporal_part's gathers). Run it under the learner's autocast: the values are those the per-chunk recomputation
        of the rooms made before a chunk gave (B30), up to summation order.
        track_grad=True is the raw-sequence writer path: call anew for each learner
        forward, including pre-chunk rooms. It does not change the legacy default."""
        D, H = self.D, self.H
        dh = D // H
        grp = td['r_grp']
        G, Rg = grp.shape
        dev = grp.device
        with contextlib.nullcontext() if track_grad else torch.no_grad():
            xr = self.room_token(o_store.index_select(0, td['r_src']), td['r_k'], td['r_stage'])
            valid = grp >= 0
            x = xr.index_select(0, grp.clamp(min=0).reshape(-1)).view(G, Rg, D)
            ar = torch.arange(Rg, device=dev)
            ok = ((ar[:, None] >= ar[None]) & valid[:, None, :]) | torch.eye(Rg, dtype=torch.bool, device=dev)
            bias = torch.where(ok, 0.0, torch.full((), T_NEG, device=dev))[:, None]
            pos = td['r_gpos']
            out = []
            for lay in self.layers:
                q, k, v = lay.qkv_of(x)   # [G, Rg, H, dh]
                out.append((k.reshape(G * Rg, H, dh).index_select(0, pos), v.reshape(G * Rg, H, dh).index_select(0, pos)))
                x = lay.finish(x, _attend(q, k, v, bias))
        return out

    def sequence(self, s_rows, tp):
        """The learner's chunks (temporal_part's plan tp): s_rows [rows, D] (with gradient) -> o [rows, D]. Per chunk:
        the room tokens made before it (stop-grad: their per-layer K / V from the epoch's room_table, B31), those made
        inside it (with gradient), the stored K / V of the window - 1 decisions before it (stop-grad) and its own
        decisions, with the actor's masks and relative bias (block-causal)."""
        C, Tc = tp['qep'].shape
        D, H, W, L = self.D, self.H, self.W, self.L
        dev = s_rows.device
        x = s_rows.float().index_select(0, tp['imap']).view(C, Tc, D)
        qep, pre_ep = tp['qep'], tp['qep'][:, 0]
        pre_ok, in_ok, in_ep, in_at = tp['pre_valid'], tp['in_valid'], tp['in_ep'], tp['in_at']
        Rp, Ri = pre_ok.shape[1], in_ok.shape[1]
        rel = self.rel
        neg = torch.full((), T_NEG, device=dev)
        kv_pre = tp['pre_kv']   # rooms made before the chunk: [(K, V) [C, Rp, H, dh] per layer] (B31: the table's)
        # rooms made inside the chunk (gradient): the earlier rooms of their episode and themselves
        xr = self.room_token(tp['in_o'], tp['in_k'], tp['in_stage'])
        ai = torch.arange(Ri, device=dev)
        ok = torch.cat([pre_ok[:, None, :] & (in_ep == pre_ep[:, None])[:, :, None],
                        (in_ok[:, None, :] & (ai[None, :, None] >= ai[None, None, :]) &
                         (in_ep[:, :, None] == in_ep[:, None, :])) | torch.eye(Ri, dtype=torch.bool, device=dev)], -1)
        bias = torch.where(ok, 0.0, neg)[:, None]
        kv_in = []
        for l_, lay in enumerate(self.layers):
            q, k, v = lay.qkv_of(xr)
            kv_in.append((k, v))
            xr = lay.finish(xr, _attend(q, torch.cat([kv_pre[l_][0], k], 1),
                                        torch.cat([kv_pre[l_][1], v], 1), bias))
        # the chunk's decisions: rooms (pre, in), the context (P decisions before the chunk), the chunk itself
        ctx, ctx_ep = tp['ctx_kv'], tp['ctx_ep']   # [C, P, L, 2, D] (as stored), [C, P]
        P = ctx.shape[1]
        i = torch.arange(Tc, device=dev)
        d_ctx = i[:, None] + P - torch.arange(P, device=dev)[None]
        d_ch = i[:, None] - i[None]
        valid = torch.cat([
            pre_ok[:, None, :] & (qep == pre_ep[:, None])[:, :, None],
            in_ok[:, None, :] & (in_at[:, None, :] <= i[None, :, None]) & (in_ep[:, None, :] == qep[:, :, None]),
            (ctx_ep[:, None, :] == qep[:, :, None]) & (d_ctx <= W - 1)[None],
            (qep[:, None, :] == qep[:, :, None]) & ((d_ch >= 0) & (d_ch <= W - 1))[None]], -1)
        b_steps = rel[:, torch.cat([d_ctx, d_ch], 1).clamp(0, W)]   # [H, Tc, P + Tc]
        b_full = torch.cat([rel[:, W][:, None, None].expand(H, Tc, Rp + Ri), b_steps], -1)
        bias = torch.where(valid[:, None], b_full[None], neg)   # [C, H, Tc, K]
        dh = D // H
        for l_, lay in enumerate(self.layers):
            q, k, v = lay.qkv_of(x)
            ck = ctx[:, :, l_, 0].float().view(C, P, H, dh)
            cv = ctx[:, :, l_, 1].float().view(C, P, H, dh)
            K = torch.cat([kv_pre[l_][0], kv_in[l_][0], ck, k], 1)
            V = torch.cat([kv_pre[l_][1], kv_in[l_][1], cv, v], 1)
            x = lay.finish(x, _attend(q, K, V, bias))
        return self.norm(x).reshape(C * Tc, D).index_select(0, tp['omap'])


class TemporalState:
    """B30 (2026-10-14): the actor's / evaluator's per-worker temporal state (module docstring) for a temporal
    TokPolicy, on its device: slots worker slots plus one scratch slot (index `scratch`: GraphActor's padded rows write
    there). kv [layers, 2, slots + 1, heads, rooms + window, dh] in kv_dtype (default bfloat16 on CUDA, float32
    elsewhere); steps / nroom [slots + 1] long; prev_o [slots + 1, D]; prev_stage; prev_map [slots + 1, 169] uint8."""

    def __init__(self, model, slots, device=None, kv_dtype=None):
        ta = model.tatt if isinstance(model, nn.Module) and hasattr(model, 'tatt') else model
        if ta is None:
            raise ValueError('TemporalState: the model has no temporal transformer')
        dev = torch.device(device) if device is not None else next(ta.parameters()).device
        if kv_dtype is None:
            kv_dtype = torch.bfloat16 if dev.type == 'cuda' else torch.float32
        self.slots, self.scratch = int(slots), int(slots)
        S1 = self.slots + 1
        self.L, self.D, self.H, self.W, self.R = ta.L, ta.D, ta.H, ta.W, ta.R
        # RebuildGraphActor owns its source/Reader rings. Keep only the legacy
        # tensor's device/dtype/slot axes, not its unused R+W history storage.
        span = 0 if getattr(model, 'rebuild_memory', False) else ta.R + ta.W
        self.kv = torch.zeros((ta.L, 2, S1, ta.H, span, ta.D // ta.H), dtype=kv_dtype, device=dev)
        self.steps = torch.zeros(S1, dtype=torch.int64, device=dev)
        self.nroom = torch.zeros(S1, dtype=torch.int64, device=dev)
        self.prev_o = torch.zeros((S1, ta.D), device=dev)
        self.prev_stage = torch.zeros(S1, dtype=torch.int64, device=dev)
        self.prev_map = torch.zeros((S1, MAP[1] * MAP[2]), dtype=torch.uint8, device=dev)
        from .room_memory import RawRoomState
        self.raw_rooms = RawRoomState(S1, ta.R) if getattr(model, 'room_writer', None) is not None else None
        from .rebuild_memory import ActorMemory
        self.rebuild = ActorMemory(S1) if getattr(model, 'rebuild_memory', False) else None

    def step_inputs(self, model, rows, pending, slots):
        kw = dict(mode='step', state=self, slot=torch.as_tensor(slots, dtype=torch.long, device=self.kv.device))
        if self.rebuild is not None:
            kw.update(raw_rows=rows, raw_pending=pending, host_slots=np.asarray(slots, np.int64))
        if self.raw_rooms is not None:
            from .room_memory import prepare_room_step
            kw['room_source'], kw['room_visits'] = prepare_room_step(model, self, rows, pending, slots)
        return kw

    @property
    def dtype(self):
        return self.kv.dtype

    def reset_(self):
        """Every slot back to the start (in place: captured graphs keep their pointers)."""
        for t_ in (self.kv, self.steps, self.nroom, self.prev_o, self.prev_stage, self.prev_map):
            t_.zero_()
        if self.raw_rooms is not None:
            self.raw_rooms.reset(list(range(self.slots + 1)))
        if self.rebuild is not None:
            for s in range(self.slots + 1): self.rebuild.reset(s)

    def _tensors(self):
        """(tensor, slot axis) of every per-slot part of the state."""
        return ((self.kv, 2), (self.steps, 0), (self.nroom, 0), (self.prev_o, 0), (self.prev_stage, 0),
                (self.prev_map, 0))

    def copy_slot_(self, src, dst, other=None):
        """B34 (2026-10-16, the counterfactual teacher's lanes): slot(s) src of `other` (a TemporalState of the same
        model and dtype; None: this one) copied into slot(s) dst of this state, on the device and in place (captured
        graphs keep their pointers): every layer's K / V ring (rooms and steps), the step and room counts, prev_o,
        prev_stage, prev_map. src / dst: ints or int64 sequences / tensors of equal length (dst distinct). A step from
        the copy is the step the source slot would have taken (same values, bit for bit)."""
        o = self if other is None else other
        dev = self.kv.device
        s = torch.as_tensor(src, dtype=torch.int64, device=dev).reshape(-1)
        d = torch.as_tensor(dst, dtype=torch.int64, device=dev).reshape(-1)
        if s.shape != d.shape:
            raise ValueError('TemporalState.copy_slot_: src and dst differ in length')
        if o.kv.shape[:2] != self.kv.shape[:2] or o.kv.shape[3:] != self.kv.shape[3:] or o.kv.dtype != self.kv.dtype:
            raise ValueError('TemporalState.copy_slot_: states of other shapes / dtypes')
        if self.raw_rooms is not None:
            self.raw_rooms.copy(s.cpu().tolist(), d.cpu().tolist(), o.raw_rooms)
        if self.rebuild is not None:
            for a, b in zip(s.cpu().tolist(), d.cpu().tolist()): self.rebuild.copy(a, b, o.rebuild)
        for (t_, ax), (u_, _) in zip(self._tensors(), o._tensors()):
            t_.index_copy_(ax, d, u_.index_select(ax, s))

    def reset_slot_(self, idx):
        """B34: slot(s) idx back to the start (zeros: the next record steps as an episode's first step would, without
        any history)."""
        d = torch.as_tensor(idx, dtype=torch.int64, device=self.kv.device).reshape(-1)
        for t_, ax in self._tensors():
            t_.index_fill_(ax, d, 0)
        if self.raw_rooms is not None:
            self.raw_rooms.reset(d.cpu().tolist())
        if self.rebuild is not None:
            for s in d.cpu().tolist(): self.rebuild.reset(s)


def temporal_out(info):
    """(B31) a step's outputs (TemporalAttention.step) for the rollout (train_tok.store): the counters as numpy
    (create bool, nroom, steps, stage_prev int64: one device -> host copy of 4 integers per record) and the step
    token's K / V, s and o as the tensors they are, on their device (kv [n, L, 2, D] and s [n, D] in the state's
    dtype, o [n, D] float32): store writes them into the rollout's device tensors (no copy to the host)."""
    c = torch.stack([info['create'].long(), info['nroom'].long(), info['steps'].long(), info['stage_prev'].long()],
                    1).cpu().numpy()
    return dict(create=c[:, 0] != 0, nroom=c[:, 1], steps=c[:, 2], stage_prev=c[:, 3], kv=info['kv'].detach(),
                s=info['s'].detach().to(info['kv'].dtype), o=info['o'].detach().float(),
                **(dict(room_visits=info['room_visits']) if 'room_visits' in info else {}),
                **(dict(raw_contexts=info['raw_contexts']) if 'raw_contexts' in info else {}))


def temporal_chunks(length, mask, T, T_mem):
    """(B30, the learner) memory_chunks at two levels: each worker's decisions cut at 0, T, 2T, ... into temporal
    chunks, each padded to a multiple of T_mem by repeating its last decision (weight 0; the GRU's chunks of T_mem nest
    inside); chunks without a trained decision are dropped. Returns memory_chunks' dict (pos, weight, start per GRU
    chunk, chunks) plus 'tch': int64 [C, 5] per temporal chunk (row offset, rows, worker, first decision, real
    decisions); row offsets and counts are multiples of T_mem.
    T_mem None (B32, 2026-10-15: a temporal model without the GRU): no GRU chunks nested, no padding: a chunk's rows
    are its decisions (weight 1 where trained, 0 where the temporal transformer only steps through them, e.g. a
    worker's open last decision); 'start' holds each temporal chunk's first decision, 'chunks' their number."""
    W, cap = mask.shape
    pos, weight, start, tch = [], [], [], []
    off = 0
    for w in range(W):
        n = int(length[w])
        for s in range(0, n, T):
            e = min(s + T, n)
            if not mask[w, s:e].any():
                continue
            rows = e - s if T_mem is None else -(-(e - s) // T_mem) * T_mem
            m = np.zeros(rows, bool)
            m[:e - s] = mask[w, s:e]
            pos.append(w * cap + np.minimum(s + np.arange(rows), e - 1))
            weight.append(m)
            start.extend([w * cap + s] if T_mem is None else w * cap + s + np.arange(0, rows, T_mem))
            tch.append((off, rows, w, s, e - s))
            off += rows
    if not pos:
        return dict(pos=np.zeros(0, np.int64), weight=np.zeros(0, np.float32), start=np.zeros(0, np.int64), chunks=0,
                    tch=np.zeros((0, 5), np.int64))
    return dict(pos=np.concatenate(pos).astype(np.int64), weight=np.concatenate(weight).astype(np.float32),
                start=np.asarray(start, np.int64), chunks=len(start), tch=np.asarray(tch, np.int64))


def temporal_inputs(roll, tch):
    """(B30 / B31, the learner, once per update) the device plan of a rollout's temporal chunks (tch rows: offset,
    rows, worker, first decision s0, real decisions) for TemporalAttention.room_table and (through temporal_part)
    sequence. The rollout's K / V and o stay where the actor wrote them (train_tok.Rollout: device tensors t_kv
    [rows, L, 2, D], t_o [rows, D]; host maps t_row: decision -> row, rooms_*: the room tokens' sources); only integer
    index arrays are built here and sent to the device once. Returns dict:
      kv, o: the rollout's tensors; ctx_rows [C, P] (rows of kv: the P decisions before each chunk), ctx_ep [C, P];
      the room tokens (every one in the rollout's room lists, per worker in order, then one dummy): r_src [N + 1] (rows
        of o: o_exit), r_k, r_stage, r_grp [G, Rg] (the rooms of each (worker, episode) in order, -1 padding: the
        causal groups of room_table), r_gpos [N + 1] (each room's position in r_grp's flat layout);
      pre_id / pre_valid [C, Rp] (the chunk's episode's rooms made before it; ids into the rooms, padding: the
        dummy), in_id / in_at / in_ep / in_valid [C, Ri] (rooms made at its real decisions, in_at the chunk position
        that first sees it); pre_n, in_n (numpy: the chunks' room counts); Rp, Ri >= 1."""
    dev = roll.t_o.device
    P = roll.tP
    tch = np.asarray(tch, np.int64).reshape(-1, 5)
    C = len(tch)
    ws, s0s = tch[:, 2], tch[:, 3]
    ext = s0s[:, None] + np.arange(P)[None]
    ctx_rows = roll.t_row[ws[:, None], ext]
    ctx_rows = np.where(ctx_rows >= 0, ctx_rows, roll.t_kv.shape[0] - 1)   # (never -1: a guard, the scratch row)
    ctx_ep = roll.t_ep[ws[:, None], ext]
    nr = np.asarray(roll.rooms_n, np.int64)
    off = np.cumsum(nr) - nr
    N = int(nr.sum())
    wid = np.repeat(np.arange(len(nr)), nr)
    j = np.arange(N) - off[wid]
    r_src = np.append(roll.rooms_src[wid, j], 0)
    r_k = np.append(roll.rooms_k[wid, j], 0)
    r_stage = np.append(roll.rooms_stage[wid, j], 0)
    r_ep, r_t = roll.rooms_ep[wid, j], roll.rooms_t[wid, j]
    newg = np.ones(N, bool)
    newg[1:] = (wid[1:] != wid[:-1]) | (r_ep[1:] != r_ep[:-1])
    gid = np.cumsum(newg) - 1
    gstart = np.flatnonzero(newg)
    within = np.arange(N) - gstart[gid] if N else np.zeros(0, np.int64)
    G = len(gstart)
    Rg = max(int(within.max()) + 1 if N else 1, 1)
    grp = np.full((G + 1, Rg), -1, np.int64)
    grp[gid, within] = np.arange(N)
    grp[G, 0] = N   # (the dummy room: a group of its own)
    r_gpos = np.append(gid * Rg + within, G * Rg)
    pre, inn = [], []
    for _, _, w, s0, nr_ in tch:
        lo, m = off[w], nr[w]
        rt_w, ep_w = r_t[lo:lo + m], r_ep[lo:lo + m]
        ep0 = roll.t_ep[w, P + s0]
        pre.append(lo + np.flatnonzero((ep_w == ep0) & (rt_w < s0)))
        inn.append(lo + np.flatnonzero((rt_w >= s0) & (rt_w < s0 + nr_)))
    Rp = max([len(x) for x in pre] + [1])
    Ri = max([len(x) for x in inn] + [1])
    pre_id, in_id = np.full((C, Rp), N, np.int64), np.full((C, Ri), N, np.int64)
    pre_valid, in_valid = np.zeros((C, Rp), bool), np.zeros((C, Ri), bool)
    in_at, in_ep = np.zeros((C, Ri), np.int64), np.full((C, Ri), -3, np.int64)
    for c in range(C):
        pre_id[c, :len(pre[c])], pre_valid[c, :len(pre[c])] = pre[c], True
        m = len(inn[c])
        in_id[c, :m], in_valid[c, :m] = inn[c], True
        in_at[c, :m] = r_t[inn[c]] - s0s[c]
        in_ep[c, :m] = r_ep[inn[c]]

    def t(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)
    raw = {}
    if getattr(roll, 'raw_rooms', None) is not None:
        visits = [roll.raw_rooms[w][j_] for w, j_ in zip(wid, j)]
        needed = []
        for a, b in zip(pre, inn):
            ids = set()
            for i in np.concatenate((a, b)):
                ids.update(range(int(gstart[gid[i]]), int(i) + 1))
            needed.append(sorted(ids))
        raw = dict(raw_visits=visits, raw_needed=needed)
    return dict(**raw, kv=roll.t_kv, o=roll.t_o, ctx_rows=t(ctx_rows), ctx_ep=t(ctx_ep), r_src=t(r_src), r_k=t(r_k),
                r_stage=t(r_stage), r_grp=t(grp), r_gpos=t(r_gpos), pre_id=t(pre_id), pre_valid=t(pre_valid),
                in_id=t(in_id), in_at=t(in_at), in_ep=t(in_ep), in_valid=t(in_valid),
                pre_n=np.array([len(x) for x in pre], np.int64), in_n=np.array([len(x) for x in inn], np.int64))


def temporal_part(td, cids, counts, qep_rows, table, to_dev=None):
    """(B30 / B31, the learner) TemporalAttention.sequence's plan for a part made of whole temporal chunks: td
    (temporal_inputs), cids (numpy) the chunks in the part's order, counts (numpy) their rows (the part's rows are the
    chunks' rows one after another), qep_rows [rows] (tensor) the episode serial of each row, table: the epoch's
    room_table. Everything is gathered on the device (the context's stored K / V, the rooms' K / V, the in-chunk
    rooms' sources); rooms trimmed to the part's largest counts. to_dev: int64 numpy -> device tensor (train_tok:
    pinned, non-blocking; default .to)."""
    dev = qep_rows.device
    if to_dev is None:
        def to_dev(a):
            return torch.from_numpy(np.ascontiguousarray(a, np.int64)).to(dev)
    C = len(cids)
    Tc = int(counts.max())
    offs = np.cumsum(counts) - counts
    pos = np.arange(Tc)[None]
    real = pos < counts[:, None]
    imap = np.where(real, offs[:, None] + pos, offs[:, None])
    omap = np.concatenate([c * Tc + np.arange(n) for c, n in enumerate(counts)])
    ci = to_dev(cids)
    Rp = max(int(td['pre_n'][cids].max()), 1)
    Ri = max(int(td['in_n'][cids].max()), 1)
    imap_t = to_dev(imap.reshape(-1))
    kv = td['kv']
    P = td['ctx_rows'].shape[1]
    ctx_kv = kv.index_select(0, td['ctx_rows'].index_select(0, ci).reshape(-1)).view(C, P, *kv.shape[1:])
    tp = dict(raw_room_memory=td.get('raw_room_memory', False), imap=imap_t, omap=to_dev(omap), qep=qep_rows.index_select(0, imap_t).view(C, Tc), ctx_kv=ctx_kv,
              ctx_ep=td['ctx_ep'].index_select(0, ci))
    pre_id = td['pre_id'].index_select(0, ci)[:, :Rp].reshape(-1)
    tp['pre_valid'] = td['pre_valid'].index_select(0, ci)[:, :Rp]
    tp['pre_kv'] = [(K.index_select(0, pre_id).view(C, Rp, *K.shape[1:]), V.index_select(0, pre_id).view(
        C, Rp, *V.shape[1:])) for K, V in table]
    in_id = td['in_id'].index_select(0, ci)[:, :Ri].reshape(-1)
    tp['in_o'] = td['o'].index_select(0, td['r_src'].index_select(0, in_id)).view(C, Ri, -1)
    tp['in_k'] = td['r_k'].index_select(0, in_id).view(C, Ri)
    tp['in_stage'] = td['r_stage'].index_select(0, in_id).view(C, Ri)
    for k in ('in_at', 'in_ep', 'in_valid'):
        tp[k] = td[k].index_select(0, ci)[:, :Ri]
    return tp


class TokPolicy(nn.Module):
    def __init__(self, width=192, layers=4, heads=4, items=False, charge=False, choice=0, ent_ext=False, pchar=False,
                 memory=0, item_tokens=False, global_token=False, item_shared=False, item_slots=INV_CAP, temporal=False,
                 temporal_width=256, temporal_layers=4, temporal_heads=4, temporal_window=256, temporal_rooms=128,
                 obs_prices=False, obs_hearts=False, obs_room_type=False, room_memory=False, obs_inventory_all=False,
                 rebuild_memory=False, memory_block=64, memory_tokens=4, source_cache=False,
                 memory_refresh_updates=1):
        super().__init__()
        if int(item_slots) != INV_CAP and not item_tokens:
            raise ValueError('TokPolicy: item_slots needs item_tokens=True (B24)')
        if item_tokens and not items:
            raise ValueError('TokPolicy: item_tokens needs items=True (ROW inv / pitem / pinv)')
        if item_shared and not item_tokens:
            raise ValueError('TokPolicy: item_shared needs item_tokens=True (B23: the item tokens read item_emb)')
        self.config = dict(width=width, layers=layers, heads=heads)
        if items:   # (the key only when set: items-free checkpoints keep their config)
            self.config['items'] = True
        if charge:   # 2026-10-07: ROW 'pcharge' as the player token's last inputs (the key only when set, as items)
            self.config['charge'] = True
        if choice:   # 2026-10-07 (Phase B2): K choice-value heads (ChoiceHeads; the key only when set)
            self.config['choice'] = int(choice)
        if ent_ext:   # 2026-10-08: entity columns ENT_F0 .. ENT_F as the entity token's last inputs (the key only when set)
            self.config['ent_ext'] = True
        if pchar:   # 2026-10-08: ROW 'pchar' one-hot as the player token's last inputs (the key only when set)
            self.config['pchar'] = True
        # 2026-10-16 (B35, observation v2; the module docstring): the keys only when set
        self.obs_prices, self.obs_hearts, self.obs_room_type = bool(obs_prices), bool(obs_hearts), bool(obs_room_type)
        for key_, on_ in (('obs_prices', obs_prices), ('obs_hearts', obs_hearts), ('obs_room_type', obs_room_type)):
            if on_:
                self.config[key_] = True
        self.obs_inventory_all = bool(obs_inventory_all)
        if obs_inventory_all:
            if not items:
                raise ValueError('obs_inventory_all requires items=True')
            self.config['obs_inventory_all'] = True
        self.items = bool(items)
        self.charge = bool(charge)
        self.ent_ext = bool(ent_ext)
        self.pchar = bool(pchar)
        self.heads = ITEM_HEADS if items else HEADS
        self.type_emb, self.variant_emb, self.subtype_emb = nn.Embedding(1024, 24), nn.Embedding(1024, 12), \
            nn.Embedding(256, 4)
        if items:   # id 0 = none: a zero vector
            self.item_emb = nn.Embedding(N_COLLECTIBLE, ITEM_EMB, padding_idx=0)
            self.trinket_emb = nn.Embedding(N_TRINKET, TRINKET_EMB, padding_idx=0)
            self.pill_emb = nn.Embedding(N_PILL_COLOR, PILL_EMB, padding_idx=0)
            self.effect_emb = nn.Embedding(N_PILL_EFFECT, EFFECT_EMB, padding_idx=0)
            self.card_emb = nn.Embedding(N_CARD, CARD_EMB, padding_idx=0)
        self.entity = mlp(ENT_F0 + 40 + ((ITEM_EMB + TRINKET_EMB) if items else 0) + ((ENT_F - ENT_F0) if ent_ext else 0)
                          + (PRICE_F if obs_prices else 0), width, width)
        self.patch = nn.Sequential(nn.Conv2d(GRID[0], 32, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.GELU(), nn.Flatten(),
                                   nn.Linear(32 * PATCH * PATCH, 128), nn.GELU())
        self.player = mlp(PLAYER_F + sum(HEADS) + 128 + (PLAYER_ITEMS_F if items else 0) + (PCHARGE_F if charge else 0)
                          + (PHEART_F if obs_hearts else 0), width, width)
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
        # 2026-10-08 (pchar): registered after everything else, zero at the start (the module docstring)
        self.char_emb = nn.Embedding(N_CHAR, width) if pchar else None
        if pchar:
            nn.init.zeros_(self.char_emb.weight)
        # 2026-10-11 (memory): registered after everything else; mem_out zero at the start (the module docstring)
        self.memory = int(memory or 0)
        if self.memory:   # (the key only when set: memory-free checkpoints keep their config)
            self.config['memory'] = self.memory
            self.mem_gru = nn.GRUCell(width, self.memory)
            self.mem_out = nn.Linear(self.memory, width)
            nn.init.zeros_(self.mem_out.weight)
            nn.init.zeros_(self.mem_out.bias)
        # 2026-10-12 (B21 / B22): registered after everything else (memory included; the module docstring); the
        # config keys only when set
        self.item_tokens, self.global_token = bool(item_tokens), bool(global_token)
        if self.item_tokens:
            self.config['item_tokens'] = True
        if self.global_token:
            self.config['global_token'] = True
        # 2026-10-13 (B23): the item tokens' collectible table is item_emb itself (the key only when set)
        self.item_shared = bool(item_shared)
        if self.item_shared:
            self.config['item_shared'] = True
        # 2026-10-14 (B24): the number of held-collectible token slots (the key only when set; the same parameters)
        if self.item_tokens and int(item_slots) != INV_CAP:
            self.config['item_slots'] = int(item_slots)
        self.itok = ItemTokens(width, coll=self.item_emb if self.item_shared else None,
                               slots=int(item_slots)) if self.item_tokens else None
        self.gtok = GlobalToken(width, self.memory) if self.global_token else None
        # 2026-10-14 (B30): the temporal transformer, registered after everything else (module docstring); the config
        # keys only when on
        self.tatt = None
        if temporal:
            self.config.update(temporal=True, temporal_width=int(temporal_width), temporal_layers=int(temporal_layers),
                               temporal_heads=int(temporal_heads), temporal_window=int(temporal_window),
                               temporal_rooms=int(temporal_rooms))
            self.tatt = TemporalAttention(width, glob=self.global_token, D=int(temporal_width),
                                          layers=int(temporal_layers), heads=int(temporal_heads),
                                          window=int(temporal_window), rooms=int(temporal_rooms))
        # 2026-10-16 (B35): the room-type table, registered after everything else, zero at the start (the module
        # docstring)
        self.room_type_emb = None
        if self.obs_room_type:
            self.room_type_emb = nn.Embedding(N_ROOM_TYPE, width)
            nn.init.zeros_(self.room_type_emb.weight)

        # Raw sequence writer is opt-in; legacy parameter order/config stay unchanged.
        self.room_writer = None
        if room_memory:
            if self.tatt is None:
                raise ValueError('room_memory requires temporal=True')
            from .room_memory import RoomSequenceWriter
            self.room_writer = RoomSequenceWriter(self.tatt.D, self.tatt.H)
            self.config['room_memory'] = True

        self.rebuild_memory = bool(rebuild_memory)
        self.source_cache = bool(source_cache)
        self.memory_refresh_updates = memory_refresh_updates
        self.cache_generation = 0
        if self.rebuild_memory:
            if not temporal or not global_token or memory or room_memory:
                raise ValueError('rebuild_memory requires temporal/global_token, no GRU or legacy room writer')
            from .rebuild_memory import MemoryReader
            self.reader = MemoryReader(width, temporal_width, temporal_layers, temporal_heads,
                                       temporal_window, memory_block, memory_tokens)
            self.config.update(rebuild_memory=True, memory_block=memory_block,
                               memory_tokens=memory_tokens, source_cache=source_cache,
                               memory_refresh_updates=memory_refresh_updates)

    def encode_rebuild(self, b):
        y, pad = self._forward_fused(b, None, bool(b.get('_source_packed', False)), False, None, False,
                                     dict(mode='rebuild_source'), None)
        if b.get('_source_compact', False):
            z = self.norm(y.index_select(0, b['starts']).float())
            g = self.gtok.norm(y.index_select(0, b['glast']).float())
        else:
            z, g = self.norm(y[:, 0].float()), self.gtok.norm(y[:, -1].float())
        u = self.reader.inp(torch.cat([z, g], -1))
        return y, pad, u

    def invalidate_source_cache(self):
        self.cache_generation += 1

    @property
    def source_generation(self):
        return self.cache_generation // self.memory_refresh_updates

    def load_state_dict(self, *args, **kwargs):
        result = super().load_state_dict(*args, **kwargs)
        self.invalidate_source_cache()
        return result

    def encode_room_sources(self, b):
        """Current-parameter spatial encoding for raw room observations, without H or a temporal readout.

        This is a separate writer input path; the policy's existing H feedback is unchanged.
        """
        return self._forward_fused(b, None, False, False, None, False, dict(mode='room_source'), None)

    def write_room_sequences(self, b, lengths):
        """One D-wide source per complete causal visit; no cached o_exit and no detach."""
        if self.room_writer is None:
            raise ValueError('write_room_sequences requires room_memory=True')
        return self.room_writer(self.encode_room_sources(b), lengths)

    def main_parameters(self):
        """The parameters PPO's optimiser owns: all but the choice heads' (in the order of parameters())."""
        return [p for n, p in self.named_parameters() if not n.startswith('choice.')]

    def set_math_attention(self, on, nhwc=None, ln=None):
        """2026-10-10 (B18): the blocks' attention as _MathAttention (train_tok --learner-fast; the same function);
        nhwc: the convolutions' inputs in channels-last layout (cuDNN's TF32 kernels work in it: no layout conversions
        of the activations before and after every convolution; the same convolutions); ln: the packed layer norms as
        _LayerNorm (the same function)."""
        for block in self.blocks:
            block.math = bool(on)
            block.ln_fast = bool(on) if ln is None else bool(ln)
        self.nhwc = bool(on) if nhwc is None else bool(nhwc)

    def _conv_in(self, x):
        """A grid input (uint8 [B, C, H, W]) as the convolutions' float input (B18: channels-last with nhwc)."""
        x = x.float()
        return x.contiguous(memory_format=torch.channels_last) if getattr(self, 'nhwc', False) else x

    def forward(self, b, entities=None, packed=False, need_h=False, mem=None, need_vg=False, temporal=None):
        """b: the dict of to_batch. Returns (logits [B, 16], value [B], danger logits [B, 9]).
        packed: the Transformer's per-token layers on the real tokens only (Block.forward_packed; the padded door and
        entity tokens are skipped). Same values up to summation order, less work when many tokens are padding (the
        learner); not for CUDA-graph capture (the number of real tokens varies).
        mem (2026-10-11, memory models only): the GRU's state (module docstring); then the new state h_new [B, D] is
        returned as the last element.
        need_vg (2026-10-12, B22 global-token models only): value_global [B] appended as the very last element (after
        h_new). The item / global tokens (module docstring) need nothing from the caller.
        temporal (2026-10-14, B30, temporal models only; None: the temporal branch is skipped, the outputs are those
        of the model without it): a dict, mode
          'step': dict(mode='step', state=TemporalState, slot=[B] long): the actor's / evaluator's step (in place on
                  the state; the records' first / map / events / stage come from b); the step's outputs (o, s, the
                  layers' K / V, room flags) are left in temporal['out'];
          'chunk': the learner: dict(mode='chunk', plan=temporal_part(...), o_prev=[k, D] stored o_{t-1}, has=[k]
                  bool) for the first k records (chunk-major, the plan's rows), the rest (teacher records) without;
          'stored': dict(mode='stored', o=[k, D] stored o_t, o_prev, has): the stored outputs for the readout (the
                  values recomputed at an update's start).
          B34 (2026-10-16): 'chunk' / 'stored' may carry tail=dict(o_prev [n - k, D], has [n - k], o [n - k, D],
                  m [n - k]) for the records past the first k: their history token from o_prev / has and the readout
                  m x P(o) (stop-grad stored outputs: the counterfactual teacher's imitation records with the lane's
                  temporal outputs, m = 1; other teacher records m = 0, has = False: as without a tail). 'stored'
                  may carry m=[k] too: the readout mask of its first k rows (default 1).
        (B31) With a temporal context the history token is one more token (_forward_fused): host plans of a packed
        batch (packed_index / learner_plan) need hist= each record's has (False past the first k)."""
        if self.rebuild_memory:
            if temporal is not None and temporal.get('source_batch') is not None:
                return temporal['source_batch'].forward(temporal['source_indices'], need_h, need_vg)
            if temporal is not None and temporal['mode'] == 'step':
                return temporal['state'].rebuild.forward(self, b, temporal, need_h, need_vg)
            if temporal is not None:
                raise ValueError('rebuild_memory learner requires current-weight raw SourceBatch, not stored H/o/KV')
            # Context-free teacher observations have no history in their protocol.
            y, pad, u = self.encode_rebuild(b)
            rd = self.reader
            sources = u.new_zeros((1, rd.D))
            kvs = [layer.kv(sources) for layer in rd.layers]
            ix = torch.zeros((len(u), 1), dtype=torch.long, device=u.device)
            valid = torch.ones_like(ix, dtype=torch.bool)
            o = rd.read_bank(u + rd.position(torch.zeros(len(u), device=u.device), u.dtype), kvs, ix, valid)
            c = rd.fuse(o, y, pad)
            return self._heads(y[:, 0] + rd.read(o) + c, need_h,
                               yg=y[:, -1] + rd.read_g(o) + rd.fusion_g(c), need_vg=need_vg)
        if need_vg and self.gtok is None:
            raise ValueError('TokPolicy.forward: need_vg needs a global-token model (global_token=True)')
        tctx = hist = None
        if temporal is not None and self.tatt is not None:   # 2026-10-14 (B30)
            tctx, hist = self._temporal_in(b, temporal)
        if self.itok is not None or self.gtok is not None or hist is not None:   # the item / global / history tokens
            return self._forward_fused(b, entities, packed, need_h, mem, need_vg, tctx, hist)
        n, used, fast, player, room, level, entity = self._embed_old(b, entities, packed)
        if fast:   # (B18) the door mlp on the real doors only; the real tokens grouped by kind, then in packed order
            door = embed(self.door, b['doors'].reshape(n * DOOR_CAP, -1).index_select(0, b['dkeep'])) + self.kind[2]
            xp = torch.cat([player, room, level, door, entity]).index_select(0, b['xperm'])
            return self._heads(self._blocks_bucketed(xp, b).index_select(0, b['starts']), need_h, mem)
        door = embed(self.door, b['doors']) + self.kind[2]
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
            if 'buckets' in b:   # 2026-10-10 (B18): attention per length bucket (learner_plan, from the host)
                xp = self._blocks_bucketed(xp, b)
            else:
                for block in self.blocks:
                    xp = block.forward_packed(xp, bias, keep, (n, t))
            x0 = xp.index_select(0, starts)   # the player's token: the first of each record
        else:
            for block in self.blocks:
                x = block(x, bias)
            x0 = x[:, 0]
        return self._heads(x0, need_h, mem)

    def _embed_old(self, b, entities, packed):
        """forward()'s token embeddings of the old tokens (player, room, map, entities; the doors are the
        caller's): (n, entity tokens used, fast, player [n, d], room, level, entity: [n, used, d], or with the B18
        fast path the real entities [m_e, d])."""
        n = b['player'].shape[0]
        used = entities or b['ent'].shape[1]
        ids = b['ent_id'][:, :used]
        it = b['ent_item'][:, :used] if self.items else None
        ent = b['ent'][:, :used]
        price = b['ent_price'][:, :used] if self.obs_prices else None   # 2026-10-16 (B35)
        ekeep = b.get('ekeep') if packed else None
        if ekeep is not None:
            # 2026-10-10 (B18, the learner's fast path): the entity tokens' embeddings and input mlp on the real
            # entities only (ekeep: their flat positions in the [n, used] layout, from the host); the padded ones are
            # zero here and dropped by the packed layers anyway. The real tokens' values are the same
            flat = n * used
            ids = ids.reshape(flat, -1).index_select(0, ekeep)
            it = it.reshape(flat, -1).index_select(0, ekeep) if it is not None else None
            ent = ent.reshape(flat, -1).index_select(0, ekeep)
            price = price.reshape(flat, -1).index_select(0, ekeep) if price is not None else None
        ids = ids.long()
        emb = [self.type_emb(ids[..., 0]), self.variant_emb(ids[..., 1]), self.subtype_emb(ids[..., 2])]
        if self.items:
            it = it.long()
            emb += [self.item_emb(it[..., 0]), self.trinket_emb(it[..., 1])]
        parts = [ent[..., :ENT_F0]] + emb
        if self.ent_ext:   # appended last: load_compatible pads these columns with zeros
            parts.append(ent[..., ENT_F0:])
        if price is not None:   # 2026-10-16 (B35): appended after the ent_ext columns (load_compatible: zeros)
            parts.append(price_columns(price[..., 0]))
        entity = embed(self.entity, torch.cat(parts, -1)) + self.kind[3]
        fast = ekeep is not None and 'xperm' in b   # B18: the packed tokens built directly (learner_plan's xperm)
        if ekeep is not None and not fast:
            entity = entity.new_zeros((n * used, entity.shape[-1])).index_copy(0, ekeep, entity).view(n, used, -1)
        player = self._component('player', b, self._player_token)
        room = self._cnn(self.room, b, 'grid') + self.kind[1]
        if self.room_type_emb is not None:   # 2026-10-16 (B35): the room's type, added to its token
            room = room + self.room_type_emb(b['room_type'].long().clamp(0, N_ROOM_TYPE - 1)).to(room.dtype)
        level = self._cnn(self.map, b, 'map') + self.kind[4]
        return n, used, fast, player, room, level, entity

    def _component(self, name, b, fn):
        if '_encoded_components' in b:
            return b['_encoded_components'][name]
        if not self.source_cache or '_raw_rows' not in b:
            return fn(b)
        if b.get('_step_components') is not None:
            return b['_step_components'].encode(name, b, fn)
        from .rebuild_memory import ComponentCache
        if not hasattr(self, '_components'):
            self._components = ComponentCache()
        return self._components.encode(self, name, b, fn)

    def _player_token(self, b):
        pending = b['pending']
        under_way = torch.cat([F.one_hot(pending[:, i], k) for i, k in enumerate(HEADS)], -1).float()
        patch = self._cnn(self.patch, b, 'patch').float()
        parts = [b['player'], under_way, patch]
        if self.items:   # appended: load_compatible pads these columns with zeros
            inv = b['inv'].long()
            pit = b['pitem'].long()
            if self.obs_inventory_all:
                with torch.autocast('cuda', enabled=False):
                    held = b['inv_all'].float() @ self.item_emb.weight.float()
            else:
                held = self.item_emb(inv[..., 0]).sum(1)
            parts += [F.one_hot(pending[:, 3], N_ITEM).float(), F.one_hot(pending[:, 4], N_PILL).float(),
                      b['pinv'], held, self.item_emb(pit[:, 0]),
                      self.trinket_emb(pit[:, 1]), self.trinket_emb(pit[:, 2]), self.pill_emb(pit[:, 3]),
                      self.effect_emb(pit[:, 4]), self.card_emb(pit[:, 5])]
        if self.charge:   # appended last: load_compatible pads these columns with zeros
            parts.append(b['pcharge'])
        if self.obs_hearts:   # 2026-10-16 (B35): appended after pcharge (load_compatible: zeros)
            parts.append(b['phearts'])
        if self.pchar:   # 2026-10-08: the character's vector added before the first GELU (exact for old checkpoints)
            char = self.char_emb(b['pchar'].long().clamp(0, N_CHAR - 1))
            seq = self.player

            def player_mlp(x):
                return seq[2](seq[1](seq[0](x) + char.to(x.dtype)))
            player = embed(player_mlp, torch.cat(parts, -1)) + self.kind[0]
        else:
            player = embed(self.player, torch.cat(parts, -1)) + self.kind[0]
        return player

    def extra_tokens(self):
        """(B24) (item tokens per record, global tokens per record) appended to the old ones ((0, 0) without them)."""
        return (self.itok.n_tok if self.itok is not None else 0), (1 if self.gtok is not None else 0)

    def _forward_fused(self, b, entities, packed, need_h, mem, need_vg, tctx=None, hist=None):
        """forward() of a model with item / global tokens or a history token (B24): the item tokens (ItemTokens,
        compacted: the present ones first) and the global token are appended to the old tokens, [player, room, map,
        doors, entities, items, global], and every block runs ONE attention over all of them (one softmax, the same
        q / k / v projections; absent items are masked keys like absent entities). The heads read the player token,
        value_global (and with memory mem_in) the global token.
        Packed layouts (the learner): the real item tokens are the first item_count() of a record, the global token is
        each record's last real token; host plans (packed_index / learner_plan with n_item) include them.
        hist (B31, temporal models; _temporal_in's dict(y [n, width], has [n] bool)): the history token, one more
        ordinary token before the global one, [..., items, history, global]; its key masked where has is False
        (packed layouts: not a real token there; host plans with hist=has)."""
        NI, G = self.extra_tokens()
        HS = int(hist is not None)
        n, used, fast, player, room, level, entity = self._embed_old(b, entities, packed)
        if NI:
            itok, ipres = self._component('items', b, lambda part: self.itok(part, compact=True))
        if fast:   # (B18 fast path: learner_plan / plan_tokens with n_item; the real tokens grouped by kind)
            door = embed(self.door, b['doors'].reshape(n * DOOR_CAP, -1).index_select(0, b['dkeep'])) + self.kind[2]
            src = [player, room, level, door, entity]
            if NI:
                src.append(itok.reshape(n * NI, -1).index_select(0, b['ikeep']))
            if HS:   # (every record's; xperm picks the present ones)
                src.append(hist['y'].to(player.dtype))
            if G:
                src.append(self.gtok.tok.view(1, -1).expand(n, -1))
            xp = self._blocks_bucketed(torch.cat(src).index_select(0, b['xperm']), b)
            if tctx is not None and tctx['mode'] == 'rebuild_source':
                # Source workers retain only real tokens and their autograd graph;
                # the learner pads selected current observations at the read boundary.
                return xp, None
            return self._heads(xp.index_select(0, b['starts']), need_h, mem,
                               xp.index_select(0, b['glast']) if G else None, need_vg, tctx)
        door = embed(self.door, b['doors']) + self.kind[2]
        dev = player.device
        toks = [player[:, None], room[:, None], level[:, None], door, entity]
        pads = [torch.zeros((n, 3), dtype=torch.bool, device=dev),
                torch.arange(DOOR_CAP, device=dev)[None] >= b['n_doors'][:, None],
                torch.arange(used, device=dev)[None] >= b['n_ent'][:, None]]
        if NI:
            toks.append(itok)
            pads.append(~ipres)
        if HS:   # (B31)
            toks.append(hist['y'].to(player.dtype)[:, None])
            pads.append(~hist['has'].bool()[:, None])
        if G:
            toks.append(self.gtok.tok.view(1, 1, -1).expand(n, 1, -1))
            pads.append(torch.zeros((n, 1), dtype=torch.bool, device=dev))
        x = torch.cat(toks, 1)
        pad = torch.cat(pads, 1)
        if b.get('_source_bf16', False): x = x.to(torch.bfloat16)
        t = x.shape[1]
        bias = torch.zeros((n, 1, 1, t), device=dev, dtype=x.dtype).masked_fill(pad[:, None, None], -1e9)
        if packed:
            if 'keep' in b:   # (host plan: packed_index with n_item)
                keep, starts = b['keep'], b['starts']
                last = torch.cat([starts[1:], starts.new_full((1,), keep.shape[0])]) - 1
            else:
                real = ~pad
                keep = real.reshape(-1).nonzero().squeeze(1)
                counts = real.sum(1)
                starts = counts.cumsum(0) - counts
                last = starts + counts - 1
            xp = x.reshape(n * t, -1).index_select(0, keep)
            if 'buckets' in b:
                xp = self._blocks_bucketed(xp, b)
            else:
                for block in self.blocks:
                    xp = block.forward_packed(xp, bias, keep, (n, t))
            if tctx is not None and tctx['mode'] == 'rebuild_source':
                return xp.new_zeros((n * t, xp.shape[-1])).index_copy(0, keep, xp).reshape(n, t, -1), pad
            return self._heads(xp.index_select(0, starts), need_h, mem, xp.index_select(0, last) if G else None,
                               need_vg, tctx)
        for block in self.blocks:
            x = block(x, bias)
        if tctx is not None and tctx['mode'] == 'rebuild_source':
            return x, pad
        return self._heads(x[:, 0], need_h, mem, x[:, -1] if G else None, need_vg, tctx)

    def _cnn(self, seq, b, key):
        """A grid input's CNN; with b[key + '_u'] / b[key + '_i'] (B18, the learner's fast path: the distinct grids of
        the batch and each record's one, from the host) on the distinct grids only, gathered back per record (the same
        values; the gradient sums the records' terms per distinct grid)."""
        if '_encoded_components' in b and key in b['_encoded_components']:
            return b['_encoded_components'][key]
        if self.source_cache and '_raw_rows' in b:
            return self._component(key, b, lambda part: seq(self._conv_in(part[key])))
        u = b.get(key + '_u')
        if u is None:
            return seq(self._conv_in(b[key]))
        pad = getattr(self, 'cnn_bucket', 0)
        if pad and u.shape[0] % pad:   # (B38 step 4: the distinct grids padded to a multiple of cnn_bucket with copies
            #                            of the first, outputs never gathered: the convolutions see few batch sizes and
            #                            reuse their cuDNN plans; the real rows' values and gradients the same)
            u = torch.cat([u, u[:1].expand(pad - u.shape[0] % pad)])
        return seq(self._conv_in(b[key].index_select(0, u))).index_select(0, b[key + '_i'])

    def _blocks_flash(self,xp,cu_seqlens,max_seqlen):
        shape=dict(cu_seqlens=cu_seqlens,max_seqlen=max_seqlen)
        for block in self.blocks:xp=block.forward_packed(xp,None,None,shape)
        return xp

    def _blocks_bucketed(self, xp, b):
        """(B18) the blocks on the packed tokens xp with attention per length bucket (learner_plan's gpos, buckets)."""
        if b.get('_source_bf16', False): xp = xp.to(torch.bfloat16)
        if b.get('_source_flash',False):
            return self._blocks_flash(xp,b['cu_seqlens'],b['max_seqlen'])
        dev = xp.device
        ar = torch.arange(max(w_ for _, w_ in b['buckets']), device=dev)
        shape = dict(gpos=b['gpos'], total=sum(len(c_) * w_ for c_, w_ in b['buckets']), buckets=[
            (len(c_), w_, torch.zeros((len(c_), 1, 1, w_), device=dev, dtype=xp.dtype).masked_fill(
                (ar[None, :w_] >= c_[:, None])[:, None, None], -1e9)) for c_, w_ in b['buckets']])
        for block in self.blocks:
            xp = block.forward_packed(xp, None, None, shape)
        return xp

    def _value_global(self, yg, h_new=None):
        """(B22) value_global [B] from the global token after the blocks (float32, also under autocast); memory
        models: its norm + mem_out(h_new), as the other heads read the player token."""
        ctx = torch.autocast('cuda', enabled=False) if _autocast_on() else contextlib.nullcontext()
        with ctx:
            zg = self.gtok.norm(yg.float())
            if h_new is not None:
                zg = zg + self.mem_out(h_new)
            return self.gtok.value(zg)[:, 0]

    def _heads(self, x0, need_h, mem=None, yg=None, need_vg=False, tctx=None):
        """The final norm and the heads on the player's tokens x0. yg, need_vg (B22): the global token after the blocks
        and whether value_global is appended to the outputs. tctx (B30): the temporal context (forward's)."""
        if tctx is not None and tctx['mode'] == 'room_source':
            z = self.norm(x0.float())
            gn = self.gtok.norm(yg.float()) if yg is not None else None
            return self.tatt.embed(z, gn)
        if tctx is not None:   # 2026-10-14 (B30; None: the code below, unchanged)
            return self._heads_temporal(x0, need_h, mem, yg, need_vg, tctx)
        if self.memory:   # 2026-10-11 (memory models; without memory the code below, unchanged)
            return self._heads_mem(x0, need_h, mem, yg, need_vg)
        if need_vg:   # (B22) the heads as below, value_global appended
            return self._heads(x0, need_h) + (self._value_global(yg),)
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

    def _heads_mem(self, x0, need_h, mem, yg=None, need_vg=False):
        """_heads of a memory model (2026-10-11): z = norm(x0), h_new = GRU(z, h), the heads on z + mem_out(h_new);
        all in float32 (also under autocast, like the heads). h_new is appended to the outputs when mem is given.
        (B22) the GRU's input is z + mem_in(norm(global token)) (mem_in zero at the start); value_global appended last
        when need_vg."""
        ctx = torch.autocast('cuda', enabled=False) if _autocast_on() else contextlib.nullcontext()
        with ctx:
            z = self.norm(x0.float())
            if yg is not None:   # (B22)
                h_new = self.recur(z + self.gtok.mem_in(self.gtok.norm(yg.float())), mem)
            else:
                h_new = self.recur(z, mem)
            z = z + self.mem_out(h_new)
            out = (self.pi(z), self.value(z)[:, 0], self.danger(z)) + ((z,) if need_h else ())
            out = out + ((h_new,) if mem is not None else ())
            return out + ((self._value_global(yg, h_new),) if need_vg else ())

    def _temporal_in(self, b, temporal):
        """(B30) forward's temporal context (the mode's inputs and, for 'step', the records' fields) and the history
        token dict(y [n, width] float32 = Q(o_{t-1}) + kind_hist, has [n] bool: a previous step exists) for the blocks
        (B31: an ordinary token of the one attention, its key masked where has is False)."""
        n = b['player'].shape[0]
        mode = temporal['mode']
        ta = self.tatt
        if self.room_writer is not None:
            if mode == 'step' and 'room_source' not in temporal:
                raise ValueError('raw room memory step requires room_source; use TemporalState.step_inputs')
            if mode == 'chunk' and not temporal['plan'].get('raw_room_memory', False):
                raise ValueError('raw room memory chunk requires recomputed raw sources (RoomSourceBatch)')
        dev = b['player'].device
        tctx = dict(temporal, user=temporal)
        if mode == 'step':
            st, slot = temporal['state'], temporal['slot']
            first = b['first'] != 0
            tctx['rec'] = dict(first=b['first'], map=b['map'], events=b['events'], stage=b['stage'])
            o_prev = st.prev_o.index_select(0, slot)
            has = (st.steps.index_select(0, slot) > 0) & ~first
        elif mode in ('chunk', 'stored'):
            k = temporal['o_prev'].shape[0]
            o_prev, has = temporal['o_prev'].float(), temporal['has'].bool()
            tail = temporal.get('tail')
            if k < n and tail is not None:   # (B34: the records after the first k with stored outputs of their own)
                o_prev = torch.cat([o_prev.to(dev), tail['o_prev'].float().to(dev)])
                has = torch.cat([has.to(dev), tail['has'].bool().to(dev)])
            elif k < n:   # (the records after the first k: no temporal context)
                o_prev = torch.cat([o_prev, o_prev.new_zeros((n - k, o_prev.shape[1]))])
                has = torch.cat([has, has.new_zeros(n - k)])
            tctx['k'] = k
        else:
            raise ValueError(f'TokPolicy: temporal mode {mode!r}')
        ctx = torch.autocast('cuda', enabled=False) if _autocast_on() else contextlib.nullcontext()
        with ctx:
            y = ta.feed(o_prev.to(dev).float()) + ta.hist_kind
        return tctx, dict(y=y, has=has.to(dev).bool())

    def _heads_temporal(self, x0, need_h, mem, yg, need_vg, tctx):
        """(B30) _heads / _heads_mem of a temporal model: the same computation with s_t = W_s [z ; norm_g(yg)] -> o_t
        (the temporal transformer: tctx's mode) -> the heads read z (+ mem_out(h_new)) + P(o), value_global
        norm_g(yg) (+ mem_out(h_new)) + P_g(o). Rows without a temporal output (chunk / stored modes: past the first k)
        get no readout term (exactly 0)."""
        ta = self.tatt
        ctx = torch.autocast('cuda', enabled=False) if _autocast_on() else contextlib.nullcontext()
        with ctx:
            z = self.norm(x0.float())
            gn = self.gtok.norm(yg.float()) if yg is not None else None
            h_new = None
            if self.memory:
                h_new = self.recur(z + self.gtok.mem_in(gn) if gn is not None else z, mem)
                mo = self.mem_out(h_new)
                zr = z + mo
            else:
                zr = z
            s_t = ta.embed(z, gn)
            n = z.shape[0]
            mode = tctx['mode']
            if mode == 'step':
                info = ta.step(tctx['state'], s_t, tctx['slot'], tctx['rec'], tctx.get('room_source'))
                if 'room_visits' in tctx:
                    info['room_visits'] = tctx['room_visits']
                tctx['user']['out'] = info
                o, m = info['o'], z.new_ones(n)
            else:
                k = tctx['k']
                o_k = ta.sequence(s_t[:k], tctx['plan']) if mode == 'chunk' else tctx['o'].float()
                tail = tctx.get('tail')
                if k < n and tail is not None:   # (B34: stored o_t and a readout mask of their own past the first k)
                    o = torch.cat([o_k, tail['o'].float().to(o_k.device)])
                    m = torch.cat([z.new_ones(k), tail['m'].float().to(z.device)])
                else:
                    o = torch.cat([o_k, o_k.new_zeros((n - k, o_k.shape[1]))]) if k < n else o_k
                    m = torch.cat([z.new_ones(k), z.new_zeros(n - k)]) if k < n else z.new_ones(n)
                if mode == 'stored' and tctx.get('m') is not None:   # (B34: a readout mask of the first k rows)
                    m = torch.cat([tctx['m'].float().to(z.device), m[k:]])
            zr = zr + m[:, None] * ta.read(o)
            out = (self.pi(zr), self.value(zr)[:, 0], self.danger(zr)) + ((zr,) if need_h else ())
            if self.memory and mem is not None:
                out = out + (h_new,)
            if need_vg:
                zg = gn + mo if self.memory else gn
                zg = zg + m[:, None] * ta.read_g(o)
                out = out + (self.gtok.value(zg)[:, 0],)
            return out

    def recur(self, z, mem):
        """The GRU over the records' pooled representations z [N, width] (float32): mem None (zeros), a tensor
        [N, D] (one step per record) or a chunk dict (truncated BPTT: h0 [C, D], T, reset [C, T] bool where a record
        starts an episode (its input state is zeroed before its step, as the actor does), tail [R, D] or None for the
        N - C x T records after the chunks, one step each from that state (None: zeros)). Returns h_new [N, D]."""
        n = z.shape[0]
        if mem is None:
            return self.mem_gru(z, z.new_zeros((n, self.memory)))
        if torch.is_tensor(mem):
            return self.mem_gru(z, mem.float())
        h, T, reset = mem['h0'].float(), int(mem['T']), mem['reset']
        k = h.shape[0] * T
        zs = z[:k].view(h.shape[0], T, -1)
        keep = (~reset.bool()).float()
        outs = []
        for t in range(T):   # (the chunk's first state is the stored one, already zero at an episode start)
            h = self.mem_gru(zs[:, t], h * keep[:, t, None])
            outs.append(h)
        hs = torch.stack(outs, 1).reshape(k, -1)
        if k < n:
            tail = mem.get('tail')
            tail = z.new_zeros((n - k, self.memory)) if tail is None else tail.float()
            hs = torch.cat([hs, self.mem_gru(z[k:], tail)])
        return hs

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

    def act(self, b, greedy=False, entities=None, u=None, mem=None, need_vg=False, temporal=None):
        """Actions [B, heads], their log-probability [B] and the value [B]. u (2026-10-07, Phase B2): sampling
        uniforms [B, sum of head sizes] for Gumbel-max (GraphActor's rule; the common random numbers of branch
        records); None: torch.multinomial as before. mem (2026-10-11, memory models): the GRU's state [B, D]; the
        new state is then returned as a fourth element. need_vg (2026-10-14, B24; global-token models): value_global
        [B] (float32) appended last (after the new state). temporal (B30): forward's."""
        kw = dict(need_vg=True) if need_vg else {}
        if temporal is not None:
            kw['temporal'] = temporal
        if mem is not None:
            outs = self.forward(b, entities, mem=mem, **kw)
            logits, value, h_new = outs[0], outs[1], outs[3]
        else:
            outs = self.forward(b, entities, **kw)
            logits, value = outs[0], outs[1]
        tail = ((h_new,) if mem is not None else ()) + ((outs[-1].float(),) if need_vg else ())
        if u is not None and not greedy:
            actions, logp = gumbel_sample(self.split(logits.float(), self.gate(b)), u)
            return (actions, logp, value.float()) + tail
        actions, logp = [], 0.0
        for head in self.split(logits.float(), self.gate(b)):
            logs = F.log_softmax(head, -1)
            a = logs.argmax(-1) if greedy else torch.multinomial(logs.exp(), 1)[:, 0]
            actions.append(a)
            logp = logp + logs.gather(-1, a[:, None])[:, 0]
        return (torch.stack(actions, 1), logp, value.float()) + tail

    def evaluate(self, b, actions, entities=None, packed=False, mem=None, temporal=None):
        """Log-probability of the actions [B], entropy per head [B, heads], value [B], danger logits [B, 9]. mem
        (memory models): the GRU's state (forward's; the new state is not returned). temporal (B30): forward's."""
        tk = dict(temporal=temporal) if temporal is not None else {}
        if mem is not None:
            logits, value, danger = self.forward(b, entities, packed, mem=mem, **tk)[:3]
        else:
            logits, value, danger = self.forward(b, entities, packed, **tk)[:3]
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

    def log_prob_fast(self, logits, gate, actions):
        """log_prob() with the heads side by side (2026-10-10, B18, the learner's fast path): each head's logits padded
        to the widest head with -inf ([B, heads, 9]; a padded entry has probability exactly 0 and adds exactly 0 to the
        normaliser and to the entropy), the gates' -1e9 as in split(); one log-softmax for all heads. The same values
        up to the order of the sums; a few kernels instead of ~8 per head."""
        n, k = logits.shape[0], max(self.heads)
        key = (len(self.heads), logits.device)
        if getattr(self, '_lp_index', (None,))[0] != key:   # the padded layout's columns (the last: the -inf column)
            cols, at = [], 0
            for h in self.heads:
                cols += list(range(at, at + h)) + [sum(self.heads)] * (k - h)
                at += h
            self._lp_index = (key, torch.tensor(cols, device=logits.device))
        lg = torch.cat([logits.float(), logits.new_full((n, 1), float('-inf'), dtype=torch.float32)], 1)
        lg = lg.index_select(1, self._lp_index[1]).view(n, len(self.heads), k)
        if gate.dim() == 1:
            gate = (gate > 0)[:, None].expand(-1, 3)
        gated = len(self.heads) - 2   # the heads 2, 3, 4 (bomb, item, pill): their "use" entry gated
        mask = torch.zeros((n, len(self.heads), k), dtype=torch.bool, device=logits.device)
        mask[:, 2:, 1] = ~gate[:, :gated]
        logs = F.log_softmax(lg.masked_fill(mask, -1e9), -1)
        logp = logs.gather(-1, actions[:, :, None])[..., 0].sum(-1)
        return logp, -(logs.exp() * logs.clamp(min=-50)).sum(-1)


ENTITY_FIELDS = ('ent', 'ent_id', 'ent_item', 'ent_price')   # per-entity batch fields (cut to the entity tokens used;
#                                                               ent_price: 2026-10-16, B35)


def used_entities(n_ent):
    """How many entity tokens a batch needs: the largest count, in steps of 8."""
    return max(8, -(-int(n_ent.max()) // 8) * 8)


def to_batch(rows, pending, device, entities=None):
    """ROW records (a numpy structured array) and the actions under way [n, 3] as tensors on the device."""
    def t(field):
        return torch.from_numpy(np.ascontiguousarray(rows[field])).to(device, non_blocking=True)

    used = min(rows['ent'].shape[1], used_entities(rows['n_ent'])) if entities is None else int(entities)
    b = dict(player=t('player'), doors=t('doors'), patch=t('patch'), grid=t('grid'), map=t('map'), n_ent=t('n_ent'),
             n_doors=t('n_doors'), bombs=t('bombs'), inv=t('inv'), inv_all=t('inv_all'), pitem=t('pitem'), pinv=t('pinv'),
             pcharge=t('pcharge'), pchar=t('pchar'),
             first=t('first'), events=t('events'), stage=t('stage'),   # (B30: the temporal step reads them)
             phearts=t('phearts'), room_type=t('room_type'))   # (B35)
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


def packed_index(n_ent, n_doors, entities, n_item=None, item_tok=0, glob=0, hist=None):
    """forward(packed=True)'s token positions from the host's entity and door counts (numpy, one per record) for a
    batch with `entities` entity tokens: (keep, starts) int64, the flat positions of the real tokens in the [n, t]
    layout (t = 3 + DOOR_CAP + entities) and each record's first real token among them. Put into the batch dict as
    'keep' and 'starts' (tensors on its device) they spare forward() the wait for the GPU that finding them costs.
    B24 (a fused model: TokPolicy.extra_tokens() = (item_tok, glob)): n_item the records' item_count(), t grows by
    item_tok + glob (the real item tokens first, the global token always real).
    hist (B31, a temporal model's forward with its temporal context): per record whether it has the history token
    (has_prev; bool); t grows by one more slot (after the item tokens, before the global token)."""
    n = len(n_ent)
    hs = int(hist is not None)
    real = np.ones((n, 3 + DOOR_CAP + entities + item_tok + hs + glob), bool)
    real[:, 3:3 + DOOR_CAP] = np.arange(DOOR_CAP)[None] < np.asarray(n_doors)[:, None]
    real[:, 3 + DOOR_CAP:3 + DOOR_CAP + entities] = np.arange(entities)[None] < np.asarray(n_ent)[:, None]
    if item_tok:
        real[:, 3 + DOOR_CAP + entities:3 + DOOR_CAP + entities + item_tok] = \
            np.arange(item_tok)[None] < np.asarray(n_item)[:, None]
    if hs:
        real[:, 3 + DOOR_CAP + entities + item_tok] = np.asarray(hist, bool)
    counts = real.sum(1)
    return np.flatnonzero(real), np.cumsum(counts) - counts


BUCKET_COST = 2.0e5   # a bucket's fixed cost in record x token^2 units (its ~12 more kernels per layer; H20, B18)


def learner_plan(n_ent, n_doors, entities, buckets=3, n_item=None, item_tok=0, glob=0, hist=None):
    """2026-10-10 (B18, the learner's fast path): forward(packed=True)'s token layout for a batch with `entities`
    entity tokens, planned on the host per record (counts only; plan_tokens expands it per token on the device). A
    record has c = 3 + doors + entities real tokens, in slot order (player, room, map, its doors, its entities). The
    attention buckets: the records split by c into at most `buckets` groups at the bounds that minimise
    sum(records x width^2) + BUCKET_COST per group (width: the group's largest c), each group a [records, width] block
    (a record's tokens first, in order), the blocks one after another. Returns dict(rec=[n, 6] int64 per record:
    c, first packed position, doors, entities, first real door, first real entity (in record order); base=[n] its
    first position in the attention layout; m, m_doors, m_ent: the real tokens, doors, entities; t, entities;
    buckets=[(counts, width)] per group in layout order).
    B24 (a fused model, TokPolicy.extra_tokens() = (item_tok, glob)): n_item the records' item_count(); a record's real
    tokens are then player, room, map, doors, entities, its real item tokens, the global token (c grows by
    n_item + glob), rec gets two more columns (items, first real item) and m_item, item_tok, glob are set.
    hist (B31, packed_index's): the same layout (also without item / global tokens) with the history token after
    the real item tokens where hist is set (c grows by one there); rec gets a ninth column (0 / 1), 'hist' = 1."""
    ne = np.minimum(np.asarray(n_ent, np.int64), entities)
    nd = np.minimum(np.asarray(n_doors, np.int64), DOOR_CAP)
    cnt = 3 + nd + ne
    n = len(cnt)
    hs = int(hist is not None)
    fused = bool(item_tok or glob or hs)
    if fused:
        ni = np.minimum(np.asarray(n_item, np.int64), item_tok) if item_tok else np.zeros(n, np.int64)
        nh = np.asarray(hist, bool).astype(np.int64) if hs else None
        cnt = cnt + ni + glob + (nh if hs else 0)
    rec = np.empty((n, (9 if hs else 8) if fused else 6), np.int64)
    rec[:, 0], rec[:, 2], rec[:, 3] = cnt, nd, ne
    rec[:, 1] = np.cumsum(cnt) - cnt
    rec[:, 4] = np.cumsum(nd) - nd
    rec[:, 5] = np.cumsum(ne) - ne
    if fused:
        rec[:, 6], rec[:, 7] = ni, np.cumsum(ni) - ni
    if hs:
        rec[:, 8] = nh
    top = int(cnt.max())
    upto = np.cumsum(np.bincount(cnt, minlength=top + 1))   # records with c <= index
    w = np.arange(top + 1, dtype=np.float64)
    a, b = np.meshgrid(w, w, indexing='ij')                  # the bounds: (0, a], (a, b], (b, top]
    na, nb = upto[a.astype(np.int64)], upto[b.astype(np.int64)]
    n1, n2, n3 = na, nb - na, n - nb
    cost = n1 * a ** 2 + n2 * b ** 2 + n3 * float(top) ** 2 + BUCKET_COST * ((n1 > 0) + (n2 > 0) + (n3 > 0))
    cost[~((a <= b) & (b <= top))] = np.inf
    if buckets < 3:
        cost[(n1 > 0) & (n2 > 0) & (n3 > 0)] = np.inf
    if buckets < 2:
        cost[(n1 > 0) | (n2 > 0)] = np.inf
    i, j = np.unravel_index(int(np.argmin(cost)), cost.shape)
    group = np.where(cnt <= i, 0, np.where(cnt <= j, 1, 2))
    base = np.zeros(n, np.int64)
    out = dict(rec=rec, base=base, m=int(cnt.sum()), m_doors=int(nd.sum()), m_ent=int(ne.sum()),
               t=3 + DOOR_CAP + entities + item_tok + hs + glob, entities=entities, buckets=[])
    if fused:
        out.update(m_item=int(ni.sum()), item_tok=int(item_tok), glob=int(glob), hist=hs)
    at = 0
    for k in range(3):
        r = np.flatnonzero(group == k)
        if not len(r):
            continue
        width = int(cnt[r].max())
        base[r] = at + np.arange(len(r)) * width
        out['buckets'].append((cnt[r], width))
        at += len(r) * width
    return out


def plan_tokens(plan, rec, base):
    """(B18) learner_plan's per-token positions on the device, from its per-record arrays there (rec [n, 6], base [n]):
    no wait for the device (every output size is known on the host). Returns dict of int64 tensors:
      keep, starts: packed_index's (the same values: the real tokens' flat positions in the [n, t] layout, each
        record's first one among them);
      ekeep, dkeep: the real entity / door tokens' flat positions in the [n, entities] / [n, DOOR_CAP] layouts (their
        input mlps run on these only);
      xperm: each packed token's row in [players; rooms; maps; real doors; real entities] (forward builds the packed
        tokens from these);
      gpos: each packed token's position in the attention buckets' layout.
    B24 (a fused plan, rec [n, 8]): xperm's source is [players; rooms; maps; real doors; real entities; real items;
    globals], and also: ikeep (the real item tokens' flat positions in the [n, item_tok] layout of ItemTokens(compact)),
    glast (each record's last packed token: the global token). B31 (rec [n, 9], plan 'hist'): the history tokens'
    source rows [n] come after the real items' (one per record, present where rec[:, 8])."""
    if rec.shape[1] >= 8:
        return _plan_tokens_fused(plan, rec, base)
    n, dev = rec.shape[0], rec.device
    m, m_d, m_e, t, e = plan['m'], plan['m_doors'], plan['m_ent'], plan['t'], plan['entities']
    cnt, starts, nd, ne, dstart, estart = rec.unbind(1)
    ar = torch.arange(n, device=dev)
    r = torch.repeat_interleave(ar, cnt, output_size=m)        # the record of each real token
    within = torch.arange(m, device=dev) - starts[r]            # its rank inside the record
    nd_r = nd[r]
    slot = within + torch.where(within >= 3 + nd_r, DOOR_CAP - nd_r, 0)
    ekeep = torch.repeat_interleave(ar * e - estart, ne, output_size=m_e) + torch.arange(m_e, device=dev)
    dkeep = torch.repeat_interleave(ar * DOOR_CAP - dstart, nd, output_size=m_d) + torch.arange(m_d, device=dev)
    xperm = torch.where(slot < 3, slot * n + r, torch.where(slot < 3 + DOOR_CAP, 3 * n + dstart[r] + slot - 3,
                                                            3 * n + m_d + estart[r] + slot - 3 - DOOR_CAP))
    return dict(keep=r * t + slot, starts=starts, ekeep=ekeep, dkeep=dkeep, xperm=xperm, gpos=base[r] + within)


def _plan_tokens_fused(plan, rec, base):
    """(B24) plan_tokens of a fused plan (learner_plan with item tokens / a global token)."""
    n, dev = rec.shape[0], rec.device
    m, m_d, m_e, t, e = plan['m'], plan['m_doors'], plan['m_ent'], plan['t'], plan['entities']
    m_i, NI, G, HS = plan['m_item'], plan['item_tok'], plan['glob'], int(plan.get('hist', 0))
    cnt, starts, nd, ne, dstart, estart, ni, istart = rec.unbind(1)[:8]
    ar = torch.arange(n, device=dev)
    r = torch.repeat_interleave(ar, cnt, output_size=m)
    within = torch.arange(m, device=dev) - starts[r]
    nd_r, ne_r, ni_r = nd[r], ne[r], ni[r]
    b1 = 3 + nd_r          # first entity
    b2 = b1 + ne_r         # first item
    b3 = b2 + ni_r         # the history token (B31), else the global token
    zero = torch.zeros_like(within)
    slot = within + torch.where(within >= b1, DOOR_CAP - nd_r, zero) + torch.where(within >= b2, e - ne_r, zero) + \
        torch.where(within >= b3, NI - ni_r, zero)
    ekeep = torch.repeat_interleave(ar * e - estart, ne, output_size=m_e) + torch.arange(m_e, device=dev)
    dkeep = torch.repeat_interleave(ar * DOOR_CAP - dstart, nd, output_size=m_d) + torch.arange(m_d, device=dev)
    ikeep = torch.repeat_interleave(ar * NI - istart, ni, output_size=m_i) + torch.arange(m_i, device=dev)
    base_e, base_i, base_g = 3 * n + m_d, 3 * n + m_d + m_e, 3 * n + m_d + m_e + m_i
    if HS:   # (B31) [..., items, history (n rows), globals]: the history slot between them
        nh_r = rec[:, 8][r]
        b4 = b3 + nh_r     # the global token
        base_h, base_g = base_g, base_g + n
        slot = slot + torch.where(within >= b4, 1 - nh_r, zero)
        xperm = torch.where(within < 3, within * n + r, torch.where(
            within < b1, 3 * n + dstart[r] + within - 3, torch.where(
                within < b2, base_e + estart[r] + within - b1, torch.where(
                    within < b3, base_i + istart[r] + within - b2, torch.where(within < b4, base_h + r, base_g + r)))))
        return dict(keep=r * t + slot, starts=starts, ekeep=ekeep, dkeep=dkeep, xperm=xperm, gpos=base[r] + within,
                    ikeep=ikeep, glast=starts + cnt - 1)
    xperm = torch.where(within < 3, within * n + r, torch.where(
        within < b1, 3 * n + dstart[r] + within - 3, torch.where(
            within < b2, base_e + estart[r] + within - b1, torch.where(
                within < b3, base_i + istart[r] + within - b2, base_g + r))))
    return dict(keep=r * t + slot, starts=starts, ekeep=ekeep, dkeep=dkeep, xperm=xperm, gpos=base[r] + within,
                ikeep=ikeep, glast=starts + cnt - 1)


def first_of(ids, size):
    """(B18) for int ids in [0, size): (one position of each distinct id, in increasing id order; each position's
    index among them). np.unique(ids, return_index=True, return_inverse=True)[1:] without the sort (any position of
    an id serves: the rows behind equal ids are equal)."""
    pos = np.full(size, -1, np.int64)
    pos[ids] = np.arange(len(ids))
    u = np.flatnonzero(pos >= 0)
    remap = np.empty(size, np.int64)
    remap[u] = np.arange(len(u))
    return pos[u], remap[ids]


def distinct_ids(x, chunk=8192):
    """2026-10-10 (B18, the learner's fast path): per row of x (uint8 [n, ...] on the device, any strides) the id of
    its value among the distinct ones (int64 [n] on the device, ids 0 .. k-1). Exact: each row is hashed by two float64
    dot products with fixed random integer weights (integer sums below 2^53: exact), the rows are grouped by the hash
    pair (torch.unique) and every row is compared byte by byte with its group's first row; on a mismatch (a hash
    collision; not seen) every row gets its own id. One wait for the device (a call per update)."""
    n = x.shape[0]
    flat = x.reshape(n, -1)
    gen = torch.Generator(device=x.device).manual_seed(0x5EED)
    w = torch.randint(1, 1 << 20, (flat.shape[1], 2), generator=gen, device=x.device).double()
    h = torch.cat([flat[i:i + chunk].double() @ w for i in range(0, n, chunk)])
    _, inv = torch.unique(h, dim=0, return_inverse=True)
    first = torch.full((n,), n, dtype=torch.int64, device=x.device).scatter_reduce_(
        0, inv, torch.arange(n, device=x.device), 'amin')
    if bool((flat == flat.index_select(0, first.index_select(0, inv))).all()):
        return inv
    return torch.arange(n, device=x.device)


DEDUP = ('grid', 'map', 'patch')   # (B18) the grid inputs whose CNN runs on a batch's distinct values only


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
           'pitem', 'pinv', 'pcharge', 'pchar', 'first', 'events', 'stage',   # (B30: the last three)
           'ent_price', 'phearts', 'room_type', 'inv_all')   # inventory v2
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


def rows_to_device(rows, index, pending, device, stage=None, with_raw=False):
    """to_batch for a large set of records without numpy copies: rows a ROW array (any shape, C-contiguous), index the
    flat positions to take (numpy int64), pending [len(index), heads] int64. The gather runs in torch (no GIL held), one
    copy goes to the device and the fields are views of it (decode_rows). Same values as
    to_batch(rows.reshape(-1)[index], pending, device).
    stage (2026-10-10, B18): a pinned uint8 tensor [>= len(index), ROW_BYTES]: the records are gathered into it and
    copied asynchronously (the caller must not reuse it before the copy is done: a later sync of the device).
    with_raw (B18): also return the records' bytes on the device and the actions under way: (batch, bytes, pending)."""
    raw = torch.from_numpy(rows.reshape(-1).view(np.uint8).reshape(-1, ROW_BYTES))
    index_t = torch.from_numpy(np.ascontiguousarray(index, np.int64))
    if stage is not None:
        dev_raw = torch.index_select(raw, 0, index_t, out=stage[:len(index)]).to(device, non_blocking=True)
    else:
        dev_raw = raw.index_select(0, index_t).to(device, non_blocking=False)
    pend = torch.from_numpy(np.ascontiguousarray(pending, np.int64)).to(device)
    return (decode_rows(dev_raw, pend), dev_raw, pend) if with_raw else decode_rows(dev_raw, pend)


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
    Memory models (2026-10-11, TokPolicy memory): `memory` is the GRU state table [slots, D] on the device (one row per
    worker, shared by every actor that answers those workers); submit() takes each record's slot and whether its
    episode starts there. Before the replay the records' states are gathered from the table into the graph's input
    (zeroed where an episode starts), after it the new states are written back (both outside the graph, on the call's
    stream, before its completion event: a worker's next record, which can only come after the answer, sees them);
    result() leaves the input state of every record in self.h_used (numpy [n, D]: what the rollout stores).
    Global-token models (2026-10-14, B24; TokPolicy global_token): the graphs also compute value_global; result() leaves
    it in self.vg (numpy [n]: the rollout stores it like the value, the learner's global returns bootstrap from it).
    Temporal models (2026-10-14, B30; TokPolicy temporal): `temporal` is the TemporalState (shared like the GRU table);
    the step runs inside the graphs (the records' table rows are their slots; rows past the call's records, the graph's
    fixed batch, use the state's scratch slot), the state is reset after the capture; result() leaves the step's
    counters in self.t_out (dict create, nroom, steps, stage_prev: 16 bytes per record come back). B31: the step
    token's K / V, s and o never leave the device: submit(tdest=...) names the rollout's device tensors and each
    record's row there (8 bytes per record go in: with fold in the control block), and after the replay the call's
    stream copies the graph's outputs into those rows (index_copy_, outside the graph: the rollout's tensors are not
    captured and may be replaced between calls).
    """

    def __new__(cls, model, *args, **kwargs):
        if model.rebuild_memory:
            from .rebuild_actor import RebuildGraphActor
            return RebuildGraphActor(model, *args, **kwargs)
        return super().__new__(cls)

    def __init__(self, model, rows_max, entities=32, stream=None, crn=True, memory=None, bf16=False, compile_mode=None,
                 extra=None, fold=False, sdpa_math=False, row_sizes=None, temporal=None):
        """bf16 (2026-10-14, B29; train_tok --actor-bf16): the network's forward under bfloat16 autocast inside the
        graphs (the input mlps, the heads, the GRU and value_global stay float32 as in the learner's --amp bf16; the
        sampling is float32). compile_mode (B29; --actor-compile): the forward and the sampling compiled with
        torch.compile(mode=compile_mode, dynamic=False) and the compiled code captured in the graphs (fewer, fused
        kernels; float32 values up to summation order). extra (B29, a measurement aid): a callable (b, model outputs,
        h_in) -> tensor run inside the graphs after the network (its output stays on the device: self.extra_out, of the
        last graph captured). entities (B29) may also be a sequence of entity-token sizes: one graph per size (and
        ENT_CAP), a call replays the smallest that holds its records (fewer padded tokens; the same values).
        fold (B29; --actor-fold): the per-call work around the replay moves into the graphs: the uniforms are drawn
        inside (graph-safe RNG: the same distribution, other draws than without fold), the common random numbers, the
        GRU states' gather from the table and their write-back too; the host packs each record's table row, reset and
        common random numbers into a small control block sent with the records (two H2D copies, the replay, one D2H
        copy: 4 enqueued operations instead of ~12). Rows past the call's records (the graphs' fixed batch) write the
        state of the call's first record to its own row (the same value: no other worker's state is touched).
        row_sizes (B29, needs fold; --actor-row-buckets): more batch sizes of the graphs (e.g. 16, 32): a call replays
        the smallest graph that holds its records instead of always rows_max rows (the same values per record).
        sdpa_math (B29, measurement aid): attention through PyTorch's math SDPA (matrix products under the process's
        TF32 setting) instead of the float32 memory-efficient kernel. All default off: the graphs as before."""
        from .tok_obs import ENT_CAP
        dev = next(model.parameters()).device
        self.model, self.n, self.stream, self._n = model, rows_max, stream, 0
        self.room_source = torch.zeros((rows_max, model.tatt.D), device=dev) if model.room_writer is not None else None
        self.bf16, self.room_visits = bf16, None
        self.mem = int(getattr(model, 'memory', 0) or 0)   # 2026-10-11 (memory): the GRU state's size, 0 without
        self.h_used = None
        if self.mem:
            if memory is None:
                raise ValueError('GraphActor: a memory model needs its state table (memory=)')
            self.table = memory
            self.h_in = torch.zeros((rows_max, self.mem), device=dev)
            self.widx = torch.zeros(rows_max, dtype=torch.int64, device=dev)
            self.keep = torch.ones(rows_max, device=dev)
            self.widx_stage = torch.zeros(rows_max, dtype=torch.int64).pin_memory()
            self.keep_stage = torch.ones(rows_max).pin_memory()
            self.widx_np, self.keep_np = self.widx_stage.numpy(), self.keep_stage.numpy()
            self.h_outs, self._h_last = [], None
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
        self.vg_on = getattr(model, 'gtok', None) is not None   # (B24) value_global as the last output column
        self.vg = None
        # 2026-10-14 (B30, temporal models): the state, the slot of each graph row, the step's counters (create, nroom,
        # steps, stage_prev) as float columns behind value_global; (B31) the step token's K / V per layer and s (state
        # dtype) and o (float32) stay in the graphs' output tensors on the device, copied into the rollout's rows
        self.t_on = getattr(model, 'tatt', None) is not None
        self.t_out = None
        self.n_tcol = 0
        self._tdest = None   # (B31) the submitted call's rollout rows (submit's tdest)
        if self.t_on:
            if temporal is None:
                raise ValueError('GraphActor: a temporal model needs its TemporalState (temporal=)')
            self.tstate = temporal
            ta = model.tatt
            self.n_tcol = 4
            self.t_dt = temporal.dtype
            self.t_L2D = ta.L * 2 * ta.D
            self.t_outs, self._t_last = [], None
            self.tslot = torch.full((n,), temporal.scratch, dtype=torch.int64, device=dev)
            self.tslot_stage = torch.full((n,), temporal.scratch, dtype=torch.int64).pin_memory()
            self.tslot_np = self.tslot_stage.numpy()
            self.tdest = torch.zeros(n, dtype=torch.int64, device=dev)   # (B31: the records' rollout rows, no fold)
            self.tdest_stage = torch.zeros(n, dtype=torch.int64).pin_memory()
            self.tdest_np = self.tdest_stage.numpy()
        self.out_host = torch.zeros((n, nh + 2 + self.n_extra + self.mem + int(self.vg_on) + self.n_tcol),
                                    dtype=torch.float32).pin_memory()
        self.out_np = self.out_host.numpy()
        self.done = torch.cuda.Event()
        ents = [entities] if np.ndim(entities) == 0 else list(entities)
        self.sizes = sorted({min(int(e), ENT_CAP) for e in ents} | {ENT_CAP})
        self.entities = self.sizes[0]
        self.fold = bool(fold)
        if row_sizes and not self.fold:
            raise ValueError('GraphActor: row_sizes needs fold=True')
        self.rsizes = sorted({min(int(r_), n) for r_ in (row_sizes or ())} | {n})
        if self.fold:   # (B29) the control block per row: table row (int64), keep, dup, crn flag (float32), crn uniforms
            H = sum(heads)
            cw = -(-(20 + 4 * H) // 8) * 8
            self.c_dest_at = cw   # (B31, temporal: the record's rollout row (int64) after them)
            if self.t_on:
                cw += 8
            self.ctrl_stage = torch.zeros((n, cw), dtype=torch.uint8).pin_memory()
            self.ctrl = torch.zeros((n, cw), dtype=torch.uint8, device=dev)
            cs = self.ctrl_stage.numpy()
            self.c_widx, self.c_keep, self.c_dup, self.c_cf = (cs[:, 0:8].view(np.int64)[:, 0], cs[:, 8:12].view(
                np.float32)[:, 0], cs[:, 12:16].view(np.float32)[:, 0], cs[:, 16:20].view(np.float32)[:, 0])
            self.c_cu = cs[:, 20:20 + 4 * H].view(np.float32)
            self.c_dest = cs[:, cw - 8:cw].view(np.int64)[:, 0] if self.t_on else None
            self.c_dup[:] = 1.0
            self.c_keep[:] = 1.0
            # (the staging copy skips the entity rows a graph does not read: ROW 'ent' rows past its size; their bytes
            # in the staging buffer and on the device are stale and never decoded)
            dt_e, off_e = ROW.fields['ent'][:2]
            self._ent_span = (off_e, dt_e.itemsize // dt_e.shape[0], off_e + dt_e.itemsize)

        mem_on, choice_on, vg_on, t_on = bool(self.mem), self.choice, self.vg_on, self.t_on

        def core(b, u, h_in, tokens, tslot=None):
            """The network and the sampling on decoded records b: (output columns [n, k], the new GRU state or None,
            the extra aid's output, (B30) the step's K / V and s or None)."""
            kw = dict(need_vg=True) if vg_on else {}   # (B24: value_global appended last)
            tdict = None
            if t_on:   # (B30) the temporal step, in place on the state
                tdict = dict(mode='step', state=self.tstate, slot=tslot)
                if self.room_source is not None:
                    tdict['room_source'] = self.room_source[:tslot.shape[0]]
                kw['temporal'] = tdict
            h_last = None
            with (torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()), \
                    (_sdpa_math() if sdpa_math else contextlib.nullcontext()):
                if mem_on:   # (memory: the state's input from h_in, the new state kept for the write-back)
                    outs = model(b, tokens, need_h=choice_on, mem=h_in, **kw)
                    logits, value, h, h_last = (outs[0], outs[1], outs[3] if choice_on else None,
                                                outs[-2] if vg_on else outs[-1])
                elif choice_on:
                    outs = model(b, tokens, need_h=True, **kw)
                    logits, value, h = outs[0], outs[1], outs[3]
                else:
                    outs = model(b, tokens, **kw)
                    logits, value = outs[0], outs[1]
            actions, logp, at = [], 0.0, 0
            for head in model.split(logits.float(), model.gate(b)):
                k = head.shape[1]
                logs = F.log_softmax(head, -1)
                gumbel = -torch.log(-torch.log(u[:, at:at + k].clamp(1e-10, 1 - 1e-7)))
                a = torch.argmax(logs + gumbel, -1)
                actions.append(a)
                logp = logp + logs.gather(-1, a[:, None])[:, 0]
                at += k
            cols = [torch.stack(actions, 1).float(), logp[:, None], value.float()[:, None]]
            if choice_on:
                item, has = ChoiceHeads.nearest_pedestal(b)
                d = model.choice.take_minus_skip(h.float(), item).float()
                keep = has.float()[:, None]
                cols += [d.mean(1, keepdim=True) * keep, d.std(1, keepdim=True) * keep]
            if mem_on:   # (memory: each record's input state, for the rollout)
                cols.append(h_in)
            if vg_on:   # (B24) value_global
                cols.append(outs[-1].float()[:, None])
            tkv = None
            if t_on:   # (B30) the step's outputs for the rollout: the counters to the host, (B31) the rest stays
                ti = tdict['out']
                cols += [ti['create'].float()[:, None], ti['nroom'].float()[:, None],
                         ti['steps'].float()[:, None], ti['stage_prev'].float()[:, None]]
                tkv = (torch.cat([ti['kv'].reshape(ti['kv'].shape[0], -1), ti['s'].to(ti['kv'].dtype)], 1),
                       ti['o'].float())
            ex = extra(b, outs, h_in) if extra is not None else None   # (B29 measurement aid: stays on the device)
            return torch.cat(cols, 1), h_last, ex, tkv

        if compile_mode:   # (every actor, graph size and model is a recompile of core: above dynamo's default limit
            # of 8 the calls would silently run uncompiled)
            import torch._dynamo as _dynamo
            dcfg = _dynamo.config
            for name_, v_ in (('recompile_limit', 256), ('cache_size_limit', 256), ('accumulated_recompile_limit', 4096),
                              ('accumulated_cache_size_limit', 4096)):
                if hasattr(dcfg, name_):
                    setattr(dcfg, name_, max(int(getattr(dcfg, name_)), v_))
        fn = core if not compile_mode else torch.compile(core, mode=compile_mode, dynamic=False)

        def run(tokens, r=None):
            if not self.fold:
                b = decode_rows(self.raw, entities=tokens, n_heads=nh)
                out, self._h_last, self.extra_out, self._t_last = fn(b, self.u, self.h_in if self.mem else None, tokens,
                                                                     self.tslot if t_on else None)
                return out
            b = decode_rows(self.raw[:r], entities=tokens, n_heads=nh)
            c = self.ctrl[:r]   # (B29 fold: the control block decoded on the device, the per-call work inside the graph)
            widx, keep, dup, cf = (c[:, 0:8].view(torch.int64)[:, 0], c[:, 8:12].view(torch.float32)[:, 0],
                                   c[:, 12:16].view(torch.float32)[:, 0], c[:, 16:20].view(torch.float32)[:, 0])
            ur = self.u[:r]
            ur.uniform_()
            u = torch.where(cf[:, None] > 0, c[:, 20:20 + 4 * self.u.shape[1]].view(torch.float32), ur)
            h_in = self.table.index_select(0, widx) * keep[:, None] if self.mem else None
            tslot = torch.where(dup > 0, torch.full_like(widx, self.tstate.scratch), widx) if t_on else None
            out, h_last, self.extra_out, self._t_last = fn(b, u, h_in, tokens, tslot)
            if self.mem:
                self.table.index_copy_(0, widx, torch.where(dup[:, None] > 0, h_last[0:1], h_last))
            self._h_last = h_last
            return out

        self.graphs, self.outs = [], []
        pool = None
        # captured on a stream of this actor's own: cuBLAS keeps one workspace per stream and a graph keeps the one of
        # the stream it was captured on. On torch.cuda.graph's default capture stream (shared by every graph) two
        # actors replayed at once (train_tok --actor-slots 2) shared one workspace and corrupted each other's
        # split-K products: the value head's output (seen as wrong stored values in about 9% of the decisions)
        capture = torch.cuda.Stream()
        with torch.inference_mode():
            for tokens, r in [(t_, r_) for r_ in self.rsizes for t_ in self.sizes]:   # (graph index: r x sizes + t)
                for _ in range(3):   # allocations and algorithm choices happen before the capture
                    run(tokens, r)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=capture):   # never replayed at once: one memory pool
                    out = run(tokens, r)
                pool = graph.pool()
                self.graphs.append(graph)
                self.outs.append(out)
                if self.mem:
                    self.h_outs.append(self._h_last)
                if t_on:
                    self.t_outs.append(self._t_last)
        torch.cuda.synchronize()
        if t_on:   # (B30) the warm-up and capture calls stepped the state: back to the start
            self.tstate.reset_()
            torch.cuda.synchronize()

    def submit(self, rows, pending, widx=None, reset=None, tdest=None, ukeys=None):
        """Start a call (returns at once; False if it does not fit). rows: a C-contiguous ROW array, pending [n, heads]
        int64. The next submit may only follow this call's result(). Memory models: widx [n] the records' rows of the
        state table (distinct), reset [n] bool (None: none) where an episode starts (its state is zero).
        tdest (B31, temporal models; train_tok Rollout.t_dest): dict(kv [rows, L, 2, D], s [rows, D], o [rows', D]
        device tensors, obase (o's row = kv's row + obase), rows [n] int64 numpy: each record's row); None: the step's
        K / V, s and o are not kept.
        ukeys (B36, 2026-10-16, the lanes): (keys [n], t [n]) int64: every record's sampling uniforms are
        crn_uniforms(keys, t) (counter-based: a record's draw depends on its key and t only, never on the call's other
        records or its row in the call); None: as before."""
        n = len(rows)
        if n > self.n:
            return False
        self._tdest = tdest if self.t_on else None
        if self.room_source is not None:
            from .room_memory import prepare_room_step
            stream = self.stream if self.stream is not None else torch.cuda.current_stream()
            with torch.cuda.stream(stream), torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.bf16):
                source, self.room_visits = prepare_room_step(self.model, self.tstate, rows, pending, widx)
                self.room_source[:n].copy_(source)
        if self.fold:
            return self._submit_fold(rows, pending, widx, reset, ukeys)
        if self.mem:
            self.widx_np[:n] = widx
            self.keep_np[:n] = 1.0 if reset is None else (~np.asarray(reset, bool)).astype(np.float32)
        if self.t_on:   # (B30) the records' slots; the graph's other rows on the scratch slot
            self.tslot_np[:n] = widx
            self.tslot_np[n:] = self.tstate.scratch
            if self._tdest is not None:   # (B31) the records' rollout rows
                self.tdest_np[:n] = self._tdest['rows']
        most = int(rows['n_ent'].max())
        g = 0 if most <= self.sizes[0] else (len(self.sizes) - 1 if len(self.sizes) == 2 else
                                             int(np.searchsorted(self.sizes, most)))
        self.stage_np[:n, :ROW_BYTES] = np.ascontiguousarray(rows).view(np.uint8).reshape(n, ROW_BYTES)
        self.stage_np[:n, ROW_BYTES:] = np.ascontiguousarray(pending, np.int64).view(np.uint8).reshape(n, 8 * self.nh)
        stream = self.stream if self.stream is not None else torch.cuda.current_stream()
        sel = None
        if ukeys is not None:   # (B36) every record's uniforms from its key
            sel = np.arange(n)
            self.crn_stage_np[:n] = crn_uniforms(ukeys[0], ukeys[1], self.u.shape[1])
            self.crn_idx_np[:n] = sel
        elif self.crn and 'crn' in rows.dtype.names:
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
            if self.mem:   # the records' states into the graph's input (zero at an episode start)
                self.widx[:n].copy_(self.widx_stage[:n], non_blocking=True)
                self.keep[:n].copy_(self.keep_stage[:n], non_blocking=True)
                self.h_in[:n] = self.table.index_select(0, self.widx[:n]) * self.keep[:n, None]
            if self.t_on:   # (B30)
                self.tslot.copy_(self.tslot_stage, non_blocking=True)
                if self._tdest is not None:
                    self.tdest[:n].copy_(self.tdest_stage[:n], non_blocking=True)
            self.graphs[g].replay()
            if self.mem:   # the new states back into the table
                self.table.index_copy_(0, self.widx[:n], self.h_outs[g][:n])
            self.out_host[:n].copy_(self.outs[g][:n], non_blocking=True)
            if self._tdest is not None:   # (B31) the step's K / V, s and o into the rollout's rows, on the device
                self._t_store(g, n, self.tdest[:n])
            self.done.record()
        self._n = n
        return True

    def _submit_fold(self, rows, pending, widx, reset, ukeys=None):
        """submit() with fold (B29): records + control block in, the replay, the outputs out."""
        g = self._pack_fold(rows, pending, widx, reset, ukeys)
        n = len(rows)
        r = self.rsizes[int(np.searchsorted(self.rsizes, n))]
        g += self.rsizes.index(r) * len(self.sizes)
        stream = self.stream if self.stream is not None else torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            self.raw[:n].copy_(self.stage[:n], non_blocking=True)
            self.ctrl[:r].copy_(self.ctrl_stage[:r], non_blocking=True)
            with torch.inference_mode():   # (the graphs draw uniforms: replay advances the RNG state they captured,
                self.graphs[g].replay()    # an inference tensor since the capture ran in inference mode)
            self.out_host[:n].copy_(self.outs[g][:n], non_blocking=True)
            if self._tdest is not None:   # (B31) the step's K / V, s and o into the rollout's rows, on the device
                a_ = self.c_dest_at
                self._t_store(g, n, self.ctrl[:n, a_:a_ + 8].view(torch.int64)[:, 0].contiguous())
            self.done.record()
        self._n = n
        return True

    def _t_store(self, g, n, rows):
        """(B31) graph g's step outputs of the call's n records into the rollout tensors of self._tdest at rows (a
        device int64 tensor [n]); on the call's stream, after the replay."""
        td = self._tdest
        tkv, o = self.t_outs[g]
        kv = td['kv']
        L2D = self.t_L2D
        kv.index_copy_(0, rows, tkv[:n, :L2D].reshape((n,) + tuple(kv.shape[1:])).to(kv.dtype))
        td['s'].index_copy_(0, rows, tkv[:n, L2D:].to(td['s'].dtype))
        td['o'].index_copy_(0, rows + int(td.get('obase', 0)), o[:n].to(td['o'].dtype))

    def _pack_fold(self, rows, pending, widx, reset, ukeys=None):
        """(fold) The records and the control block into the pinned staging buffers; returns the graph to replay."""
        n = len(rows)
        most = int(rows['n_ent'].max())
        g = int(np.searchsorted(self.sizes, most))
        src = np.ascontiguousarray(rows).view(np.uint8).reshape(n, ROW_BYTES)
        off_e, per_e, end_e = self._ent_span
        cut = off_e + per_e * self.sizes[g]
        self.stage_np[:n, :cut] = src[:, :cut]
        self.stage_np[:n, end_e:ROW_BYTES] = src[:, end_e:]
        self.stage_np[:n, ROW_BYTES:] = np.ascontiguousarray(pending, np.int64).view(np.uint8).reshape(n, 8 * self.nh)
        if self.mem or self.t_on:   # ((B30) temporal: the slot of each row; the padded rows go to the scratch slot)
            self.c_widx[:n] = widx
            self.c_widx[n:] = widx[0]
        if self.mem:
            self.c_keep[:n] = 1.0 if reset is None else (~np.asarray(reset, bool)).astype(np.float32)
        if self._tdest is not None:   # (B31) each record's rollout row
            self.c_dest[:n] = self._tdest['rows']
        self.c_dup[:n] = 0.0
        self.c_dup[n:] = 1.0
        self.c_cf[:] = 0.0
        if self.crn and 'crn' in rows.dtype.names:
            crn = rows['crn']
            if crn.any():
                sel = np.flatnonzero(crn)
                self.c_cu[sel] = crn_uniforms(crn[sel], rows['t'][sel], self.u.shape[1])
                self.c_cf[sel] = 1.0
                self.crn_rows += len(sel)
        if ukeys is not None:   # (B36) every record's uniforms from its key (the graph's own draws unused)
            self.c_cu[:n] = crn_uniforms(ukeys[0], ukeys[1], self.u.shape[1])
            self.c_cf[:n] = 1.0
        return g

    def result(self):
        """(actions [n, heads] int64, log-probabilities [n], values [n]) of the submitted call, as numpy arrays."""
        self.done.synchronize()
        out = self.out_np[:self._n].copy()
        nh = self.nh
        self.extra = out[:, nh + 2:nh + 2 + self.n_extra] if self.n_extra else None
        if self.mem:   # (memory) each record's input state
            self.h_used = out[:, nh + 2 + self.n_extra:nh + 2 + self.n_extra + self.mem]
        at = nh + 2 + self.n_extra + self.mem
        self.vg = out[:, at] if self.vg_on else None   # (B24) value_global
        if self.t_on:   # (B30) the step's counters (B31: K / V, s and o went to the rollout on the device)
            tc = out[:, at + int(self.vg_on):]
            self.t_out = dict(create=tc[:, 0] > 0.5, nroom=np.rint(tc[:, 1]).astype(np.int64),
                              steps=np.rint(tc[:, 2]).astype(np.int64), stage_prev=np.rint(tc[:, 3]).astype(np.int64))
            if self.room_source is not None:
                self.t_out['room_visits'] = self.room_visits
        return out[:, :nh].astype(np.int64), out[:, nh], out[:, nh + 1]

    def act(self, rows, pending, widx=None, reset=None, tdest=None, ukeys=None):
        """submit + result: (actions, log-probabilities, values), or None if it does not fit."""
        return self.result() if self.submit(rows, pending, widx, reset, tdest, ukeys) else None


def _sdpa_math():
    """(B29) scaled_dot_product_attention through PyTorch's math backend inside the context."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel(SDPBackend.MATH)


def memory_chunks(length, mask, T):
    """2026-10-11 (memory, the learner's truncated BPTT): a rollout's decisions cut into chunks of T per worker.
    length [W]: decisions stored per worker; mask [W, cap] bool: the decisions trained on. Each worker's sequence
    0 .. length-1 (untrained decisions included: the actor's GRU stepped through them too) is cut at 0, T, 2T, ...; a
    worker's last chunk is padded to T by repeating its last decision (weight 0); chunks without a trained decision are
    dropped. Returns dict(pos [C*T] int64 flat positions w * cap + t (chunk-major), weight [C*T] float32 (1: trained),
    start [C] int64 the flat position of each chunk's first decision (its stored state starts the chunk), chunks C)."""
    W, cap = mask.shape
    ar = np.arange(T)
    pos, weight, start = [], [], []
    for w in range(W):
        n = int(length[w])
        for s in range(0, n, T):
            e = min(s + T, n)
            m = np.zeros(T, bool)
            m[:e - s] = mask[w, s:e]
            if not m.any():
                continue
            pos.append(w * cap + np.minimum(s + ar, e - 1))
            weight.append(m)
            start.append(w * cap + s)
    if not pos:
        return dict(pos=np.zeros(0, np.int64), weight=np.zeros(0, np.float32), start=np.zeros(0, np.int64), chunks=0)
    return dict(pos=np.concatenate(pos).astype(np.int64), weight=np.concatenate(weight).astype(np.float32),
                start=np.asarray(start, np.int64), chunks=len(start))


def pad_optimizer_state(osd, n_params):
    """2026-10-11 (memory): an optimiser state dict of a model with fewer parameters (a memory-free checkpoint) made
    loadable into an optimiser over n_params parameters whose first ones are the checkpoint's (the memory modules are
    registered last): the single param group gets the new parameters' ids, with no state (Adam initialises it at the
    first step). Returns the number of parameters added."""
    g = osd['param_groups'][0]
    have = len(g['params'])
    if len(osd['param_groups']) != 1 or have >= n_params:
        return 0
    top = max(list(g['params']) + [-1])
    g['params'] = list(g['params']) + list(range(top + 1, top + 1 + n_params - have))
    return n_params - have


def remap_optimizer_state(osd, old_names, new_names, drop=()):
    """2026-10-12 (B21 / B22): an optimiser state dict (one param group) saved for the parameters old_names (in its
    order) made loadable into an optimiser over new_names: each parameter keeps the state of the same name, new ones
    start without state (Adam initialises it at the first step). General form of pad_optimizer_state (any new
    parameters anywhere, e.g. memory added to a checkpoint that already has item tokens). drop (B23): names whose saved
    state is not carried over (a tensor whose shape changed: share_item_table's ITEM_SHARED_KEY). Returns the number of
    parameters without state. Raises ValueError if the group's size is not len(old_names)."""
    g = osd['param_groups'][0]
    if len(osd['param_groups']) != 1 or len(g['params']) != len(old_names):
        raise ValueError(f"remap_optimizer_state: {len(g['params'])} saved parameters for {len(old_names)} names")
    old_id = dict(zip(old_names, g['params']))
    state = {}
    for j, name in enumerate(new_names):
        i = old_id.get(name) if name not in drop else None
        if i is not None and i in osd['state']:
            state[j] = osd['state'][i]
    added = sum(name not in old_id or name in drop for name in new_names)
    osd['state'] = state
    g['params'] = list(range(len(new_names)))
    return added


def main_parameter_names(names):
    """(B21 / B22) The names of the parameters PPO's optimiser owns (TokPolicy.main_parameters' order) among a state
    dict's keys or named_parameters() names (TokPolicy has no buffers: a state dict's keys are its parameters, in the
    order of parameters())."""
    return [k for k in names if not k.startswith('choice.')]


ITEM_SHARED_KEY = 'itok.mlp.0.weight'   # B23: the item tokens' first Linear (its collectible columns come first)


def share_item_table(model, state):
    """B23 (2026-10-13): a B21 checkpoint's state dict (item tokens with their own 'itok.coll.weight' table, IT_COLL =
    32 columns) made loadable into a model with item_shared (the item tokens read item_emb, ITEM_EMB = 16 columns).
    The old table is dropped; the first Linear of the item-token mlp ('itok.mlp.0.weight', [width, 32 + 44]) becomes
    [width, 16 + 44]: its 44 non-collectible columns (trinket, pill, effect, card embeddings, count, count > 1, charge
    share, max charge) are the checkpoint's, its 16 collectible columns keep the model's fresh initialisation (torch's
    nn.Linear default, U(-1/sqrt(60), 1/sqrt(60)), drawn when the model was built). Every other tensor (bias,
    slot / kind embeddings, the rest of the mlp, item_emb itself) is the checkpoint's. Returns (state, a log line or
    None when nothing was converted). The input dict is not changed."""
    itok = getattr(model, 'itok', None)
    if itok is None or not itok.shared or 'itok.coll.weight' not in state:
        return state, None
    state = dict(state)
    old = state.pop('itok.coll.weight')
    own = model.state_dict()[ITEM_SHARED_KEY]
    w_old = state.get(ITEM_SHARED_KEY)
    rest = own.shape[1] - itok.dc
    if w_old is None or w_old.shape[0] != own.shape[0] or w_old.shape[1] - old.shape[1] != rest:
        raise ValueError(f'share_item_table: {ITEM_SHARED_KEY} {tuple(w_old.shape) if w_old is not None else None} does '
                         f'not fit a {old.shape[1]}-column collectible table and {rest} other columns')
    w = own.detach().clone().to(device=w_old.device, dtype=w_old.dtype)
    w[:, itok.dc:] = w_old[:, old.shape[1]:]
    state[ITEM_SHARED_KEY] = w
    return state, (f"item_shared (B23): the checkpoint's own item-token table itok.coll ({old.shape[0]} x {old.shape[1]}) "
                   f'dropped, the item tokens now read item_emb ({itok.dc} columns); {ITEM_SHARED_KEY} '
                   f'{tuple(w_old.shape)} -> {tuple(w.shape)}: its {rest} other columns kept, its {itok.dc} collectible '
                   "columns re-initialised (the fresh model's nn.Linear default); everything else loaded as saved")


MEMORY_PREFIXES = ('mem_gru.', 'mem_out.', 'gtok.mem_in.')   # B32: the GRU memory's tensors (B20; B22's mem_in)


def drop_memory(model, state):
    """B32 (2026-10-15): a GRU-memory checkpoint's state dict (B20 --memory gru: mem_gru, mem_out and, with a global
    token, gtok.mem_in) made loadable into the same model without memory (TokPolicy memory=0; train_tok --resume with
    --memory off). Exactly those tensors are dropped, every other one is kept as saved. The model then computes the
    checkpoint's outputs with the GRU's contribution removed: mem_out(h_t) (its weight and its bias) no longer added
    to the heads' input z and to value_global's input, i.e. the checkpoint's model with mem_out zeroed (mem_in only fed
    the GRU, whose state reaches the outputs through mem_out alone; tests/test_no_gru_smoke.py (c)). Returns (state,
    dropped names, a log line), (state, [], None) when there is nothing to drop. Raises ValueError if the model has
    memory or a dropped name is one of its tensors. The input dict is not changed."""
    names = [k for k in state if k.startswith(MEMORY_PREFIXES)]
    if not names:
        return state, [], None
    if int(getattr(model, 'memory', 0) or 0):
        raise ValueError('drop_memory: the model has memory (TokPolicy memory > 0)')
    own = model.state_dict()
    clash = [k for k in names if k in own]
    if clash:
        raise ValueError(f'drop_memory: {clash[:4]} are tensors of the model')
    n_par = sum(int(state[k].numel()) for k in names)
    size = int(state['mem_out.weight'].shape[1]) if 'mem_out.weight' in state else 0
    mag = ''
    if 'mem_out.weight' in state and 'mem_out.bias' in state:
        w_, b_ = state['mem_out.weight'].float(), state['mem_out.bias'].float()
        mag = (f'; mem_out |W| {float(w_.norm()):.3f} (max row {float(w_.norm(dim=1).max()):.3f}), |b| '
               f'{float(b_.norm()):.3f}')
    note = (f'memory (B32): --resume of a GRU checkpoint (size {size}) with --memory off: dropped {len(names)} tensors '
            f'({n_par / 1e6:.3f}M parameters: {", ".join(names)}) and their Adam state; every other tensor exact. The '
            'outputs change by exactly what the GRU contributed: they equal the checkpoint\'s with mem_out(h_t) removed '
            '(mem_out weight and bias zero; mem_in only fed the GRU) from the heads\' and value_global\'s input' + mag)
    return {k: v for k, v in state.items() if k not in names}, names, note


def load_compatible(model, state):
    """Load a checkpoint's weights into a model whose input layers may have grown (more door features, ...): a
    tensor of another shape is copied into the leading part of the model's (the rest keeps its fresh values, zero
    for added input columns). Returns the names that did not match exactly. (B23: a B21 checkpoint's own item-token
    table into an item_shared model goes through share_item_table first, reported as 'itok.coll.weight (shared: ...)'.)"""
    if model.room_writer is None and any(k.startswith('room_writer.') for k in state):
        raise ValueError('room_memory checkpoint requires a room_memory=True model; writer cannot be dropped')
    state, note = share_item_table(model, state)
    own = model.state_dict()
    changed = [] if note is None else ['itok.coll.weight (shared: ' + note + ')']
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
