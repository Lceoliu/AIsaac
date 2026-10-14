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

Extra tokens (2026-10-12, B21 item tokens / B22 global token; TokPolicy(item_tokens=True) needs items=True,
TokPolicy(global_token=True); config keys 'item_tokens', 'global_token', only when set). A fixed number G of tokens per
record (static shapes for the CUDA graphs) beside the old ones (player, room, map, doors, entities):
  B21 (ItemTokens, 20): one per held collectible (ROW 'inv', up to INV_CAP = 16, newest first: a collectible table of
  its own (dim 32), the count and count > 1, a slot embedding), one for the active item (its id in the same table,
  'pinv' columns 0-1: charge share, max charge / 12), one per trinket (2), one for the pocket (pill colour, pill effect,
  card); each token = mlp(those inputs) + a type embedding (held, active, trinket, pocket). Absent ones are masked as
  keys (-1e9 in the attention bias, as absent entities). The old multi-hot / pinv inputs of the player token stay.
  B22 (GlobalToken, 1): a learned vector; a second value head `value_global` reads it (after a norm of its own; with
  memory + mem_out(h_new) as the other heads); with memory the GRU's input is z + mem_in(norm(global token)) (mem_in
  zero at the start: the concatenation of the two through a zero-initialised projection of the global part).
Why not simply zero tokens in the one softmax: a zero token is not neutral there. Its key is W_k LN(0) + b_k =
W_k beta + b_k (beta = n1's bias), not zero, so it takes softmax mass from every query with a non-zero bias, and its
value W_v beta + b_v is mixed in; even a key masked with -1e9 changes the softmax's and the p @ v product's length and
so their summation order (not bit-identical, see load_compatible's note on padded columns). A key bias log(g) would give
exactly zero weight at g = 0 but its gradient is 0 x 1/g (NaN), and any exactly-zero softmax weight has a zero
gradient through the softmax. So the old tokens' attention is left exactly as it was (same tensors, same calls) and the
extra tokens reach them through separate, gated terms (Block._ext): for every old query, per layer l,
  a_old = softmax(q k_old^T + bias) v_old   (unchanged)
          + gate_items[l] * has_item * softmax(q k_items^T + item mask) v_items
          + gate_global[l] * v_global          (one key: its softmax weight is 1)
before the block's output projection; the gates are scalars per layer, zero at the start. The extra tokens themselves
are updated like any token (their queries attend over every real old and extra key in one softmax, then proj and the
feed-forward of the same block). At gates 0 every gated term is exactly +-0 (its factors are finite: a softmax over
all-masked keys is a finite mean, has_item = 0 removes it), a_old + 0 = a_old, and the policy, value, danger and memory
outputs are bit-identical to a model without the options (tests/test_item_global_smoke.py). Gradients: d loss / d gate
= <term, d loss / d a_old> is non-zero from the first step (the gates open by themselves); everything upstream of a gate
(the item embeddings, the global token's vector) gets a zero gradient through that path until its gate is non-zero
(the same as the GRU behind B20's zero mem_out). value_global's loss reaches the item tokens at once (the global token
attends to them inside the extra tokens' own softmax). Cost: G more tokens per record through every layer (queries
over t + G keys) and a G-key cross term for every old token: about G / t more per-token work (actor: t = 43 at 32
entity tokens, G = 21). The learner's packed paths compute the extra attention in the padded [records, t] layout (the
old tokens' q / k / v scattered there once per layer). The new modules are registered after everything else (memory
included): an optimiser state without them is remapped by parameter names (remap_optimizer_state).

Shared collectible table (2026-10-13, B23; TokPolicy(item_shared=True), needs item_tokens, config key 'item_shared'):
ItemTokens reads the policy's item_emb (ITEM_EMB = 16 columns, the pedestal ids' and the multi-hot's table) instead of
a table of its own (IT_COLL = 32): one representation of a collectible whether it lies on a pedestal or is held. The
table is not registered a second time (one parameter, one state-dict key). A B21 checkpoint is converted by
share_item_table (its own table dropped, the item-token mlp's first Linear keeps its 44 other columns and gets fresh
collectible columns); off: the B21 module, bit for bit (tests/test_pickup_smoke.py).

Fused extra tokens and item slots (2026-10-14, B24; TokPolicy(item_tokens_fused=True), config key 'item_tokens_fused';
TokPolicy(item_slots=N), config key 'item_slots' when N != INV_CAP; both need item / global tokens). The separate gated
terms above cost an extra attention per layer over a padded copy of every record's q / k / v (the learner: 2.25 x the
time of an update without them, C94). Fused (_forward_fused): the N + 4 item tokens and the global token are appended to
the old tokens and every block runs ONE attention over all of them (one softmax; the same LayerNorm, q / k / v
projections, output projection and feed-forward as every token); absent items are masked keys like absent entities.
The per-layer gates stay as multipliers of the item / global tokens' value vectors (Block vscale; 1 at a fresh start).
The heads read the player token; value_global (and with memory mem_in) the global token. The learner's packed layouts
treat the present item tokens like entities (ItemTokens(compact=True) moves them to the front, item_count() gives
their number on the host, packed_index / learner_plan / plan_tokens take it): no padded copies, no cross terms.
item_slots N: the newest N held collectibles get tokens (slot embeddings 0 .. N-1 of the INV_CAP table: the parameters
keep their shapes). A B21 / B22 / B23 checkpoint loads into a fused model with every tensor as saved (the same names and
shapes, so Adam's state too; convert_to_fused): the gates keep their values as value multipliers. That is NOT the same
function (one softmax cannot reproduce softmax_old + gate x softmax_items: the extra keys take attention mass from the
old ones), so the outputs change at the switch; tests/test_fused_tokens_smoke.py (c) measures by how much.
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

from .tok_obs import (DOOR_CAP, DOOR_F, ENT_F, ENT_F0, GRID, INV_CAP, MAP, N_CARD, N_CHAR, N_COLLECTIBLE,
                      N_PILL_COLOR, N_PILL_EFFECT, N_TRINKET, PATCH, PCHARGE_F, PINV_F, PLAYER_F, ROW)

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
        self.math = False      # 2026-10-10 (B18): attention() as _MathAttention (TokPolicy.set_math_attention)
        self.ln_fast = False   # (B18) the packed layer norms as _LayerNorm

    def forward(self, x, bias, ext=None, vscale=None):
        """vscale (B24 fused): [t] multipliers of the tokens' value vectors (1 for the old tokens, the item / global
        gates for the extra ones); None: unscaled (as before)."""
        b, t, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if vscale is not None:
            v = v * vscale.to(v.dtype).view(1, 1, t, 1)
        a = attention(q, k, v, bias, self.math)
        if ext is not None:   # 2026-10-12 (B21 / B22): the gated terms of the extra tokens (module docstring)
            a = a + self._ext(q, k, v, bias, ext)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.ff(self.n2(x))

    def _ext(self, q, k, v, bias, ext):
        """(B21 / B22) The extra tokens of this layer: returns the gated terms they add to the old tokens' attention
        output ([b, heads, t, dh], q / k / v / bias the old tokens' in the padded [b, t] layout) and updates ext['y'] (the
        extra tokens [b, G, d]: attention over every real old and extra key, the projection and the feed-forward of
        this block). ext: y, ybias [b, 1, 1, G] (0 / -1e9), cross [(lo, hi, gates [layers], has [b, 1, 1, 1] bool or
        None)], l (this layer's index)."""
        y = ext['y']
        b, G, d = y.shape
        t = q.shape[2]
        qy, ky, vy = self.qkv(self.n1(y)).view(b, G, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        add = None
        for lo, hi, gates, has in ext['cross']:
            if hi - lo == 1:   # one key: its softmax weight is exactly 1, the term is its value
                c = vy[:, :, lo:hi].expand(-1, -1, t, -1)
            else:   # (a record without any real key: a finite mean of the masked values, removed by has = 0)
                c = F.scaled_dot_product_attention(q, ky[:, :, lo:hi], vy[:, :, lo:hi],
                                                   attn_mask=ext['ybias'][..., lo:hi].to(q.dtype))
            g = gates[ext['l']]
            c = c * (g.to(c.dtype) if has is None else g.to(c.dtype) * has.to(c.dtype))
            add = c if add is None else add + c
        mask = torch.cat([bias.expand(b, 1, 1, t), ext['ybias']], -1)
        ay = F.scaled_dot_product_attention(qy, torch.cat([k, ky], 2), torch.cat([v, vy], 2), attn_mask=mask.to(qy.dtype))
        y = y + self.proj(ay.transpose(1, 2).reshape(b, G, d))
        ext['y'] = y + self.ff(self.n2(y))
        return add

    def _ext_packed(self, qkv, ext):
        """(B21 / B22) _ext for the packed layers: the real old tokens' q / k / v (qkv [m, 3d], at the flat positions
        ext['keep'] of the [n, t] layout) scattered into that layout; the gated terms back at the real tokens [m, d]."""
        m, d3 = qkv.shape
        d = d3 // 3
        n, t, keep = ext['n'], ext['t'], ext['keep']
        full = qkv.new_zeros((n * t, d3)).index_copy(0, keep, qkv)
        q, k, v = full.view(n, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        add = self._ext(q, k, v, ext['pbias'], ext)
        return add.transpose(1, 2).reshape(n * t, d).index_select(0, keep)

    def forward_packed(self, x, bias, keep, shape, ext=None, vscale=None):
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
        if vscale is not None:   # (B24 fused: [m] multipliers of the real tokens' value vectors)
            qkv = torch.cat([qkv[:, :2 * d], qkv[:, 2 * d:] * vscale.to(qkv.dtype)[:, None]], 1)
        if isinstance(shape, dict):   # 2026-10-10 (B18): attention per length bucket (learner_plan)
            gpos = shape['gpos']
            full = qkv.new_zeros((shape['total'], 3 * d)).index_copy(0, gpos, qkv)
            if self.math and full.dtype == torch.float32 and not _autocast_on():   # (one node for all buckets)
                a = _BucketAttention.apply(full, self.heads, shape['buckets']).index_select(0, gpos)
                if ext is not None:   # (B21 / B22)
                    a = a + self._ext_packed(qkv, ext)
                x = x + self.proj(a)
                return x + self.ff(layer_norm(self.n2, x, self.ln_fast))
            outs, at = [], 0
            for nb, t, bias_k in shape['buckets']:
                q, k, v = full[at:at + nb * t].view(nb, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
                a = attention(q, k, v, bias_k, self.math)
                outs.append(a.transpose(1, 2).reshape(nb * t, d))
                at += nb * t
            a = (torch.cat(outs) if len(outs) > 1 else outs[0]).index_select(0, gpos)
            if ext is not None:   # (B21 / B22)
                a = a + self._ext_packed(qkv, ext)
            x = x + self.proj(a)
            return x + self.ff(layer_norm(self.n2, x, self.ln_fast))
        b, t = shape
        full = qkv.new_zeros((b * t, 3 * d)).index_copy(0, keep, qkv)
        q, k, v = full.view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = attention(q, k, v, bias, self.math)
        if ext is not None:   # (B21 / B22)
            a = a + self._ext(q, k, v, bias, ext)
        x = x + self.proj(a.transpose(1, 2).reshape(b * t, d).index_select(0, keep))
        return x + self.ff(layer_norm(self.n2, x, self.ln_fast))


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
    module docstring) and their key mask, plus the per-layer gates of the old tokens' term (zero at the start).
    B23 (2026-10-13; TokPolicy(item_shared=True)): coll = the policy's item_emb (ITEM_EMB = 16 columns, the table of the
    pedestals' ids and the multi-hot), passed in and not registered here (one parameter, one state-dict key
    'item_emb.weight'); the mlp's first Linear then reads ITEM_EMB collectible columns instead of IT_COLL."""

    def __init__(self, width, layers, coll=None, slots=INV_CAP, fused=False):
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
        # (B24 fused: the per-layer multipliers of the item tokens' value vectors, 1 = ordinary tokens at the start)
        self.gate = nn.Parameter(torch.ones(layers) if fused else torch.zeros(layers))

    @property
    def coll(self):
        """The collectible table: its own (B21) or the policy's item_emb (B23 item_shared)."""
        own = self._modules.get('coll')
        return own if own is not None else self.__dict__['_shared_coll']

    def forward(self, b, compact=False):
        """(tokens [n, n_tok, width], present [n, n_tok] bool). No host synchronisation, static shapes. n_tok =
        N_ITEM_TOK unless slots (B24 --item-slots) keeps fewer held collectibles (the newest ones: ROW 'inv' is newest
        first). compact (B24 fused): the present tokens moved to the front of each record, in their order (the absent
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
    """B22 (2026-10-12; TokPolicy(global_token=True)): a learned token, the per-layer gates of its term for the old
    tokens (zero at the start), the second value head value_global on it (own norm) and, with memory, mem_in (zero at
    the start: the GRU's input is z + mem_in(norm(global token)))."""

    def __init__(self, width, layers, memory=0, fused=False):
        super().__init__()
        self.tok = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.tok, std=0.02)
        self.gate = nn.Parameter(torch.ones(layers) if fused else torch.zeros(layers))   # (B24 fused: as ItemTokens)
        self.norm = nn.LayerNorm(width)
        self.value = mlp(width, width, 1)
        self.mem_in = nn.Linear(width, width) if memory else None
        if memory:
            nn.init.zeros_(self.mem_in.weight)
            nn.init.zeros_(self.mem_in.bias)


class TokPolicy(nn.Module):
    def __init__(self, width=192, layers=4, heads=4, items=False, charge=False, choice=0, ent_ext=False, pchar=False,
                 memory=0, item_tokens=False, global_token=False, item_shared=False, item_tokens_fused=False,
                 item_slots=INV_CAP):
        super().__init__()
        if item_tokens_fused and not (item_tokens or global_token):
            raise ValueError('TokPolicy: item_tokens_fused needs item_tokens or global_token (B24)')
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
        self.entity = mlp(ENT_F0 + 40 + ((ITEM_EMB + TRINKET_EMB) if items else 0) + ((ENT_F - ENT_F0) if ent_ext else 0),
                          width, width)
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
        # 2026-10-12 (B21 / B22): registered after everything else (memory included), gates zero at the start (the
        # module docstring); the config keys only when set
        self.item_tokens, self.global_token = bool(item_tokens), bool(global_token)
        if self.item_tokens:
            self.config['item_tokens'] = True
        if self.global_token:
            self.config['global_token'] = True
        # 2026-10-13 (B23): the item tokens' collectible table is item_emb itself (the key only when set)
        self.item_shared = bool(item_shared)
        if self.item_shared:
            self.config['item_shared'] = True
        # 2026-10-14 (B24): the item / global tokens as ordinary tokens of the one attention (_forward_fused), and the
        # number of held-collectible token slots (the keys only when set; the same parameters, names and shapes)
        self.fused = bool(item_tokens_fused)
        if self.fused:
            self.config['item_tokens_fused'] = True
        if self.item_tokens and int(item_slots) != INV_CAP:
            self.config['item_slots'] = int(item_slots)
        self.itok = ItemTokens(width, layers, coll=self.item_emb if self.item_shared else None, slots=int(item_slots),
                               fused=self.fused) if self.item_tokens else None
        self.gtok = GlobalToken(width, layers, self.memory, fused=self.fused) if self.global_token else None

    def gate_values(self):
        """(B21 / B22) {'item_gate': mean |gate| over the layers, 'global_gate': ...} of the options that are on."""
        out = {}
        with torch.no_grad():
            if self.itok is not None:
                out['item_gate'] = float(self.itok.gate.abs().mean())
            if self.gtok is not None:
                out['global_gate'] = float(self.gtok.gate.abs().mean())
        return out

    def _ext_tokens(self, b):
        """(B21 / B22) The extra tokens of a batch and their attention bookkeeping (Block._ext's ext dict); None when
        both options are off."""
        if self.itok is None and self.gtok is None:
            return None
        toks, masks, cross = [], [], []
        n = b['player'].shape[0]
        dev = b['player'].device
        if self.itok is not None:
            tok, present = self.itok(b)
            toks.append(tok)
            masks.append(present)
            cross.append((0, self.itok.n_tok, self.itok.gate, present.any(1)[:, None, None, None]))
        glob = None
        if self.gtok is not None:
            glob = sum(t_.shape[1] for t_ in toks)
            toks.append(self.gtok.tok.view(1, 1, -1).expand(n, 1, -1))
            masks.append(torch.ones((n, 1), dtype=torch.bool, device=dev))
            cross.append((glob, glob + 1, self.gtok.gate, None))
        y = torch.cat(toks, 1) if len(toks) > 1 else toks[0]
        present = torch.cat(masks, 1) if len(masks) > 1 else masks[0]
        ybias = torch.zeros((n, 1, 1, y.shape[1]), device=dev, dtype=y.dtype).masked_fill(
            ~present[:, None, None], -1e9)
        return dict(y=y, ybias=ybias, cross=cross, l=0, glob=glob)

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

    def forward(self, b, entities=None, packed=False, need_h=False, mem=None, need_vg=False):
        """b: the dict of to_batch. Returns (logits [B, 16], value [B], danger logits [B, 9]).
        packed: the Transformer's per-token layers on the real tokens only (Block.forward_packed; the padded door and
        entity tokens are skipped). Same values up to summation order, less work when many tokens are padding (the
        learner); not for CUDA-graph capture (the number of real tokens varies).
        mem (2026-10-11, memory models only): the GRU's state (module docstring); then the new state h_new [B, D] is
        returned as the last element.
        need_vg (2026-10-12, B22 global-token models only): value_global [B] appended as the very last element (after
        h_new). The extra tokens of B21 / B22 (module docstring) need nothing from the caller."""
        if need_vg and self.gtok is None:
            raise ValueError('TokPolicy.forward: need_vg needs a global-token model (global_token=True)')
        if self.fused:   # 2026-10-14 (B24): one attention over the old and the extra tokens
            return self._forward_fused(b, entities, packed, need_h, mem, need_vg)
        ext = self._ext_tokens(b)   # (B21 / B22; None when off: the code below is the one before)
        n, used, fast, player, room, level, entity = self._embed_old(b, entities, packed)
        if fast:   # (B18) the door mlp on the real doors only; the real tokens grouped by kind, then in packed order
            door = embed(self.door, b['doors'].reshape(n * DOOR_CAP, -1).index_select(0, b['dkeep'])) + self.kind[2]
            xp = torch.cat([player, room, level, door, entity]).index_select(0, b['xperm'])
            if ext is not None:   # (B21 / B22: the padded layout of the extra attention)
                t_ = 3 + DOOR_CAP + used
                real_ = torch.zeros(n * t_, dtype=torch.bool, device=xp.device).index_fill_(0, b['keep'], True)
                ext.update(n=n, t=t_, keep=b['keep'], pbias=torch.zeros(
                    (n, 1, 1, t_), device=xp.device, dtype=xp.dtype).masked_fill(~real_.view(n, 1, 1, t_), -1e9))
            x0 = self._blocks_bucketed(xp, b, ext).index_select(0, b['starts'])
            return self._heads(x0, need_h, mem, self._yg(ext), need_vg)
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
            if ext is not None:   # (B21 / B22: the padded layout of the extra attention)
                ext.update(n=n, t=t, keep=keep, pbias=bias)
            if 'buckets' in b:   # 2026-10-10 (B18): attention per length bucket (learner_plan, from the host)
                xp = self._blocks_bucketed(xp, b, ext)
            else:
                for l_, block in enumerate(self.blocks):
                    if ext is not None:
                        ext['l'] = l_
                    xp = block.forward_packed(xp, bias, keep, (n, t), ext)
            x0 = xp.index_select(0, starts)   # the player's token: the first of each record
        else:
            for l_, block in enumerate(self.blocks):
                if ext is not None:
                    ext['l'] = l_
                x = block(x, bias, ext)
            x0 = x[:, 0]
        return self._heads(x0, need_h, mem, self._yg(ext), need_vg)

    def _embed_old(self, b, entities, packed):
        """forward()'s token embeddings of the old tokens (player, room, map, entities; the doors are the
        caller's): (n, entity tokens used, fast, player [n, d], room, level, entity: [n, used, d], or with the B18
        fast path the real entities [m_e, d])."""
        n = b['player'].shape[0]
        used = entities or b['ent'].shape[1]
        ids = b['ent_id'][:, :used]
        it = b['ent_item'][:, :used] if self.items else None
        ent = b['ent'][:, :used]
        ekeep = b.get('ekeep') if packed else None
        if ekeep is not None:
            # 2026-10-10 (B18, the learner's fast path): the entity tokens' embeddings and input mlp on the real
            # entities only (ekeep: their flat positions in the [n, used] layout, from the host); the padded ones are
            # zero here and dropped by the packed layers anyway. The real tokens' values are the same
            flat = n * used
            ids = ids.reshape(flat, -1).index_select(0, ekeep)
            it = it.reshape(flat, -1).index_select(0, ekeep) if it is not None else None
            ent = ent.reshape(flat, -1).index_select(0, ekeep)
        ids = ids.long()
        emb = [self.type_emb(ids[..., 0]), self.variant_emb(ids[..., 1]), self.subtype_emb(ids[..., 2])]
        if self.items:
            it = it.long()
            emb += [self.item_emb(it[..., 0]), self.trinket_emb(it[..., 1])]
        parts = [ent[..., :ENT_F0]] + emb
        if self.ent_ext:   # appended last: load_compatible pads these columns with zeros
            parts.append(ent[..., ENT_F0:])
        entity = embed(self.entity, torch.cat(parts, -1)) + self.kind[3]
        fast = ekeep is not None and 'xperm' in b   # B18: the packed tokens built directly (learner_plan's xperm)
        if ekeep is not None and not fast:
            entity = entity.new_zeros((n * used, entity.shape[-1])).index_copy(0, ekeep, entity).view(n, used, -1)
        pending = b['pending']
        under_way = torch.cat([F.one_hot(pending[:, i], k) for i, k in enumerate(HEADS)], -1).float()
        patch = self._cnn(self.patch, b, 'patch').float()
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
        if self.pchar:   # 2026-10-08: the character's vector added before the first GELU (exact for old checkpoints)
            char = self.char_emb(b['pchar'].long().clamp(0, N_CHAR - 1))
            seq = self.player

            def player_mlp(x):
                return seq[2](seq[1](seq[0](x) + char.to(x.dtype)))
            player = embed(player_mlp, torch.cat(parts, -1)) + self.kind[0]
        else:
            player = embed(self.player, torch.cat(parts, -1)) + self.kind[0]
        room = self._cnn(self.room, b, 'grid') + self.kind[1]
        level = self._cnn(self.map, b, 'map') + self.kind[4]
        return n, used, fast, player, room, level, entity

    def extra_tokens(self):
        """(B24) (item tokens per record, global tokens per record) of the fused layout ((0, 0) when not fused)."""
        if not self.fused:
            return 0, 0
        return (self.itok.n_tok if self.itok is not None else 0), (1 if self.gtok is not None else 0)

    def _vtable(self):
        """(B24 fused) [layers, 3] value multipliers per token group: old tokens (1), item tokens (itok.gate), the
        global token (gtok.gate)."""
        one = self.kind.new_ones(len(self.blocks))
        return torch.stack([one, self.itok.gate if self.itok is not None else one,
                            self.gtok.gate if self.gtok is not None else one], 1)

    def _forward_fused(self, b, entities, packed, need_h, mem, need_vg):
        """forward() of a B24 model (item_tokens_fused): the item tokens (ItemTokens, compacted: the present ones first)
        and the global token are appended to the old tokens, [player, room, map, doors, entities, items, global], and
        every block runs ONE attention over all of them (one softmax, the same q / k / v projections; absent items
        are masked keys like absent entities); the gates multiply the value vectors of the item / global tokens
        (Block vscale). The heads read the player token, value_global (and with memory mem_in) the global token.
        Packed layouts (the learner): the real item tokens are the first item_count() of a record, the global token is
        each record's last real token; host plans (packed_index / learner_plan with n_item) include them."""
        NI, G = self.extra_tokens()
        n, used, fast, player, room, level, entity = self._embed_old(b, entities, packed)
        vt = self._vtable()
        if NI:
            itok, ipres = self.itok(b, compact=True)
        if fast:   # (B18 fast path: learner_plan / plan_tokens with n_item; the real tokens grouped by kind)
            door = embed(self.door, b['doors'].reshape(n * DOOR_CAP, -1).index_select(0, b['dkeep'])) + self.kind[2]
            src = [player, room, level, door, entity]
            if NI:
                src.append(itok.reshape(n * NI, -1).index_select(0, b['ikeep']))
            if G:
                src.append(self.gtok.tok.view(1, -1).expand(n, -1))
            xp = self._blocks_bucketed(torch.cat(src).index_select(0, b['xperm']), b, vgroup=(b['tgroup'], vt))
            return self._heads(xp.index_select(0, b['starts']), need_h, mem,
                               xp.index_select(0, b['glast']) if G else None, need_vg)
        door = embed(self.door, b['doors']) + self.kind[2]
        dev = player.device
        toks = [player[:, None], room[:, None], level[:, None], door, entity]
        pads = [torch.zeros((n, 3), dtype=torch.bool, device=dev),
                torch.arange(DOOR_CAP, device=dev)[None] >= b['n_doors'][:, None],
                torch.arange(used, device=dev)[None] >= b['n_ent'][:, None]]
        if NI:
            toks.append(itok)
            pads.append(~ipres)
        if G:
            toks.append(self.gtok.tok.view(1, 1, -1).expand(n, 1, -1))
            pads.append(torch.zeros((n, 1), dtype=torch.bool, device=dev))
        x = torch.cat(toks, 1)
        pad = torch.cat(pads, 1)
        t = x.shape[1]
        group = torch.cat([torch.zeros(t - NI - G, dtype=torch.int64, device=dev),
                           torch.ones(NI, dtype=torch.int64, device=dev),
                           torch.full((G,), 2, dtype=torch.int64, device=dev)])
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
            tg = group.index_select(0, keep.remainder(t))
            if 'buckets' in b:
                xp = self._blocks_bucketed(xp, b, vgroup=(tg, vt))
            else:
                for l_, block in enumerate(self.blocks):
                    xp = block.forward_packed(xp, bias, keep, (n, t), vscale=vt[l_].index_select(0, tg))
            return self._heads(xp.index_select(0, starts), need_h, mem, xp.index_select(0, last) if G else None,
                               need_vg)
        for l_, block in enumerate(self.blocks):
            x = block(x, bias, vscale=vt[l_].index_select(0, group))
        return self._heads(x[:, 0], need_h, mem, x[:, -1] if G else None, need_vg)

    @staticmethod
    def _yg(ext):
        """(B22) the global token after the blocks [B, width], or None."""
        return ext['y'][:, ext['glob']] if ext is not None and ext['glob'] is not None else None

    def _cnn(self, seq, b, key):
        """A grid input's CNN; with b[key + '_u'] / b[key + '_i'] (B18, the learner's fast path: the distinct grids of
        the batch and each record's one, from the host) on the distinct grids only, gathered back per record (the same
        values; the gradient sums the records' terms per distinct grid)."""
        u = b.get(key + '_u')
        if u is None:
            return seq(self._conv_in(b[key]))
        return seq(self._conv_in(b[key].index_select(0, u))).index_select(0, b[key + '_i'])

    def _blocks_bucketed(self, xp, b, ext=None, vgroup=None):
        """(B18) the blocks on the packed tokens xp with attention per length bucket (learner_plan's gpos, buckets).
        ext (B21 / B22): the extra tokens (forward's, with n, t, keep, pbias of the padded layout). vgroup (B24 fused):
        (each packed token's group [m] (0 old, 1 item, 2 global), _vtable()): the value multipliers per layer."""
        dev = xp.device
        ar = torch.arange(max(w_ for _, w_ in b['buckets']), device=dev)
        shape = dict(gpos=b['gpos'], total=sum(len(c_) * w_ for c_, w_ in b['buckets']), buckets=[
            (len(c_), w_, torch.zeros((len(c_), 1, 1, w_), device=dev, dtype=xp.dtype).masked_fill(
                (ar[None, :w_] >= c_[:, None])[:, None, None], -1e9)) for c_, w_ in b['buckets']])
        for l_, block in enumerate(self.blocks):
            if ext is not None:
                ext['l'] = l_
            if vgroup is not None:
                xp = block.forward_packed(xp, None, None, shape, vscale=vgroup[1][l_].index_select(0, vgroup[0]))
            else:
                xp = block.forward_packed(xp, None, None, shape, ext)
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

    def _heads(self, x0, need_h, mem=None, yg=None, need_vg=False):
        """The final norm and the heads on the player's tokens x0. yg, need_vg (B22): the global token after the blocks
        and whether value_global is appended to the outputs."""
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

    def act(self, b, greedy=False, entities=None, u=None, mem=None, need_vg=False):
        """Actions [B, heads], their log-probability [B] and the value [B]. u (2026-10-07, Phase B2): sampling
        uniforms [B, sum of head sizes] for Gumbel-max (GraphActor's rule; the common random numbers of branch
        records); None: torch.multinomial as before. mem (2026-10-11, memory models): the GRU's state [B, D]; the
        new state is then returned as a fourth element. need_vg (2026-10-14, B24; global-token models): value_global
        [B] (float32) appended last (after the new state)."""
        kw = dict(need_vg=True) if need_vg else {}
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

    def evaluate(self, b, actions, entities=None, packed=False, mem=None):
        """Log-probability of the actions [B], entropy per head [B, heads], value [B], danger logits [B, 9]. mem
        (memory models): the GRU's state (forward's; the new state is not returned)."""
        if mem is not None:
            logits, value, danger = self.forward(b, entities, packed, mem=mem)[:3]
        else:
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
             pcharge=t('pcharge'), pchar=t('pchar'))
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


def packed_index(n_ent, n_doors, entities, n_item=None, item_tok=0, glob=0):
    """forward(packed=True)'s token positions from the host's entity and door counts (numpy, one per record) for a
    batch with `entities` entity tokens: (keep, starts) int64, the flat positions of the real tokens in the [n, t]
    layout (t = 3 + DOOR_CAP + entities) and each record's first real token among them. Put into the batch dict as
    'keep' and 'starts' (tensors on its device) they spare forward() the wait for the GPU that finding them costs.
    B24 (a fused model: TokPolicy.extra_tokens() = (item_tok, glob)): n_item the records' item_count(), t grows by
    item_tok + glob (the real item tokens first, the global token always real)."""
    n = len(n_ent)
    real = np.ones((n, 3 + DOOR_CAP + entities + item_tok + glob), bool)
    real[:, 3:3 + DOOR_CAP] = np.arange(DOOR_CAP)[None] < np.asarray(n_doors)[:, None]
    real[:, 3 + DOOR_CAP:3 + DOOR_CAP + entities] = np.arange(entities)[None] < np.asarray(n_ent)[:, None]
    if item_tok:
        real[:, 3 + DOOR_CAP + entities:3 + DOOR_CAP + entities + item_tok] = \
            np.arange(item_tok)[None] < np.asarray(n_item)[:, None]
    counts = real.sum(1)
    return np.flatnonzero(real), np.cumsum(counts) - counts


BUCKET_COST = 2.0e5   # a bucket's fixed cost in record x token^2 units (its ~12 more kernels per layer; H20, B18)


def learner_plan(n_ent, n_doors, entities, buckets=3, n_item=None, item_tok=0, glob=0):
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
    n_item + glob), rec gets two more columns (items, first real item) and m_item, item_tok, glob are set."""
    ne = np.minimum(np.asarray(n_ent, np.int64), entities)
    nd = np.minimum(np.asarray(n_doors, np.int64), DOOR_CAP)
    cnt = 3 + nd + ne
    n = len(cnt)
    fused = bool(item_tok or glob)
    if fused:
        ni = np.minimum(np.asarray(n_item, np.int64), item_tok) if item_tok else np.zeros(n, np.int64)
        cnt = cnt + ni + glob
    rec = np.empty((n, 8 if fused else 6), np.int64)
    rec[:, 0], rec[:, 2], rec[:, 3] = cnt, nd, ne
    rec[:, 1] = np.cumsum(cnt) - cnt
    rec[:, 4] = np.cumsum(nd) - nd
    rec[:, 5] = np.cumsum(ne) - ne
    if fused:
        rec[:, 6], rec[:, 7] = ni, np.cumsum(ni) - ni
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
               t=3 + DOOR_CAP + entities + item_tok + glob, entities=entities, buckets=[])
    if fused:
        out.update(m_item=int(ni.sum()), item_tok=int(item_tok), glob=int(glob))
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
    tgroup (each packed token's group: 0 old, 1 item, 2 global), glast (each record's last packed token: the global
    token)."""
    if rec.shape[1] == 8:
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
    m_i, NI, G = plan['m_item'], plan['item_tok'], plan['glob']
    cnt, starts, nd, ne, dstart, estart, ni, istart = rec.unbind(1)
    ar = torch.arange(n, device=dev)
    r = torch.repeat_interleave(ar, cnt, output_size=m)
    within = torch.arange(m, device=dev) - starts[r]
    nd_r, ne_r, ni_r = nd[r], ne[r], ni[r]
    b1 = 3 + nd_r          # first entity
    b2 = b1 + ne_r         # first item
    b3 = b2 + ni_r         # the global token
    zero = torch.zeros_like(within)
    slot = within + torch.where(within >= b1, DOOR_CAP - nd_r, zero) + torch.where(within >= b2, e - ne_r, zero) + \
        torch.where(within >= b3, NI - ni_r, zero)
    ekeep = torch.repeat_interleave(ar * e - estart, ne, output_size=m_e) + torch.arange(m_e, device=dev)
    dkeep = torch.repeat_interleave(ar * DOOR_CAP - dstart, nd, output_size=m_d) + torch.arange(m_d, device=dev)
    ikeep = torch.repeat_interleave(ar * NI - istart, ni, output_size=m_i) + torch.arange(m_i, device=dev)
    base_e, base_i, base_g = 3 * n + m_d, 3 * n + m_d + m_e, 3 * n + m_d + m_e + m_i
    xperm = torch.where(within < 3, within * n + r, torch.where(
        within < b1, 3 * n + dstart[r] + within - 3, torch.where(
            within < b2, base_e + estart[r] + within - b1, torch.where(
                within < b3, base_i + istart[r] + within - b2, base_g + r))))
    tgroup = (within >= b2).long() + (within >= b3).long()
    return dict(keep=r * t + slot, starts=starts, ekeep=ekeep, dkeep=dkeep, xperm=xperm, gpos=base[r] + within,
                ikeep=ikeep, tgroup=tgroup, glast=starts + cnt - 1)


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
           'pitem', 'pinv', 'pcharge', 'pchar')
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
    """

    def __init__(self, model, rows_max, entities=32, stream=None, crn=True, memory=None):
        from .tok_obs import ENT_CAP
        dev = next(model.parameters()).device
        self.model, self.n, self.stream, self._n = model, rows_max, stream, 0
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
        self.out_host = torch.zeros((n, nh + 2 + self.n_extra + self.mem + int(self.vg_on)),
                                    dtype=torch.float32).pin_memory()
        self.out_np = self.out_host.numpy()
        self.done = torch.cuda.Event()
        self.sizes = sorted({min(int(entities), ENT_CAP), ENT_CAP})
        self.entities = self.sizes[0]

        def run(tokens):
            b = decode_rows(self.raw, entities=tokens, n_heads=nh)
            kw = dict(need_vg=True) if self.vg_on else {}   # (B24: value_global appended last)
            if self.mem:   # (memory: the state's input from self.h_in, the new state kept for the write-back)
                outs = model(b, tokens, need_h=self.choice, mem=self.h_in, **kw)
                logits, value, h, self._h_last = (outs[0], outs[1], outs[3] if self.choice else None,
                                                  outs[-2] if self.vg_on else outs[-1])
            elif self.choice:
                outs = model(b, tokens, need_h=True, **kw)
                logits, value, h = outs[0], outs[1], outs[3]
            else:
                outs = model(b, tokens, **kw)
                logits, value = outs[0], outs[1]
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
            if self.mem:   # (memory: each record's input state, for the rollout)
                cols.append(self.h_in)
            if self.vg_on:   # (B24) value_global
                cols.append(outs[-1].float()[:, None])
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
                if self.mem:
                    self.h_outs.append(self._h_last)
        torch.cuda.synchronize()

    def submit(self, rows, pending, widx=None, reset=None):
        """Start a call (returns at once; False if it does not fit). rows: a C-contiguous ROW array, pending [n, heads]
        int64. The next submit may only follow this call's result(). Memory models: widx [n] the records' rows of the
        state table (distinct), reset [n] bool (None: none) where an episode starts (its state is zero)."""
        n = len(rows)
        if n > self.n:
            return False
        if self.mem:
            self.widx_np[:n] = widx
            self.keep_np[:n] = 1.0 if reset is None else (~np.asarray(reset, bool)).astype(np.float32)
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
            if self.mem:   # the records' states into the graph's input (zero at an episode start)
                self.widx[:n].copy_(self.widx_stage[:n], non_blocking=True)
                self.keep[:n].copy_(self.keep_stage[:n], non_blocking=True)
                self.h_in[:n] = self.table.index_select(0, self.widx[:n]) * self.keep[:n, None]
            self.graphs[g].replay()
            if self.mem:   # the new states back into the table
                self.table.index_copy_(0, self.widx[:n], self.h_outs[g][:n])
            self.out_host[:n].copy_(self.outs[g][:n], non_blocking=True)
            self.done.record()
        self._n = n
        return True

    def result(self):
        """(actions [n, heads] int64, log-probabilities [n], values [n]) of the submitted call, as numpy arrays."""
        self.done.synchronize()
        out = self.out_np[:self._n].copy()
        nh = self.nh
        self.extra = out[:, nh + 2:nh + 2 + self.n_extra] if self.n_extra else None
        if self.mem:   # (memory) each record's input state
            self.h_used = out[:, nh + 2 + self.n_extra:nh + 2 + self.n_extra + self.mem]
        self.vg = out[:, -1] if self.vg_on else None   # (B24) value_global
        return out[:, :nh].astype(np.int64), out[:, nh], out[:, nh + 1]

    def act(self, rows, pending, widx=None, reset=None):
        """submit + result: (actions, log-probabilities, values), or None if it does not fit."""
        return self.result() if self.submit(rows, pending, widx, reset) else None


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
    nn.Linear default, U(-1/sqrt(60), 1/sqrt(60)), drawn when the model was built). Every other tensor (bias, gates,
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


def convert_to_fused(model, state):
    """B24 (2026-10-14): a B21 / B22 / B23 checkpoint's state dict (separate gated terms) for a fused model
    (TokPolicy item_tokens_fused, any item_slots). Deterministic and data-free: every tensor is kept as saved, the
    gates (itok.gate, gtok.gate: per-layer scalars of the old tokens' extra terms) become the multipliers of the item /
    global tokens' value vectors (tests/test_fused_tokens_smoke.py (c): closer to the checkpoint's outputs than gates
    reset to 1 on synthetic records with C92-sized gates). Approximate, not exact (module docstring). Returns (state,
    a log line); raises ValueError if the model is not fused or a tensor does not fit."""
    if not getattr(model, 'fused', False):
        raise ValueError('convert_to_fused: the model is not fused (item_tokens_fused)')
    own = model.state_dict()
    bad = [k for k, v in state.items() if k not in own or own[k].shape != v.shape]
    missing = [k for k in own if k not in state and not k.startswith('choice.')]
    if bad or missing:
        raise ValueError(f'convert_to_fused: tensors that do not fit {bad[:4]}, missing {missing[:4]}')
    gates = {k: [round(float(v_), 4) for v_ in state[k]] for k in ('itok.gate', 'gtok.gate') if k in state}
    return dict(state), ('item_tokens_fused (B24): every tensor kept as saved; the gates ' + str(gates) + ' are now the '
                         'multipliers of the extra tokens\' value vectors; approximate (one softmax over the old and the '
                         'extra tokens instead of the gated terms)')


def load_compatible(model, state):
    """Load a checkpoint's weights into a model whose input layers may have grown (more door features, ...): a
    tensor of another shape is copied into the leading part of the model's (the rest keeps its fresh values, zero
    for added input columns). Returns the names that did not match exactly. (B23: a B21 checkpoint's own item-token
    table into an item_shared model goes through share_item_table first, reported as 'itok.coll.weight (shared: ...)'.)"""
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
