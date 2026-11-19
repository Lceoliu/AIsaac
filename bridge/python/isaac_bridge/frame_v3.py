"""The v3 frame encoder (package A, seq-v1 step 2): the tokens of one decoded visible-v3 frame (isaac_bridge.v3_input)
through a spatial transformer of the old model's size; TokPolicy(frame='v3') reads the controlled body's token and the
global token from it, then the temporal side and the heads exactly as C114 T1 (docs/SINGLE_TRANSFORMER_DESIGN.md 3.2).

Tokens per frame (v3_input.pad_batch's padded layout; a padded token is a masked key):
  P   one per body (P_MAX rows): the numeric / flag columns, the categorical rows' embeddings, P_off, P_rel,
      P_rel_valid; the controlled body's row (0) also gets the action under way (pending, one-hot over the first arm's
      five heads, 20 inputs; other rows zeros)
  L   one per body: its 9 x 9 local patch (the 21 decoded canvas channels, a small CNN); masked when local_pad (C13)
  E   one per entity: numeric / flag columns, identity (type, variant, subtype), categorical rows, anim row, price
      number, E_rel, E_laser_rel
  N   one per inventory row: numeric / flag columns, the item row
  S   one per slot row: numeric / flag columns, categorical rows, the item row
  R   28 room tokens: the canvas (21 channels) through a CNN down to 4 x 7 blocks of 4 x 4 cells, each block with its
      centre (room_xy) and the centre's offset / distance to the controlled body (room_rel)
  M   one map token: the minimap's 4 bits + its icon embedding through a CNN
  G   the learned global token (TokPolicy.gtok.tok)
Each token type adds its own kind vector. Order [P, L, E, N, S, R, M, G]: the controlled body is token 0, the global
token the last one. The embedding tables follow schema_v3's namespaces: identity (65,536 x 24, one table, the three id
columns side by side), item (8,192 x 24), cat (1,024 x 16), anim (4,096 x 16), map icon (64 x 8); row 0 (none) is an
ordinary row here (the decode's 0 means none / padding and the token's other inputs say so)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import schema_v3 as V3
from . import v3_input as VI

ID_EMB, ITEM_EMB, CAT_EMB, ANIM_EMB, ICON_EMB = 24, 24, 16, 16, 8
# (input revision 2, the coordinator 2026-11-19: v3_input.WIDEN's columns mapped back inside the model, in float32:
#  E age (game minutes) -> log2(1 + game seconds) / 2; S charge / max_charge (120ths) -> u = 12ths, u up to 1, 1 + log2(u) past it)
LOG_INPUTS = {('E', 'age'): 'age', ('S', 'charge'): 'charge', ('S', 'max_charge'): 'charge'}
L_INIT_GAIN = 5.0   # (the local patch token's initial scale: 0.027 -> ~0.14, as the other kinds; the range table's probe)
G_INIT_STD = 0.15   # (the global token's parameter, TokPolicy.gtok.tok, for a v3 model: 0.02 -> 0.15, likewise)
PENDING_HEADS = (9, 5, 2, 2, 2)   # the first arm's five heads (move, shoot, bomb, item, pill): drop is not an input
KINDS = ('P', 'L', 'E', 'N', 'S', 'R', 'M', 'G')
RC = V3.RC


def _width(plan):
    """(numeric, flag) input widths of a token type's decoded columns."""
    flag = len(plan['known']) + len(plan['bit']) + VI.W8 * len(plan['bits8']) + VI.W16 * len(plan['word16'])
    return len(plan['num']), flag


def _mlp(n_in, width):
    return nn.Sequential(nn.Linear(n_in, width), nn.GELU(), nn.Linear(width, width))


class FrameEncoderV3(nn.Module):
    def __init__(self, width=192, layers=4, heads=4):
        super().__init__()
        from .tok_policy import Block
        P, E, N, S = (VI.PLAN[t] for t in 'PENS')
        self.id_emb = nn.Embedding(V3.IDENTITY_ROWS, ID_EMB)
        self.item_emb = nn.Embedding(V3.ITEM_ROWS, ITEM_EMB)
        self.cat_emb = nn.Embedding(V3.CAT_ROWS, CAT_EMB)
        self.anim_emb = nn.Embedding(V3.ANIM_ROWS, ANIM_EMB)
        self.icon_emb = nn.Embedding(V3.MAP_ICONS, ICON_EMB)
        pn, pf = _width(P)
        self.n_pending = sum(PENDING_HEADS)
        self.p_in = pn + pf + len(P['cat']) * CAT_EMB + 2 + 3 + 1 + self.n_pending
        en, ef = _width(E)
        self.e_in = en + ef + len(E['idn']) * ID_EMB + len(E['cat']) * CAT_EMB + len(E['anim']) * ANIM_EMB + 1 + 3 + 3
        nn_, nf = _width(N)
        self.n_in = nn_ + nf + len(N['item']) * ITEM_EMB
        sn, sf = _width(S)
        self.s_in = sn + sf + len(S['cat']) * CAT_EMB + len(S['item']) * ITEM_EMB
        self.p_mlp, self.e_mlp = _mlp(self.p_in, width), _mlp(self.e_in, width)
        self.n_mlp, self.s_mlp = _mlp(self.n_in, width), _mlp(self.s_in, width)
        side = V3.LOCAL_SIDE
        self.l_cnn = nn.Sequential(nn.Conv2d(RC, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1),
                                   nn.GELU(), nn.Flatten(), nn.Linear(32 * side * side, width))
        self.r_cnn = nn.Sequential(nn.Conv2d(RC, 32, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(32, 64, VI.R_BLOCK, stride=VI.R_BLOCK), nn.GELU())
        self.r_out = nn.Linear(64 + 2 + 3, width)
        self.m_cnn = nn.Sequential(nn.Conv2d(4 + ICON_EMB, 32, 3, padding=1), nn.GELU(),
                                   nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(), nn.Flatten(),
                                   nn.Linear(64 * ((V3.MAP_SIDE + 1) // 2) ** 2, width))
        with torch.no_grad():   # (input revision 2: the L token's initial output scaled as the other kinds')
            self.l_cnn[-1].weight.mul_(L_INIT_GAIN)
            self.l_cnn[-1].bias.mul_(L_INIT_GAIN)
        self.log_cols = {t: [(VI.PLAN[t]['num'].index(V3.col(cols, name)), how)
                             for (t_, name), how in LOG_INPUTS.items() if t_ == t]
                         for t, cols in (('E', V3.E_COLS), ('S', V3.S_COLS))}
        self.kind = nn.Parameter(torch.zeros(len(KINDS), width))
        nn.init.normal_(self.kind, std=0.02)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))

    def _emb(self, table, rows, real=None):
        """[..., k] int rows -> [..., k * dim] (the k columns' embeddings side by side); a padded token's rows (real
        False) are row 0 and every row is clamped into the table (a reused buffer's stale ids never index past it)."""
        rows = rows.long()
        if real is not None:
            rows = torch.where(real[..., None], rows, torch.zeros_like(rows))
        e = table(rows.clamp(0, table.num_embeddings - 1))
        return e.reshape(*rows.shape[:-1], rows.shape[-1] * e.shape[-1])

    def _log_inputs(self, x, t):
        """x [..., numeric columns] of token type t with LOG_INPUTS' columns mapped back (float32; the clamp before)."""
        if not self.log_cols.get(t):
            return x
        x = x.float().clone()
        for j, how in self.log_cols[t]:
            c = x[..., j]
            if how == 'age':
                x[..., j] = torch.log2(1 + 60 * c.clamp(min=0)) / 2
            else:
                u = 10 * c.clamp(min=0)
                x[..., j] = torch.where(u <= 1, u, 1 + torch.log2(u.clamp(min=1)))
        return x

    def tokens(self, v, pending, gtok):
        """v: v3_tensors(pad_batch(...)) on the model's device; pending [B, >= 5] long (the action under way); gtok [width]
        (the global token's parameter). Returns (x [B, T, width], pad [B, T] bool, g index)."""
        B, PM = v['P_num'].shape[:2]
        k = self.kind
        # (Codex, step 2's review: padding was masked only in attention -- a NaN left in a padded row of a reused buffer
        #  reached the real tokens through 0 x NaN, a stale id past a table raised. A padded token's inputs are zeroed
        #  and its ids taken as row 0 before anything reads them, and its token is zeroed before the blocks.)
        pm, em, nm, sm = v['P_mask'], v['E_mask'], v['N_mask'], v['S_mask']
        lm = pm & ~v['local_pad']
        z = lambda x, m: torch.where(m.view(*m.shape, *([1] * (x.dim() - m.dim()))), x, torch.zeros_like(x))   # noqa: E731
        one = torch.cat([F.one_hot(pending[:, i].long().clamp(0, h - 1), h) for i, h in enumerate(PENDING_HEADS)],
                        -1).float()
        pend = torch.zeros((B, PM, self.n_pending), device=one.device)
        pend[:, 0] = one
        P = torch.cat([z(v['P_num'], pm), z(v['P_flag'], pm), self._emb(self.cat_emb, v['P_cat'], pm), z(v['P_off'], pm),
                       z(v['P_rel'], pm), z(v['P_rel_valid'][..., None], pm), pend], -1)
        tp = self.p_mlp(P) + k[0]
        loc = z(v['local'], lm).permute(0, 1, 4, 2, 3).reshape(B * PM, RC, V3.LOCAL_SIDE, V3.LOCAL_SIDE)
        tl = self.l_cnn(loc).view(B, PM, -1) + k[1]
        E = torch.cat([z(self._log_inputs(v['E_num'], 'E'), em), z(v['E_flag'], em), self._emb(self.id_emb, v['E_id'], em),
                       self._emb(self.cat_emb, v['E_cat'], em), self._emb(self.anim_emb, v['E_anim'], em),
                       z(v['E_price'], em), z(v['E_rel'], em), z(v['E_laser_rel'], em)], -1)
        te = self.e_mlp(E) + k[2]
        Nn = torch.cat([z(v['N_num'], nm), z(v['N_flag'], nm), self._emb(self.item_emb, v['N_item'], nm)], -1)
        tn = self.n_mlp(Nn) + k[3]
        S = torch.cat([z(self._log_inputs(v['S_num'], 'S'), sm), z(v['S_flag'], sm), self._emb(self.cat_emb, v['S_cat'], sm),
                       self._emb(self.item_emb, v['S_item'], sm)], -1)
        ts = self.s_mlp(S) + k[4]
        r = self.r_cnn(v['room'].permute(0, 3, 1, 2))                       # [B, 64, 4, 7]
        r = r.flatten(2).transpose(1, 2)                                     # [B, 28, 64] (block row-major)
        tr = self.r_out(torch.cat([r, v['room_xy'], v['room_rel']], -1)) + k[5]
        side = V3.MAP_SIDE
        mi = self.icon_emb(v['map_icon'].long()).view(B, side, side, -1)
        mb = v['map_bits'].view(B, side, side, 4)
        tm = (self.m_cnn(torch.cat([mb, mi], -1).permute(0, 3, 1, 2)) + k[6])[:, None]
        tg = (gtok.view(1, 1, -1) + k[7]).expand(B, 1, -1)
        x = torch.cat([tp, tl, te, tn, ts, tr, tm, tg], 1)
        dev = x.device
        ones = lambda n: torch.ones((B, n), dtype=torch.bool, device=dev)   # noqa: E731
        real = torch.cat([pm, lm, em, nm, sm, ones(tr.shape[1]), ones(1), ones(1)], 1)
        return z(x, real), ~real, x.shape[1] - 1

    def forward(self, v, pending, gtok):
        """(x [B, T, width] after the blocks, pad [B, T], g index)."""
        x, pad, g = self.tokens(v, pending, gtok)
        bias = torch.zeros((x.shape[0], 1, 1, x.shape[1]), device=x.device, dtype=x.dtype).masked_fill(
            pad[:, None, None], -1e9)
        for block in self.blocks:
            x = block(x, bias)
        return x, pad, g


def v3_tensors(padded, device):
    """v3_input.pad_batch's numpy dict -> torch tensors on `device` (float32 / int64 / bool as they are)."""
    import numpy as np
    out = {}
    for k, a in padded.items():
        t = torch.from_numpy(np.ascontiguousarray(a))
        out[k] = t.to(device, non_blocking=True)
    return out
