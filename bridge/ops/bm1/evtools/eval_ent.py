"""main1 held-out evaluation (the coordinator, 2026-11-14): the training tree's own eval_tok.py, run unchanged, plus
two read-only side outputs in its --out directory:
  entropy.json  the policy's mean entropy per action head over every decision it acted on (TokPolicy.split wrapped:
                the heads' logits after the action gate are read, nothing they return is changed; computing it draws
                no random numbers, so the sampled actions are those of a plain eval_tok run); 'use_prob': the bomb /
                item / pill heads' probability of "use" after the gate, mean over every decision ('all') and over the
                decisions whose gate was open ('open', with 'open_share'); 'use_actions': decisions whose sampled
                action said "use" per head (eval_tok.policy_actions wrapped: its returned actions counted)
  config.json   the checkpoint's config, update, decisions, game hours (whatever the checkpoint holds)
usage (in the container, cwd = <tree>/bridge/python): python /bench/bm1/evtools/eval_ent.py <eval_tok.py arguments>"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())   # (the tree's bridge/python: eval_tok and isaac_bridge)
import eval_tok  # noqa: E402

HEADS = ('move', 'shoot', 'bomb', 'item', 'pill')


def main():
    import torch
    acc = {}
    use = {k: [0.0, 0.0, 0] for k in (2, 3, 4)}   # head: [sum of p(use), sum over open gates, open gates]
    acts = {k: 0 for k in (2, 3, 4)}
    meta = {}
    orig = eval_tok.build_eval_model

    def build(ck, items, device, item_slots=None):
        model, changed = orig(ck, items, device, item_slots)
        meta.update(config={k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else repr(v))
                            for k, v in dict(ck.get('config', {})).items()},
                    **{k: ck.get(k) for k in ('update', 'decisions', 'game_hours') if k in ck},
                    heads=list(getattr(model, 'heads', [])))
        inner = model.split

        def split(logits, gate):
            out = inner(logits, gate)
            with torch.no_grad():
                for k, h in enumerate(out):
                    lp = torch.log_softmax(h.float(), -1)
                    ent = -(lp.exp() * lp).sum(-1)
                    s = acc.setdefault(k, [0.0, 0])
                    s[0] += float(ent.sum())
                    s[1] += int(ent.numel())
                    if k in use:
                        g_ = gate if gate.dim() == 2 else (gate > 0)[:, None].expand(-1, 3)
                        p1 = lp[:, 1].exp()
                        op = g_[:, k - 2]
                        use[k][0] += float(p1.sum())
                        use[k][1] += float(p1[op].sum())
                        use[k][2] += int(op.sum())
            return out
        model.split = split   # (an instance attribute: TokPolicy.act calls self.split)
        return model, changed
    eval_tok.build_eval_model = build
    orig_pa = eval_tok.policy_actions

    def policy_actions(*a, **k):
        r = orig_pa(*a, **k)
        for h in acts:
            if h < r.shape[1]:
                acts[h] += int((r[:, h] == 1).sum())
        return r
    eval_tok.policy_actions = policy_actions
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    try:
        eval_tok.main()
    finally:
        if out.is_dir():
            ent = {(HEADS[k] if k < len(HEADS) else f'head{k}'): v[0] / max(v[1], 1) for k, v in sorted(acc.items())}
            ent['decisions'] = max((v[1] for v in acc.values()), default=0)
            ent['greedy'] = '--greedy' in sys.argv
            n_ = max(ent['decisions'], 1)
            ent['use_prob'] = {HEADS[k]: dict(all=v[0] / n_, open=v[1] / max(v[2], 1), open_share=v[2] / n_)
                               for k, v in use.items() if k in acc}
            ent['use_actions'] = {HEADS[k]: v for k, v in acts.items() if k in acc}
            (out / 'entropy.json').write_text(json.dumps(ent, indent=1))
            (out / 'config.json').write_text(json.dumps(meta, indent=1, default=repr))


if __name__ == '__main__':
    main()
