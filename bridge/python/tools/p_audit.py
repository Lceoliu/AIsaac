"""P package (Codex round 17, gate 3 in a real training run): the rollouts handed to the learner reconciled with what
the workers sent and read back.

A training run with ISAAC_TOK_AUDIT=<dir> leaves, per worker, its first ISAAC_TOK_AUDIT_N records as sent (episode, t,
the snapshot request, the row's bytes up to its frame's end) and each reply as read (the actions, main_value, main_vg)
-- tok_floor's audit_sent / audit_got --, and the book as configured before any record (book.pkl: its state, the
truncation rule, the reward terms); --dump-rollouts K the first K rollouts as the learner got them (rows, pending,
actions, logp, value, reward, terminal, boot, version, length, open).

This tool rebuilds every worker's decisions from its own log alone -- each record through a fresh Episodes with the
saved state (arrive: go, the rewards, ends, truncations' boots, the pending resets) and store, with the actions the
worker read -- and reconciles each handed rollout with it, decision by decision, by identity (worker, episode, t):
  - the row as handed against the bytes the worker sent (every field; the frame up to its end);
  - pending, actions against the rebuild; actions, value against what the worker read (value as float32);
  - reward, terminal, boot of every closed decision against the rebuild;
  - logp finite; version never decreasing per worker;
  - sequence: per worker, the decisions of rollouts 1..K (a rollout's open decision carried as the next one's first,
    identical in every field) are exactly the rebuild's, in order, none lost / doubled -- up to the last record logged;
  - carry: a rollout's open decision equals the next rollout's first one (rows, pending, actions, logp, value,
    version, boot).
Run it on a Python-loop run too (ISAAC_TOK_CPP_DRIVER=0): the same checks must pass there (the tool's own control).

   usage: python tools/p_audit.py <audit dir> <rollouts dir> [--out f.json]
Prints one JSON object; exit 0 when everything reconciles."""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def load_log(path):
    sends, replies = [], {}
    with open(path, 'rb') as f:
        while True:
            try:
                e = pickle.load(f)
            except EOFError:
                break
            except Exception:   # (a record cut by the worker's end)
                break
            if e[0] == 's':
                sends.append(dict(ep=e[1], t=e[2], cf=e[3], raw=e[4]))
            else:
                replies[(e[1], e[2])] = dict(actions=e[3], value=e[4], vg=e[5])
    return sends, replies


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('audit')
    ap.add_argument('rollouts')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    import __main__
    import train_tok as TT
    from isaac_bridge.tok_obs import ROW
    for k, v in vars(TT).items():   # (the book's state may name train_tok's own objects, pickled as __main__'s)
        if not hasattr(__main__, k):
            setattr(__main__, k, v)
    meta = pickle.load(open(os.path.join(a.audit, 'book.pkl'), 'rb'))
    TT.CAP_T, TT.TRUNC = int(meta['cap_t']), int(meta['trunc'])
    W, nh = int(meta['workers']), int(meta['nh'])
    fa = ROW.fields['visible_frame'][1]
    res = dict(audit=a.audit, driver=meta.get('driver'), workers=W, cap_t=TT.CAP_T, trunc=TT.TRUNC, problems=[])
    P = res['problems']

    def problem(s):
        if len(P) < 60:
            P.append(s)

    # ---- the workers' logs
    logs = [load_log(os.path.join(a.audit, f'w{w:03d}.pkl')) for w in range(W)]
    res['records_logged'] = int(sum(len(s) for s, _ in logs))

    # ---- the rebuild: each worker's records through a fresh book (the saved state) and store
    book = TT.Episodes.__new__(TT.Episodes)
    book.__dict__.update(meta['book'])
    book.origin, book.plr, book.origin_rec = None, None, None
    cap = max(len(s) for s, _ in logs) + 2
    roll = TT.Rollout(W, cap, nh)
    pending = np.zeros((W, nh), np.int64)
    ident = {}             # (w, ep, t) -> the rebuild's decision index k
    order = [[] for _ in range(W)]   # the rebuild's decisions per worker, in order: (ep, t)
    last_ok = np.zeros(W, np.int64)  # records rebuilt per worker (stops at the first reply missing)
    for w in range(W):
        sends, replies = logs[w]
        for i, s in enumerate(sends):
            row = np.zeros(1, ROW)
            b = row.view(np.uint8).reshape(-1)
            b[:len(s['raw'])] = np.frombuffer(s['raw'], np.uint8)
            idx = np.array([w], np.int64)
            go = book.arrive(roll, row, idx, pending)
            if go[0]:
                r = replies.get((s['ep'], s['t']))
                if r is None or r['actions'] is None:
                    break   # (no reply read: the log's end)
                act = np.array([r['actions']], np.int64)
                k = int(roll.length[w])
                TT.store(roll, idx, row, pending[idx].copy(), act, np.zeros(1, np.float32),
                         np.array([np.float32(r['value'] if r['value'] is not None else 0.0)]), 0)
                ident[(w, s['ep'], s['t'])] = k
                order[w].append((s['ep'], s['t']))
                pending[idx] = act
            last_ok[w] = i + 1
    res['decisions_rebuilt'] = int(sum(len(o) for o in order))

    # ---- the handed rollouts
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    res['rollouts'] = len(paths)
    seen = [[] for _ in range(W)]    # per worker the decisions handed, carried ones once
    carry = [None] * W
    n_cmp = dict(rows=0, closed=0, carried=0, skipped_unlogged=0, boots=0, ends_trunc=0)
    last_ver = np.full(W, -1, np.int64)
    upd_prev = None
    for p in paths:
        z = np.load(p)
        upd = int(z['update'])
        if upd_prev is not None and upd != upd_prev + 1:
            problem(f'rollout {upd}: not consecutive with {upd_prev} (the carry is not checked across)')
            carry = [None] * W
        upd_prev = upd
        rows, length, opn = z['rows'], z['length'], z['open']
        for w in range(W):
            for k in range(int(length[w])):
                r = rows[w, k:k + 1]
                key = (w, int(r['episode'][0]), int(r['t'][0]))
                fields = {f: z[f][w, k] for f in ('pending', 'actions', 'logp', 'value', 'version', 'boot', 'reward',
                                                    'terminal')}
                if k == 0 and carry[w] is not None:   # (the open decision of the rollout before, carried)
                    ck, cr, cf_ = carry[w]
                    same = ck == key and cr.tobytes() == r.tobytes() and all(
                        np.array_equal(cf_[f], fields[f]) for f in ('pending', 'actions', 'logp', 'value', 'version',
                                                                     'boot'))
                    if not same:
                        problem(f'rollout {upd} worker {w}: the carried decision differs ({ck} -> {key})')
                    n_cmp['carried'] += 1
                    carry[w] = None
                    is_dup = True
                else:
                    is_dup = False
                    if carry[w] is not None:
                        problem(f'rollout {upd} worker {w}: the open decision {carry[w][0]} not carried')
                        carry[w] = None
                kr = ident.get(key)
                if kr is None:
                    sends = logs[w][0]
                    if any((s['ep'], s['t']) == key[1:] for s in sends[:int(last_ok[w])]):
                        problem(f'rollout {upd} worker {w}: decision {key} not one the rebuild made')
                    else:
                        n_cmp['skipped_unlogged'] += 1
                    if k == int(opn[w]):
                        carry[w] = (key, r.copy(), fields)
                    continue
                if not is_dup:
                    seen[w].append(key[1:])
                n_cmp['rows'] += 1
                # the row against the sender's bytes
                raw = next(s['raw'] for s in logs[w][0] if (s['ep'], s['t']) == key[1:])
                rb = r.view(np.uint8).reshape(-1)
                bad_f = [f for f in ROW.names if f != 'visible_frame' and ROW.fields[f][1] < fa and
                         rb[ROW.fields[f][1]:ROW.fields[f][1] + ROW.fields[f][0].itemsize].tobytes()
                         != raw[ROW.fields[f][1]:ROW.fields[f][1] + ROW.fields[f][0].itemsize]]
                if bad_f or rb[fa:len(raw)].tobytes() != raw[fa:]:
                    problem(f'rollout {upd} worker {w} {key}: the row differs from the sent one ({bad_f[:6]})')
                # the rebuild and the worker's reply
                if not np.array_equal(fields['pending'], roll.pending[w, kr]):
                    problem(f'rollout {upd} worker {w} {key}: pending {fields["pending"]} vs {roll.pending[w, kr]}')
                if not np.array_equal(fields['actions'], roll.actions[w, kr]):
                    problem(f'rollout {upd} worker {w} {key}: actions {fields["actions"]} vs {roll.actions[w, kr]}')
                rep = logs[w][1].get(key[1:])
                if rep is not None and rep['value'] is not None and \
                        np.float32(fields['value']).tobytes() != np.float32(rep['value']).tobytes():
                    problem(f'rollout {upd} worker {w} {key}: value {fields["value"]} vs the reply {rep["value"]}')
                if not np.isfinite(fields['logp']):
                    problem(f'rollout {upd} worker {w} {key}: logp {fields["logp"]}')
                if int(fields['version']) < last_ver[w]:
                    problem(f'rollout {upd} worker {w} {key}: version {fields["version"]} after {last_ver[w]}')
                last_ver[w] = int(fields['version'])
                if k == int(opn[w]):   # (open: its reward is still to come; carried into the next rollout)
                    carry[w] = (key, r.copy(), fields)
                    continue
                n_cmp['closed'] += 1
                for f in ('reward', 'terminal', 'boot'):
                    ref = getattr(roll, f)[w, kr]
                    if np.asarray(fields[f]).tobytes() != np.asarray(ref, np.asarray(fields[f]).dtype).tobytes():
                        problem(f'rollout {upd} worker {w} {key}: {f} {fields[f]} vs the rebuild {ref}')
                n_cmp['boots'] += int(bool(fields['boot']))
    # every decision of the rebuild handed once, in order, up to the last one handed per worker
    for w in range(W):
        if not seen[w]:
            continue
        last = seen[w][-1]
        try:
            upto = order[w].index(last) + 1
        except ValueError:
            continue
        first = order[w].index(seen[w][0]) if seen[w][0] in order[w] else 0
        want = order[w][first:upto]
        if seen[w] != want:
            miss = [x for x in want if x not in seen[w]][:5]
            dup = [x for x in seen[w] if seen[w].count(x) > 1][:5]
            problem(f'worker {w}: handed {len(seen[w])} decisions, the rebuild {len(want)} (missing {miss}, '
                    f'doubled {dup})')
    res.update(n_cmp)
    res['boot_reasons'] = dict(
        cap=int(sum(1 for w in range(W) for s in logs[w][0][:int(last_ok[w])]
                    if s['raw'] and _done(s['raw']) == 3 and _t(s['raw']) >= TT.CAP_T)),
        stall=int(sum(1 for w in range(W) for s in logs[w][0][:int(last_ok[w])]
                      if s['raw'] and _done(s['raw']) == 3 and _t(s['raw']) < TT.CAP_T)),
        branch=int(sum(1 for w in range(W) for s in logs[w][0][:int(last_ok[w])] if _done(s['raw']) == 4)),
        other_ends=int(sum(1 for w in range(W) for s in logs[w][0][:int(last_ok[w])] if _done(s['raw']) in (1, 2))))
    res['pass'] = not P and n_cmp['rows'] > 0
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)
    sys.exit(0 if res['pass'] else 1)


def _field(raw, name, dt):
    from isaac_bridge.tok_obs import ROW
    o = ROW.fields[name][1]
    return int(np.frombuffer(raw[o:o + np.dtype(dt).itemsize], dt)[0])


def _done(raw):
    return _field(raw, 'done', np.uint8)


def _t(raw):
    return _field(raw, 't', np.int32)


if __name__ == '__main__':
    main()
