"""P package (Codex rounds 17-18, gate 3 in a real training run): the rollouts handed to the learner reconciled with
what the workers sent and read back.

A training run with ISAAC_TOK_AUDIT=<dir> leaves, per worker, its first ISAAC_TOK_AUDIT_N records as sent (episode, t,
the snapshot request, the row's bytes up to its frame's end), each reply as read (the actions, main_value, main_vg;
None for an end the worker did not act on), where the log stops ('cut': the cap reached) and the worker's end ('end')
-- tok_floor's audit_sent / audit_got --, and the book as configured before any record (book.pkl: its state, the
truncation rule, the reward terms); --dump-rollouts K the first K rollouts as the learner got them (rows, pending,
actions, logp, value, value_g, reward, terminal, boot, version, length, open).

The rebuild: every worker's records from its own log alone through a fresh Episodes with the saved state (arrive: go,
rewards, ends, truncations' boots, the pending resets) and store, with the actions the worker read. A go record with no
reply in the MIDDLE of a log fails (only a log's last record may lack one: the cut / the end). The rebuild's last
decision has no successor observed exactly when the log stops right after it: the only decision whose reward is not
checked.

The handed rollouts, per worker, against the rebuild -- the boundaries independent of what was handed:
  - the first rollout (update 1) starts at the worker's first decision; each later one starts with the open decision of
    the one before (carried: identical in every field) or, when none was open, with the next decision;
  - every rollout is a contiguous run of the rebuild's decisions (none lost, doubled, reordered);
  - open is -1 or the rollout's last decision, and an open decision has reward 0 and terminal 0 (still to come);
  - per decision: the row (every field before the frame, the frame up to its end) against the sent bytes; pending,
    actions against the rebuild; actions, value, value_g against what the worker read; reward, terminal, boot of every
    closed decision against the rebuild (except the one with no successor observed); logp finite; version never
    decreasing;
  - decisions past the rebuild (the log's cap) only as the tail of a worker's last rollouts, and only when its log was
    cut there.
Boots: those observed (truncation / branch ends asked for) and those whose flag and reward were verified.

   usage: python tools/p_audit.py <audit dir> <rollouts dir> [--log-cap 1500] [--out f.json]
(--log-cap: for logs written before the 'cut' marker: a log of exactly that many records was cut.)
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
    sends, replies, marks = [], {}, {}
    with open(path, 'rb') as f:
        while True:
            try:
                e = pickle.load(f)
            except EOFError:
                break
            except Exception:   # (a record cut by the worker's end)
                marks['torn'] = True
                break
            if e[0] == 's':
                sends.append(dict(ep=e[1], t=e[2], cf=e[3], raw=e[4]))
            elif e[0] == 'r':
                replies[(e[1], e[2])] = dict(actions=e[3], value=e[4], vg=e[5])
            elif e[0] in ('cut', 'end'):
                marks[e[0]] = e[1:]
    return sends, replies, marks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('audit')
    ap.add_argument('rollouts')
    ap.add_argument('--log-cap', type=int, default=1500)
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
    fields_pre = [f for f in ROW.names if ROW.fields[f][1] < fa]
    res = dict(audit=a.audit, driver=meta.get('driver'), workers=W, cap_t=TT.CAP_T, trunc=TT.TRUNC, problems=[])
    P = res['problems']
    n_problems = [0]

    def problem(s):
        n_problems[0] += 1
        if len(P) < 60:
            P.append(s)

    # ---- the workers' logs
    logs = [load_log(os.path.join(a.audit, f'w{w:03d}.pkl')) for w in range(W)]
    res['records_logged'] = int(sum(len(s) for s, _, _ in logs))
    cut = [bool(m.get('cut')) or (len(s) == a.log_cap) for s, _, m in logs]
    res['logs_cut'] = int(sum(cut))
    res['logs_ended'] = int(sum(bool(m.get('end')) for _, _, m in logs))

    # ---- the rebuild
    book = TT.Episodes.__new__(TT.Episodes)
    book.__dict__.update(meta['book'])
    book.origin, book.plr, book.origin_rec = None, None, None
    cap = max(len(s) for s, _, _ in logs) + 2
    roll = TT.Rollout(W, cap, nh, value_g=True)
    pending = np.zeros((W, nh), np.int64)
    ident = {}                        # (w, ep, t) -> the rebuild's decision index
    order = [[] for _ in range(W)]    # the rebuild's decisions per worker: (ep, t)
    unread = set()                    # decisions answered with an action the worker did not read (an end's boot)
    raw_of, reply_of = {}, {}
    no_succ = [None] * W              # the rebuild's last decision when no record after it is in the log
    boots_seen = 0
    for w in range(W):
        sends, replies, _ = logs[w]
        last_dec_i = -1
        for i, s in enumerate(sends):
            row = np.zeros(1, ROW)
            b = row.view(np.uint8).reshape(-1)
            b[:len(s['raw'])] = np.frombuffer(s['raw'], np.uint8)
            idx = np.array([w], np.int64)
            go = book.arrive(roll, row, idx, pending)
            if not go[0]:
                continue
            boots_seen += int(row['done'][0] != 0)
            r = replies.get((s['ep'], s['t']))
            last = i == len(sends) - 1
            if r is None or (r['actions'] is None and row['done'][0] == 0):
                if not last:   # (Codex 18-3: a reply missing in the middle of a log is an error, not its end)
                    problem(f'worker {w}: record {i} ({s["ep"]}, {s["t"]}) has no reply in the middle of the log')
                break
            if r['actions'] is None:
                unread.add((w, s['ep'], s['t']))
                act = np.zeros((1, nh), np.int64)
            else:
                act = np.array([r['actions']], np.int64)
            k = int(roll.length[w])
            TT.store(roll, idx, row, pending[idx].copy(), act, np.zeros(1, np.float32),
                     np.array([np.float32(r['value'] if r['value'] is not None else 0.0)]), 0,
                     vg=np.array([np.float32(r['vg'] if r['vg'] is not None else 0.0)]))
            key = (w, s['ep'], s['t'])
            ident[key] = k
            order[w].append((s['ep'], s['t']))
            raw_of[key], reply_of[key] = s['raw'], r
            pending[idx] = act
            last_dec_i = i
        if order[w] and last_dec_i == len(sends) - 1:   # (nothing after the last decision in the log)
            no_succ[w] = len(order[w]) - 1
    res['decisions_rebuilt'] = int(sum(len(o) for o in order))

    # ---- the handed rollouts
    paths = sorted(glob.glob(os.path.join(a.rollouts, 'rollout-*.npz')))
    res['rollouts'] = len(paths)
    nxt = [0] * W                       # per worker: the rebuild's index of the next handed (not carried) decision
    carry = [None] * W
    beyond = [False] * W                # a worker's handed decisions went past its rebuild
    n_cmp = dict(rows=0, closed=0, carried=0, open=0, unread=0, no_successor=0, past_log=0, boots_verified=0,
                 value_g=0)
    last_ver = np.full(W, -1, np.int64)
    upd_prev, first_upd = None, None
    for p in paths:
        z = np.load(p)
        upd = int(z['update'])
        if first_upd is None:
            first_upd = upd
        if upd_prev is not None and upd != upd_prev + 1:
            problem(f'rollout {upd}: not consecutive with {upd_prev}')
        upd_prev = upd
        rows, length, opn = z['rows'], z['length'], z['open']
        vg_d = z['value_g'] if 'value_g' in z.files else None
        for w in range(W):
            L_, o_ = int(length[w]), int(opn[w])
            if not (o_ == -1 or (L_ > 0 and o_ == L_ - 1)):
                problem(f'rollout {upd} worker {w}: open {o_} is not the last decision ({L_} handed)')
            for k in range(L_):
                r = rows[w, k:k + 1]
                key = (w, int(r['episode'][0]), int(r['t'][0]))
                fl = {f: z[f][w, k] for f in ('pending', 'actions', 'logp', 'value', 'version', 'boot', 'reward',
                                              'terminal')}
                if vg_d is not None:
                    fl['value_g'] = vg_d[w, k]
                if k == 0 and carry[w] is not None:   # (the open decision of the rollout before, carried)
                    ck, cr, cf_ = carry[w]
                    diff_f = [f for f in ('pending', 'actions', 'logp', 'value', 'version', 'boot', 'value_g')
                              if f in cf_ and np.asarray(cf_[f]).tobytes() != np.asarray(fl[f]).tobytes()]
                    rb0, rb1 = cr.view(np.uint8).reshape(-1), r.view(np.uint8).reshape(-1)
                    diff_f += ['row.' + f for f in ROW.names if
                               rb0[ROW.fields[f][1]:ROW.fields[f][1] + ROW.fields[f][0].itemsize].tobytes() !=
                               rb1[ROW.fields[f][1]:ROW.fields[f][1] + ROW.fields[f][0].itemsize].tobytes()]
                    if ck != key or diff_f:
                        problem(f'rollout {upd} worker {w}: the carried decision differs ({ck} -> {key}: {diff_f[:6]})')
                    n_cmp['carried'] += 1
                    carry[w] = None
                    is_carried = True
                else:
                    is_carried = False
                    if carry[w] is not None:
                        problem(f'rollout {upd} worker {w}: the open decision {carry[w][0]} not carried')
                        carry[w] = None
                    if k == 0 and upd == first_upd and first_upd <= 1 and order[w] and key[1:] != order[w][0]:
                        problem(f'rollout {upd} worker {w}: starts at {key[1:]}, the worker\'s first decision is '
                                f'{order[w][0]}')
                is_open = k == o_
                if is_open:
                    carry[w] = (key, r.copy(), fl)
                    if np.asarray(fl['reward']).tobytes() != np.zeros((), np.asarray(fl['reward']).dtype).tobytes() \
                            or bool(fl['terminal']):
                        problem(f'rollout {upd} worker {w} {key}: open with reward {fl["reward"]} / terminal '
                                f'{fl["terminal"]} (still to come)')
                kr = ident.get(key)
                if not is_carried:   # (the sequence: the next of the rebuild's decisions, none skipped)
                    if beyond[w]:
                        if kr is not None:
                            problem(f'rollout {upd} worker {w} {key}: back in the rebuild after decisions past it')
                    elif nxt[w] < len(order[w]):
                        if key[1:] != order[w][nxt[w]]:
                            problem(f'rollout {upd} worker {w}: handed {key[1:]}, the rebuild\'s next is '
                                    f'{order[w][nxt[w]]} (a decision lost / doubled / reordered)')
                        nxt[w] += 1
                    else:
                        beyond[w] = True
                        if not cut[w]:
                            problem(f'rollout {upd} worker {w} {key}: past the rebuild, the log not cut')
                if kr is None:
                    n_cmp['past_log'] += 1
                    continue
                n_cmp['rows'] += 1
                raw = raw_of[key]
                rb = r.view(np.uint8).reshape(-1)
                bad_f = [f for f in fields_pre if rb[ROW.fields[f][1]:ROW.fields[f][1] + ROW.fields[f][0].itemsize]
                         .tobytes() != raw[ROW.fields[f][1]:ROW.fields[f][1] + ROW.fields[f][0].itemsize]]
                if bad_f or rb[fa:len(raw)].tobytes() != raw[fa:]:
                    problem(f'rollout {upd} worker {w} {key}: the row differs from the sent one ({bad_f[:6]})')
                if not np.array_equal(fl['pending'], roll.pending[w, kr]):
                    problem(f'rollout {upd} worker {w} {key}: pending {fl["pending"]} vs {roll.pending[w, kr]}')
                rep = reply_of[key]
                if key in unread:
                    n_cmp['unread'] += 1
                else:
                    if not np.array_equal(fl['actions'], roll.actions[w, kr]):
                        problem(f'rollout {upd} worker {w} {key}: actions {fl["actions"]} vs {roll.actions[w, kr]}')
                    if rep['value'] is not None and \
                            np.float32(fl['value']).tobytes() != np.float32(rep['value']).tobytes():
                        problem(f'rollout {upd} worker {w} {key}: value {fl["value"]} vs the reply {rep["value"]}')
                    if 'value_g' in fl and rep['vg'] is not None:
                        n_cmp['value_g'] += 1
                        if np.float32(fl['value_g']).tobytes() != np.float32(rep['vg']).tobytes():
                            problem(f'rollout {upd} worker {w} {key}: value_g {fl["value_g"]} vs the reply {rep["vg"]}')
                if not np.isfinite(fl['logp']):
                    problem(f'rollout {upd} worker {w} {key}: logp {fl["logp"]}')
                if int(fl['version']) < last_ver[w]:
                    problem(f'rollout {upd} worker {w} {key}: version {fl["version"]} after {last_ver[w]}')
                last_ver[w] = int(fl['version'])
                if bool(fl['boot']) != bool(roll.boot[w, kr]):
                    problem(f'rollout {upd} worker {w} {key}: boot {fl["boot"]} vs the rebuild {roll.boot[w, kr]}')
                if is_open:
                    n_cmp['open'] += 1
                    continue
                if no_succ[w] == kr:   # (closed when handed; its closing record past the log: the one exemption)
                    n_cmp['no_successor'] += 1
                    continue
                n_cmp['closed'] += 1
                for f in ('reward', 'terminal'):
                    ref = getattr(roll, f)[w, kr]
                    if np.asarray(fl[f]).tobytes() != np.asarray(ref, np.asarray(fl[f]).dtype).tobytes():
                        problem(f'rollout {upd} worker {w} {key}: {f} {fl[f]} vs the rebuild {ref}')
                n_cmp['boots_verified'] += int(bool(fl['boot']))
    res.update(n_cmp)
    res['boots_seen_in_logs'] = int(boots_seen)
    res['n_problems'] = n_problems[0]
    res['pass'] = n_problems[0] == 0 and n_cmp['rows'] > 0 and n_cmp['closed'] > 0
    s = json.dumps(res, indent=1, default=str)
    print(s)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(s)
    sys.exit(0 if res['pass'] else 1)


if __name__ == '__main__':
    main()
