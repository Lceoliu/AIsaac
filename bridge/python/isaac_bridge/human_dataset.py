"""Causal, episode-split human demonstrations for the existing masked policy.

Store each 15 Hz observation once, and reconstruct H=64 windows on demand.
Unrepresentable input intervals remain in the history, but never become labels.
This module does not create PPO returns, old log probabilities, or an optimizer.
"""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .human_recording import read_episode, validate_episode
from .transformer_obs import VisibleHistory, DEADLINE_SCHEMA, HISTORY, ENTITY_CAPACITY

MOVE = {(0, 0): 0, (0, -1): 1, (1, -1): 2, (1, 0): 3,
        (1, 1): 4, (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}
SHOOT = {(0, 0, 0, 0): 0, (0, 0, 1, 0): 1, (0, 1, 0, 0): 2,
         (0, 0, 0, 1): 3, (1, 0, 0, 0): 4}


def align_action(header, before, pair):
    """Observation at t -> queries in (t,t+2] -> observation at t+2.

    Direction labels require agreement across ALL native value/pressed polls.
    A bomb pulse is representable only in the first logical frame, with a
    matching resource decrement. A second-frame pulse is not moved backwards.
    Unknown previous actions are explicitly masked, not imputed from outcomes.
    """
    ids, hooks = header['action_ids'], header['hooks']
    queries = [q for row in pair for q in row['queries']]
    acts = np.zeros(3, np.int64)
    known = np.ones(3, bool)
    reasons = []
    directions = []
    for action in ids[:8]:
        values = [q['value'] for q in queries if q['action'] == action and q['hook'] == hooks['value']]
        pressed = [q['value'] for q in queries if q['action'] == action and q['hook'] == hooks['pressed']]
        every_frame = all(any(q['action'] == action and q['hook'] == hooks['value'] for q in row['queries']) for row in pair)
        if not every_frame or not values or len(set(values + pressed)) != 1 or values[0] not in (0, 1):
            known[0] = False
        directions.append(int(values[0]) if values and values[0] in (0, 1) else 0)
    if not known[0]:
        reasons.append('direction_missing_changed_or_analog')
    elif (directions[0] and directions[1]) or (directions[2] and directions[3]) or tuple(directions[4:]) not in SHOOT:
        known[0] = False
        reasons.append('direction_not_in_policy_action_space')
    else:
        move = MOVE[directions[1]-directions[0], directions[3]-directions[2]]
        acts[0] = move * 5 + SHOOT[tuple(directions[4:])]

    player = before['players'][0]
    masks = np.asarray([True]*45 + [True, player['bombs'] > 0] + [True, player['active_ready']], bool)
    if player['bombs'] > 0:
        pulses = [[bool(q['value']) for q in row['queries']
                   if q['action'] == ids[8] and q['hook'] == hooks['triggered']] for row in pair]
        counts = [player['bombs']] + [row['obs']['players'][0]['bombs'] for row in pair]
        if any(pulses[0]) and counts[0]-counts[1] == 1:
            acts[1] = 1
        elif any(pulses[1]) or counts[1] != counts[2]:
            known[1] = False
            reasons.append('bomb_late_or_resource_mismatch')
        elif not all(pulses) or any(pulses[0]) or counts[0] != counts[1]:
            known[1] = False
            reasons.append('bomb_missing_or_resource_mismatch')
    # This curriculum has no active item/card/drop action. Never silently map
    # a future demonstration using one of these to "do nothing".
    unsupported = player['active'] != 0 or any(q['value'] for q in queries if q['action'] in ids[9:])
    if unsupported:
        known[:] = False
        reasons.append('unsupported_item_card_or_drop')
    if any(row['obs']['paused'] or not row['obs']['players'][0]['controls'] or row['wall_delta_ms'] > 250 for row in pair):
        known[:] = False
        reasons.append('pause_controls_or_wall_gap')
    # No loss on an unavailable action head: otherwise post-bomb zeros swamp
    # the few genuine bomb decisions. Item head is disabled in this scenario.
    labels = known & np.asarray([True, player['bombs'] > 0, False])
    return acts, labels, known, masks, reasons


def prepare_dataset(source, output, validation_episode):
    source, output = Path(source).resolve(), Path(output).resolve()
    paths = sorted(source.glob('episode-*.jsonl.gz'))
    if len(paths) < 2 or validation_episode not in {p.name for p in paths}:
        raise ValueError('Need at least two episodes and an existing whole validation episode')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='isaac-human-bc-v1', observation_schema=DEADLINE_SCHEMA,
                  source=str(source), history=HISTORY, entity_capacity=ENTITY_CAPACITY,
                  logic_hz=30, decision_hz=15, action_nvec=[45, 2, 2], batch_size=32,
                  validation_episode=validation_episode, optimizer_steps=0,
                  split_scope='whole episodes; same-seed validation, NOT unseen-seed evaluation',
                  alignment='all-poll stable directions; first-logical-frame bomb pulses only',
                  purpose='masked behavior cloning only; not an on-policy PPO rollout', episodes=[])
    with (output/'alignment.jsonl').open('w', encoding='utf8') as audit:
        for path in paths:
            checked = validate_episode(path)
            rows = read_episode(path)
            header, frames = rows[0], rows[1:-1]
            if header['source'] != 'human' or header['qa_motion'] or header['qa_safe'] or header['logic_hz'] != 30:
                raise ValueError(f'Not a native 30 Hz human demonstration: {path}')
            if checked['outcome'] not in ('win', 'death', 'timeout'):
                raise ValueError(f'Incomplete task episode: {path}')
            encoder = VisibleHistory(history=1, deadline=True)
            before = header['obs']
            stored = {k: [] for k in encoder.space.spaces}
            actions, labels, masks, logical, terminal = [], [], [], [], []
            reasons_count = Counter()
            for start in range(0, len(frames)-1, 2):
                pair = frames[start:start+2]
                encoded = encoder.append(before)
                for key, value in encoded.items():
                    stored[key].append(value[0])
                act, label, known, available, reasons = align_action(header, before, pair)
                actions.append(act); labels.append(label); masks.append(available)
                logical.append(before['logic_frames'])
                terminal.append(pair[-1]['frame'] == len(frames))
                # Only the NEXT observation receives this interval's action.
                encoder.previous_action[:] = (*act, 1) if known.all() else (0, 0, 0, 0)
                if reasons:
                    reasons_count.update(reasons)
                    audit.write(json.dumps(dict(episode=path.name,logic_frame=before['logic_frames'],
                                               reasons=reasons,label_mask=label.tolist()))+'\n')
                before = pair[-1]['obs']
            arrays = {k: np.asarray(v) for k, v in stored.items()}
            arrays.update(acts=np.asarray(actions), label_mask=np.asarray(labels),
                          action_masks=np.asarray(masks), logic_frame=np.asarray(logical), terminal=np.asarray(terminal))
            name = path.name.removesuffix('.jsonl.gz')+'.npz'
            np.savez_compressed(output/name, **arrays)
            # Reopen the actual serialized arrays, not just the in-memory object.
            with np.load(output/name, allow_pickle=False) as loaded:
                for k, value in arrays.items():
                    np.testing.assert_array_equal(loaded[k], value)
            item = dict(file=name, source_file=str(path), split='validation' if path.name == validation_episode else 'train',
                        source_frames=len(frames), decisions=len(actions), odd_tail_frames=len(frames)%2,
                        usable_samples=int(arrays['label_mask'].any(axis=1).sum()),
                        joint_labels=int(arrays['label_mask'][:,0].sum()), bomb_labels=int(arrays['label_mask'][:,1].sum()),
                        bomb_positive_labels=int(((arrays['acts'][:,1] == 1) & arrays['label_mask'][:,1]).sum()),
                        joint_histogram=np.bincount(arrays['acts'][arrays['label_mask'][:,0],0],minlength=45).tolist(),
                        unknown_previous_actions=int((arrays['previous_action'][1:,3] == 0).sum()),
                        excluded_reasons=dict(reasons_count), seed=header['seed'], room_seed=header['room_seed'],
                        outcome=checked['outcome'], compressed_bytes=(output/name).stat().st_size,
                        frame_storage_bytes=sum(a.nbytes for a in arrays.values()))
            report['episodes'].append(item)
    for split in ('train', 'validation'):
        items = [e for e in report['episodes'] if e['split'] == split]
        report[split] = {k: sum(e[k] for e in items) for k in ('usable_samples','joint_labels','bomb_labels','bomb_positive_labels')}
    (output/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    return report


class HumanDemonstrations:
    """Torch Dataset-compatible lazy causal windows, never crossing episodes."""
    def __init__(self, root, split, cache_episodes=2):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf8'))
        if split not in ('train', 'validation'):
            raise ValueError('split must be train or validation')
        self.space = VisibleHistory(deadline=True).space
        self.index = []
        for episode in self.manifest['episodes']:
            if episode['split'] == split:
                with np.load(self.root/episode['file'],allow_pickle=False) as data:
                    self.index.extend((episode['file'],int(i)) for i in np.flatnonzero(data['label_mask'].any(axis=1)))
        # Per-instance cache, rather than an unbounded cache of full H=64 samples.
        self.load = lru_cache(maxsize=cache_episodes)(self._load)

    def _load(self, name):
        with np.load(self.root/name,allow_pickle=False) as data:
            return {k:data[k] for k in data.files}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, end = self.index[index]
        data = self.load(name)
        start = max(0,end-HISTORY+1)
        length = end-start+1
        obs = {}
        for key, space in self.space.spaces.items():
            value = np.zeros(space.shape,dtype=space.dtype)
            value[:length] = data[key][start:end+1]
            obs[key] = value
        return dict(obs=obs,acts=data['acts'][end],label_mask=data['label_mask'][end],
                    action_masks=data['action_masks'][end],episode=name,logic_frame=int(data['logic_frame'][end]))


def masked_bc_loss(policy, batch):
    """Standard categorical BC, masking only unrepresentable/unavailable heads.

    Uses the SAME SB3 masked action distribution as PPO. No human old-policy
    likelihood is invented, and no critic targets are inferred from these demos.
    """
    distribution = policy.get_distribution(batch['obs'], action_masks=batch['action_masks'])
    losses = []
    for head, categorical in enumerate(distribution.distributions):
        valid = batch['label_mask'][:,head].bool()
        if valid.any():
            losses.append(-categorical.log_prob(batch['acts'][:,head])[valid].mean())
    if not losses:
        raise ValueError('Batch has no supervised action labels')
    return sum(losses)
