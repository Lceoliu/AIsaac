"""Small human BC experiment; validation selects weights before paired sim tests."""
import gzip
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import default_collate
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.save_util import load_from_zip_file

from .human_dataset import HumanDemonstrations, masked_bc_loss
from .transformer_policy import CombatTransformer


def policy_from_checkpoint(checkpoint, space, device):
    policy = MaskableMultiInputActorCriticPolicy(space, spaces.MultiDiscrete([45,2,2]), lambda _:3e-4,
        features_extractor_class=CombatTransformer,
        features_extractor_kwargs=dict(features_dim=256,layers=4,heads=8),
        net_arch=dict(pi=[256],vf=[256]),normalize_images=False).to(device)
    _, params, _ = load_from_zip_file(Path(checkpoint)/'model.zip',device=device,load_data=False)
    policy.load_state_dict(params['policy'],strict=True)
    return policy


def get_batch(dataset, indices, device):
    batch = default_collate([dataset[int(i)] for i in indices])
    batch['obs'] = {k:v.to(device) for k,v in batch['obs'].items()}
    for k in ('acts','label_mask','action_masks'):
        batch[k] = batch[k].to(device)
    return batch


def action_metrics(target, predicted, masks, nll, previous):
    valid = masks[:,0]
    correct = target[:,0] == predicted[:,0]
    known = valid & (previous[:,3] == 1)
    changed = known & (previous[:,0] != target[:,0])
    bomb = masks[:,1];positive = bomb & (target[:,1] == 1)
    proposed = bomb & (predicted[:,1] == 1)
    return dict(joint_labels=int(valid.sum()),joint_nll=float(nll[valid,0].mean()),
        joint_accuracy=float(correct[valid].mean()),
        move_accuracy=float((target[valid,0]//5 == predicted[valid,0]//5).mean()),
        shoot_accuracy=float((target[valid,0]%5 == predicted[valid,0]%5).mean()),
        changed_action_samples=int(changed.sum()),
        changed_action_accuracy=float(correct[changed].mean()) if changed.any() else None,
        persistence_accuracy=float((previous[known,0] == target[known,0]).mean()) if known.any() else None,
        bomb_labels=int(bomb.sum()),bomb_nll=float(nll[bomb,1].mean()) if bomb.any() else None,
        bomb_positives=int(positive.sum()),bomb_predicted_positives=int(proposed.sum()),
        bomb_true_positives=int((positive & proposed).sum()))


def score_policy(policy, dataset, device):
    values = [[],[],[],[],[]]
    policy.set_training_mode(False)
    with torch.inference_mode():
        for start in range(0,len(dataset),32):
            b = get_batch(dataset,range(start,min(start+32,len(dataset))),device)
            dist = policy.get_distribution(b['obs'],action_masks=b['action_masks'])
            pred = torch.stack([d.probs.argmax(-1) for d in dist.distributions],-1)
            loss = torch.stack([-d.log_prob(b['acts'][:,h]) for h,d in enumerate(dist.distributions)],-1)
            last = b['obs']['history_mask'].sum(-1).long()-1
            old = b['obs']['previous_action'][torch.arange(len(last),device=device),last]
            for dst,src in zip(values,[b['acts'],pred,b['label_mask'],loss,old]):
                dst.append(src.cpu().numpy())
    return action_metrics(*(np.concatenate(v) for v in values))


def export_candidate(source, output, policy, selection):
    """SB3-compatible policy archive, with empty optimizer state; NOT PPO resume."""
    output.mkdir()
    replacements = {}
    for name, obj in [('policy.pth',policy.state_dict()),('policy.optimizer.pth',policy.optimizer.state_dict())]:
        stream = io.BytesIO();torch.save(obj,stream);replacements[name] = stream.getvalue()
    with zipfile.ZipFile(source/'model.zip') as src, zipfile.ZipFile(output/'model.zip','w') as dst:
        for entry in src.infolist():
            dst.writestr(entry,replacements.get(entry.filename,src.read(entry.filename)))
    state = json.loads((source/'state.json').read_text())
    state['bc'] = selection
    state['resume_supported'] = False
    (output/'state.json').write_text(json.dumps(state,indent=2),encoding='utf8')


def fit(config):
    output = Path(config['output'])
    train = HumanDemonstrations(config['dataset'],'train',cache_episodes=5)
    validation = HumanDemonstrations(config['dataset'],'validation',cache_episodes=1)
    torch.manual_seed(config['seed']);rng = np.random.default_rng(config['seed'])
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    policy = policy_from_checkpoint(config['checkpoint'],train.space,config['device'])
    # Original PPO optimizer is not reused. Critic-only parameters get no BC gradients.
    optimizer = torch.optim.Adam(policy.parameters(),lr=config['learning_rate'])
    best_loss = float('inf');best_epoch = 0;wait = 0;steps = 0;history = []
    with (output/'learning-curve.jsonl').open('w',encoding='utf8') as log:
        for epoch in range(config['max_epochs']+1):
            started = time.perf_counter()
            if epoch:
                policy.set_training_mode(True)
                order = rng.permutation(len(train))
                for start in range(0,len(order),32):
                    batch = get_batch(train,order[start:start+32],config['device'])
                    optimizer.zero_grad(set_to_none=True)
                    loss = masked_bc_loss(policy,batch)
                    if not torch.isfinite(loss):raise ValueError('Non-finite BC loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
                    optimizer.step();steps += 1
            row = dict(epoch=epoch,optimizer_steps=steps,
                train=score_policy(policy,train,config['device']),
                validation=score_policy(policy,validation,config['device']))
            row['seconds'] = time.perf_counter()-started
            improved = row['validation']['joint_nll'] < best_loss-config['min_delta']
            if improved:
                best_loss = row['validation']['joint_nll'];best_epoch = epoch;wait = 0
                torch.save(policy.state_dict(),output/'best-policy.pt')
            else:wait += 1
            row.update(selected_so_far=best_epoch,validation_regression=not improved)
            history.append(row);log.write(json.dumps(row)+'\n');log.flush()
            print(json.dumps(row),flush=True)
            if wait >= config['patience']:break
    policy.load_state_dict(torch.load(output/'best-policy.pt',map_location=config['device'],weights_only=True))
    selection = dict(selected_epoch=best_epoch,stopped_epoch=history[-1]['epoch'],optimizer_steps=steps,
        selection_metric='sixth-episode joint NLL only; 64 paired-test seeds never used for selection',
        early_stopped=wait >= config['patience'],initial=history[0],selected=history[best_epoch],last=history[-1])
    export_candidate(Path(config['checkpoint']),output/'candidate',policy,selection)
    # Reopen the actual SB3 archive and verify exact selected policy tensors.
    _, params, _ = load_from_zip_file(output/'candidate/model.zip',device=config['device'],load_data=False)
    assert all(torch.equal(v,params['policy'][k]) for k,v in policy.state_dict().items())
    (output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf8')
    return selection


def replay_metrics(folder):
    summary = json.loads((folder/'summary.json').read_text())
    results = []
    initial = {}
    for episode in summary['results']:
        path = folder/episode['replay']
        with gzip.open(path,'rt',encoding='utf8') as f:
            lines = [json.loads(line) for line in f]
        rows = lines[1:]
        initial[episode['seed']] = rows[0]['state']
        hp = [row['state']['hp'] for row in rows]
        damage = [max(0,a-b) for a,b in zip(hp,hp[1:])]
        results.append(dict(**episode,seconds=episode['frames']/30,
            hurt_events=sum(d>0 for d in damage),health_lost=sum(damage),
            initial_hp=hp[0],final_hp=hp[-1]))
        if not (folder/f"seed-{episode['seed']}.html").read_text(encoding='utf8').startswith('<!doctype html>'):
            raise ValueError('Missing replay HTML')
        assert rows[-1]['done'] and rows[-1]['outcome'] == episode['outcome']
    wins = [r for r in results if r['outcome']=='win']
    aggregate = dict(episodes=len(results),wins=len(wins),win_rate=len(wins)/len(results),
        deaths=sum(r['outcome']=='death' for r in results),timeouts=sum(r['outcome']=='time_limit' for r in results),
        mean_hurt_events=float(np.mean([r['hurt_events'] for r in results])),
        mean_health_lost=float(np.mean([r['health_lost'] for r in results])),
        win_mean_seconds=float(np.mean([r['seconds'] for r in wins])) if wins else None)
    return aggregate,results,initial


def compare_evaluations(a_dir,b_dir):
    a,ar,ai = replay_metrics(a_dir);b,br,bi = replay_metrics(b_dir)
    if ai != bi:raise ValueError('Paired evaluation initial states differ')
    amap = {r['seed']:r for r in ar};bmap = {r['seed']:r for r in br}
    differences = np.array([int(bmap[s]['outcome']=='win')-int(amap[s]['outcome']=='win') for s in amap])
    rng = np.random.default_rng(23)
    ci = np.quantile(rng.choice(differences,(10000,len(differences)),replace=True).mean(1),[.025,.975])
    common = [s for s in amap if amap[s]['outcome']=='win' and bmap[s]['outcome']=='win']
    return dict(original=a,behavior_cloned=b,identical_initial_states=True,
        paired_win_gain=float(differences.mean()),paired_bootstrap_95ci=ci.tolist(),
        gained_wins=int((differences==1).sum()),lost_wins=int((differences==-1).sum()),
        both_win_seeds=common,
        common_win_time_difference_seconds=float(np.mean([bmap[s]['seconds']-amap[s]['seconds'] for s in common])) if common else None,
        episodes_original=ar,episodes_behavior_cloned=br)


def write_comparison_index(output, report):
    a = {r['seed']:r for r in report['episodes_original']}
    b = {r['seed']:r for r in report['episodes_behavior_cloned']}
    rows = []
    for seed in a:
        cells = [f'<td>{seed}</td>',f'<td>{a[seed]["layout"]}</td>']
        for arm,data in [('original',a),('behavior-cloned',b)]:
            r = data[seed]
            cells.append(f'<td><a href="{arm}/seed-{seed}.html">{r["outcome"]}</a></td>'
                         f'<td>{r["seconds"]:.1f}</td><td>{r["hurt_events"]}</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Isaac BC 配对实战</title>
<style>body{font:16px system-ui;background:#171a1e;color:#ddd;max-width:1100px;margin:32px auto}
a{color:#91bedb}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #444;text-align:left}</style>
<h1>Isaac · 行为克隆配对实战</h1><p>PPO 原权重 vs 从同一权重进行行为克隆后由验证集选出的模型。相同种子、出生位置与地形，120 秒上限。</p>
<p>点击结果打开逐帧回放。几何状态回放，不是原版贴图视频。</p>
<p><a href="comparison.json">完整指标</a> · <a href="selection.json">离线验证与选模记录</a></p>
<table><tr><th>种子</th><th>地形</th><th>原模型</th><th>秒</th><th>受伤</th><th>行为克隆</th><th>秒</th><th>受伤</th></tr>'''
    (output/'index.html').write_text(html+''.join(rows)+'</table></html>',encoding='utf8')
