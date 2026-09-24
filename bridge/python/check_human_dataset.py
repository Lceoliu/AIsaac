"""Full-size policy/BC gradient smoke test only. No optimizer steps or checkpoint."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import default_collate
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy

from isaac_bridge.human_dataset import HumanDemonstrations, masked_bc_loss
from isaac_bridge.transformer_policy import CombatTransformer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(23)
    data=HumanDemonstrations(args.dataset,'train')
    held=HumanDemonstrations(args.dataset,'validation')
    policy=MaskableMultiInputActorCriticPolicy(data.space,spaces.MultiDiscrete([45,2,2]),lambda _:3e-4,
        features_extractor_class=CombatTransformer,features_extractor_kwargs=dict(features_dim=256,layers=4,heads=8),
        net_arch=dict(pi=[256],vf=[256]),normalize_images=False).to(args.device)
    weights=[p.detach().clone() for p in policy.parameters()]
    rng=np.random.default_rng(23)
    report=dict(device=args.device,batch_size=32,history=64,optimizer_steps=0,checkpoint_written=False,
                weights='fresh random policy for plumbing test; losses are NOT trained performance')
    for split,dataset in [('train',data),('validation',held)]:
        indices=rng.choice(len(dataset),32,replace=False)
        # Exercise actual rare bomb-positive labels, not just all-negative batches.
        positives=[i for i,(name,t) in enumerate(dataset.index)
                   if dataset.load(name)['label_mask'][t,1] and dataset.load(name)['acts'][t,1]==1]
        indices[:len(positives[:4])]=positives[:4]
        batch=default_collate([dataset[int(i)] for i in indices])
        batch['obs']={k:v.to(args.device) for k,v in batch['obs'].items()}
        for k in ('acts','label_mask','action_masks'):batch[k]=batch[k].to(args.device)
        started=time.perf_counter()
        policy.set_training_mode(split=='train')
        with torch.set_grad_enabled(split=='train'):
            loss=masked_bc_loss(policy,batch)
            if not torch.isfinite(loss):raise ValueError('Non-finite BC loss')
            if split=='train':
                loss.backward()
                gradients=[p.grad for p in policy.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise ValueError('Missing or non-finite gradient')
                report['finite_gradient_tensors']=len(gradients)
                report['feature_gradient_nonzero']=bool(policy.features_extractor.fusion[0].weight.grad.abs().sum()>0)
                policy.zero_grad(set_to_none=True)
        report[split]=dict(loss=float(loss.detach()),seconds=time.perf_counter()-started,
                           supervised_per_head=batch['label_mask'].sum(0).cpu().tolist(),
                           bomb_positives=int(((batch['acts'][:,1]==1)&batch['label_mask'][:,1]).sum()),
                           full_history_samples=int((batch['obs']['history_mask'].sum(1)==64).sum()))
    if not all(torch.equal(old,new) for old,new in zip(weights,policy.parameters())):
        raise ValueError('Smoke test changed model weights')
    report['weights_unchanged']=True
    path=args.dataset/'model-check.json';path.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(path.read_text(encoding='utf8'))


if __name__=='__main__':main()
