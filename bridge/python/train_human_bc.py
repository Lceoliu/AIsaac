"""Explicit opt-in BC training followed by 64 paired held-out simulator rooms."""
import argparse
import gc
import json
from pathlib import Path
import torch

from isaac_bridge.bc_experiment import fit,compare_evaluations,write_comparison_index
from isaac_bridge.evaluation import evaluate_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    config=dict(checkpoint=str(args.checkpoint.resolve()),dataset=str(args.dataset.resolve()),output=str(args.out.resolve()),
        seed=23,device='cuda',batch_size=32,learning_rate=1e-4,max_epochs=30,patience=5,min_delta=.001,
        eval_seeds=list(range(2**31+16,2**31+80)),eval_deterministic=True,
        selection='minimum sixth-episode joint NLL; never use paired evaluation to select weights')
    print(json.dumps(config),flush=True)
    if not args.run:return
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    fit(config)
    gc.collect();torch.cuda.empty_cache()
    for name,checkpoint in [('original',args.checkpoint),('behavior-cloned',args.out/'candidate')]:
        print('Starting paired evaluation: '+name,flush=True)
        evaluate_checkpoint(checkpoint,args.out/name,config['eval_seeds'])
        gc.collect();torch.cuda.empty_cache()
    report=compare_evaluations(args.out/'original',args.out/'behavior-cloned')
    (args.out/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    write_comparison_index(args.out,report)
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('episodes_')}),flush=True)


if __name__=='__main__':main()
