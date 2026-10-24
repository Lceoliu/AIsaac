"""Localize component compile drift on actual recorded observations."""
import argparse,gc,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke
import torch
from isaac_bridge.tok_policy import TokPolicy
from isaac_bridge.source_pipeline import SourcePipeline

p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--checkpoint',required=True)
a=p.parse_args();torch.set_num_threads(2)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
job=torch.load(a.job,map_location='cpu',weights_only=False)
rows=job['arrays']['rows'][0,:64];pending=job['arrays']['pending'][0,:64]
captures=[]
for enabled in ['0','1']:
    os.environ['ISAAC_SOURCE_POINTWISE_RECOMPUTE']=enabled
    os.environ['ISAAC_SOURCE_POINTWISE_COMPONENTS']='1'
    model=TokPolicy(**ck['config']).to('cuda:0');model.load_state_dict(ck['model'])
    pipe=SourcePipeline(model,['cuda:1'],micro=2048,amp=True,compile_source=1,flash_source=True)
    replica=pipe.versions[0][0];captured={};handles=[]
    for name in ['entity','door','patch','room','map']:
        def capture(module,inputs,output,name=name):
            captured[name]=dict(input=inputs[0].detach().float().cpu(),output=output.detach().float().cpu(),
                dtype=str(inputs[0].dtype),weight_dtype=str(next(module.parameters()).dtype))
        handles.append(getattr(replica,name).register_forward_hook(capture))
    result=pipe._encode(0,[(rows,pending)],0)
    captures.append(captured)
    for h in handles:h.remove()
    pipe.close();del result,replica,pipe,model;gc.collect();torch.cuda.empty_cache()
for name in captures[0]:
    ref,got=captures[0][name],captures[1][name]
    delta=got['output']-ref['output']
    print(json.dumps(dict(component=name,input_max_abs=float((got['input']-ref['input']).abs().max()) if ref['input'].numel() else 0,
        output_max_abs=float(delta.abs().max()) if delta.numel() else 0,
        output_relative_l2=float(delta.norm()/ref['output'].norm()) if ref['output'].numel() else 0,
        input_dtype=got['dtype'],weight_dtype=got['weight_dtype'],shape=list(delta.shape))),flush=True)
