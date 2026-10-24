"""Graph/eager equivalence, ring wrap, growth, forks and versioned cache checks."""
import copy,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import test_memory_smoke as M
import test_temporal_smoke as TS
import numpy as np
import torch
from isaac_bridge.tok_policy import TokPolicy,TemporalState,to_batch,crn_uniforms,gumbel_sample
from isaac_bridge.rebuild_actor import RebuildGraphActor

torch.set_num_threads(2);torch.manual_seed(23)
cfg=dict(width=32,layers=1,heads=4,**TS.BASE,**TS.FUSED,temporal=True,
         temporal_width=32,temporal_layers=2,temporal_heads=4,temporal_window=4,
         rebuild_memory=True,memory_block=4,memory_tokens=2,source_cache=True)
model=TokPolicy(**cfg).cuda().eval();ref=copy.deepcopy(model)
state=TemporalState(model,6,'cuda',torch.float32);rs=TemporalState(ref,6,'cuda',torch.float32)
actor=RebuildGraphActor(model,4,entities=[8,16],temporal=state,row_sizes=[2,4],
                        compile_mode='default' if '--compile' in sys.argv else None,
                        sdpa_math='--compile' in sys.argv)
rng=np.random.default_rng(23)
seq=TS.make_seq(rng,nw=4,T=168,changes={(0,4),(0,8),(0,12)},resets=set(),exits=set())
maximum=0.
with torch.no_grad():
 for t,rows in enumerate(seq):
  rows=rows.copy();n=[4,3,2,4][t%4];rows=rows[:n]
  if t==17:rows['first'][0]=1
  if t==35:
   state.copy_slot_(0,1);rs.copy_slot_(0,1)
  pending=M.synth_pending(rng,n,model.heads)
  keys=(np.arange(1,n+1,dtype=np.uint64),np.full(n,t,dtype=np.uint64))
  td=rs.step_inputs(ref,rows,pending,np.arange(n))
  expected=ref(to_batch(rows,pending,'cuda'),temporal=td,need_h=True,need_vg=True)
  uniforms=torch.as_tensor(crn_uniforms(*keys,sum(model.heads)),device='cuda')
  actions,logp=gumbel_sample(ref.split(expected[0],ref.gate(to_batch(rows,pending,'cuda'))),uniforms)
  if t==70:
   # A caller exceeding its graph bucket can still use the eager interface;
   # its next graphed decision must see that step's exact updated history.
   own=state.step_inputs(model,rows,pending,np.arange(n))
   actual=model(to_batch(rows,pending,'cuda'),temporal=own,need_h=True,need_vg=True)
   torch.testing.assert_close(actual[0],expected[0],rtol=4e-4,atol=3e-5)
   continue
  actor.submit(rows,pending,widx=np.arange(n),ukeys=keys);a,lp,v=actor.result()
  if t==0:
   width=actor.nh+2+actor.n_extra+int(actor.vg_on)
   transport=(actor.cpu[:n,:width] if actor.cpu.ndim==2 else actor.cpu[:n*width].view(n,width))
   assert transport.is_contiguous(), f'D2H output has a strided host destination: {transport.stride()}, width={width}'
   owned=state.rebuild.raw[0].tail[0][0];root=owned
   while isinstance(root,np.ndarray) and root.base is not None:root=root.base
   retained=root.nbytes if isinstance(root,np.ndarray) else len(root)
   assert retained==owned.nbytes, f'one episode row pins {retained} bytes instead of {owned.nbytes}'
  err=float((actor.last_logits-expected[0]).abs().max());maximum=max(maximum,err)
  torch.testing.assert_close(actor.last_logits,expected[0],rtol=4e-4,atol=3e-5)
  np.testing.assert_array_equal(a,actions.cpu().numpy())
  np.testing.assert_allclose(lp,logp.cpu().numpy(),rtol=4e-4,atol=3e-5)
  np.testing.assert_allclose(v,expected[1].cpu().numpy(),rtol=4e-4,atol=3e-5)
  np.testing.assert_allclose(actor.vg,expected[-1].cpu().numpy(),rtol=4e-4,atol=3e-5)
 assert state.rebuild.fast.capacity>64
 if '--compile' in sys.argv:
  from torch._dynamo.utils import counters
  compiled=counters['stats']['unique_graphs']
  state.rebuild.fast.grow(state.rebuild.fast.capacity+1)
  actor.submit(rows,pending,widx=np.arange(n),ukeys=keys);actor.result()
  after=counters['stats']['unique_graphs']
  print('Actor capacity growth compiled graphs:',compiled,'->',after)
  assert after==compiled, 'memory capacity growth recompiled the policy'
 before=state.rebuild.fast.u.clone()
 model.reader.inp.weight.add_(.01);model.invalidate_source_cache()
 with patch.object(model,'encode_rebuild',wraps=model.encode_rebuild) as encoder:
  state.rebuild.fast.refresh();assert encoder.call_count==0
 torch.testing.assert_close(before,state.rebuild.fast.u,rtol=0,atol=0)
 # Snapshot-bank copy is independent, including a prefix longer than one capacity.
 snapshot=TemporalState(model,2,'cuda',torch.float32)
 snapshot.copy_slot_(0,1,other=state)
 assert snapshot.rebuild.raw[1].uid!=state.rebuild.raw[0].uid
 torch.testing.assert_close(snapshot.rebuild.fast.u[1],state.rebuild.fast.u[0])
 saved=snapshot.rebuild.fast.u[1].clone();state.rebuild.fast.u[0].zero_()
 torch.testing.assert_close(snapshot.rebuild.fast.u[1],saved)
print('PASS CUDA Graph/eager logits, CRN actions, logp, values; variable batches, wrap, reset, fork, history growth',maximum)
print('PASS actor version change reprojects stored values without any source re-encoding; snapshot isolation')
