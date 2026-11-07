"""Stateless graph encoding must not change delayed-action memory semantics."""
import copy
import numpy as np
import pytest
import torch
from test_block_memory import make_policy
from test_visible_input import synth_rows,synth_pending,components
from isaac_bridge.structured_input import write_frame,pack_frame
from isaac_bridge.block_training import BlockActor


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph contract')
@pytest.mark.parametrize('bf16',[False,True])
def test_graph_matches_eager_across_buckets_publish_and_reset(bf16):
    torch.manual_seed(611)
    m=make_policy('cuda').eval()
    g=copy.deepcopy(m)
    a=BlockActor(m,4)
    b=BlockActor(g,4,graph_entities=32,bf16=bf16)
    rng=np.random.default_rng(611)
    for t,n in enumerate([1,3,4,2,3,1,4,4]):
        rows=synth_rows(rng,n)
        rows['first']=int(t in (0,6));rows['episode']=1+int(t>=6)
        if t==4:
            write_frame(rows[:1],pack_frame(*components(bodies=5,entities=128,inventory=80,slots=11)))
        pending=synth_pending(rng,n,m.heads)
        keys=(np.full(n,t),np.arange(n))
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            aa=a.step(rows,pending,range(n),ukeys=keys)
            bb=b.step(rows,pending,range(n),ukeys=keys)
        for x,y in zip(aa,bb):
            torch.testing.assert_close(x,y,rtol=3e-3 if bf16 else 3e-5,atol=3e-3 if bf16 else 3e-6)
        for i in range(n):
            assert a.memory[i].step==b.memory[i].step
            torch.testing.assert_close(a.memory[i].partial,b.memory[i].partial,
                # Upward token/batch padding changes BF16 reduction tiling (one
                # BF16 ulp around unit scale); FP32 remains tightly matched.
                rtol=1e-2 if bf16 else 3e-5,atol=1e-2 if bf16 else 3e-6)
        if t==3:
            with torch.no_grad():
                m.structured_spatial.body[0].weight.add_(.002)
            b.publish(m.state_dict(),1)
            a.version=1
    shapes=[{k:shape for k,shape,dtype in key} for key in b.encoder.graphs]
    assert {s['v_body'][0] for s in shapes}=={1,2,4}
    assert any(s['v_body'][1]==8 and s['v_entity'][1]==128 and s['v_inventory'][1]==128
               and s['v_slots'][1]==16 for s in shapes)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA capture admission')
def test_capture_miss_does_not_overlap_learner():
    import threading
    from isaac_bridge.block_training import learner_cuda_activity
    held,release=threading.Event(),threading.Event()
    def learner():
        with learner_cuda_activity():
            held.set();assert release.wait(20)
    thread=threading.Thread(target=learner);thread.start();assert held.wait(20)
    m=make_policy('cuda').eval();actor=BlockActor(m,1,graph_entities=128)
    rows=synth_rows(np.random.default_rng(1),1);pending=np.zeros((1,5),np.int64)
    try:
        _,busy=actor.encoder(rows,pending)
        assert actor.encoder.busy_misses==1 and len(actor.encoder.graphs)==0
    finally:
        release.set();thread.join()
    _,idle=actor.encoder(rows,pending)
    assert len(actor.encoder.graphs)==1
    torch.testing.assert_close(busy,idle,rtol=3e-5,atol=3e-6)
