"""Contract tests for block-memory V1; no game process required."""
from dataclasses import replace, asdict
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from isaac_bridge.block_memory import (BlockConfig, BlockMemory, MemoryBlock, short_mask,
    long_mask, historical_blocks, recompute_history, TeacherGenerations, warm_start_spatial)
from isaac_bridge.tok_policy import TokPolicy, to_batch
from isaac_bridge.tok_teacher2 import teacher_v2_loss
from test_visible_input import synth_rows, synth_pending

torch.set_num_threads(4)


def small(**kw):
    d = dict(latent_width=16, width=16, layers=2, heads=2, writer_heads=2,
             window=3, block_steps=2, memory_tokens=2, memory_blocks=3)
    d.update(kw)
    return BlockConfig(**d)


def test_default_spec_and_scaling():
    c = BlockConfig()
    assert (c.latents, c.latent_width, c.window, c.block_steps, c.memory_tokens, c.memory_blocks) == (4,256,128,32,8,256)
    assert c.cross_layers == (1,3)
    assert BlockConfig(width=384,layers=6,heads=6).cross_layers == (2,5)
    assert BlockConfig(width=512,layers=8,heads=8).cross_layers == (3,7)


def test_masks_exact_decision_boundaries():
    q = torch.tensor([[128,128,128,128]])
    k = torch.tensor([[0,1,127,128,129,-1]])
    assert torch.equal(short_mask(q,k,128)[0,0], torch.tensor([False,True,True,True,False,False]))
    assert short_mask(q,q,128).all()
    ends = torch.tensor([[31,63,95,-1]])
    assert torch.equal(long_mask(torch.tensor([[31,32,64,96]]),ends,32,2)[0],
                       torch.tensor([[False,False,False,False],[True,False,False,False],
                                     [True,True,False,False],[False,True,True,False]]))


def test_writer_is_independent_and_all_queries_learn():
    torch.manual_seed(1)
    m = BlockMemory(12,small())
    x = torch.randn(3,7,12,requires_grad=True)
    z = m.resampler(x)
    assert z.shape == (3,4,16)
    z.square().sum().backward()
    assert m.resampler.queries.grad.abs().sum(-1).min() > 0
    blocks = torch.randn(3,2,4,16)
    with torch.no_grad():
        batch = m.writer(blocks)
        separate = torch.cat([m.writer(a[None]) for a in blocks])
    torch.testing.assert_close(batch,separate,atol=2e-6,rtol=2e-5)
    changed = blocks.clone(); changed[0] += 100
    torch.testing.assert_close(m.writer(changed)[1:],batch[1:],atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_incremental_parallel_and_chunk_equivalence(device):
    if device == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    torch.manual_seed(4)
    m = BlockMemory(12,small()).to(device).eval()
    z = torch.randn(2,11,4,16,device=device)
    initial = [m.empty(episode=i) for i in range(2)]
    with torch.no_grad():
        full = m(z,initial)
        states = initial
        parts = []
        for t in range(z.shape[1]):
            r = m(z[:,t:t+1],states); states=r["states"]; parts.append(r["hidden"])
        torch.testing.assert_close(torch.cat(parts,1),full["hidden"],atol=4e-6,rtol=4e-5)
        a = m(z[:,:5],initial); b = m(z[:,5:],a["states"])
        torch.testing.assert_close(b["hidden"],full["hidden"][:,5:],atol=4e-6,rtol=4e-5)
        for sa,sb in zip(states,full["states"]):
            for va,vb in zip(sa.kv,sb.kv):
                for x,y in zip(va,vb): torch.testing.assert_close(x,y,atol=5e-6,rtol=5e-5)
            assert [x.end_step for x in sa.blocks] == [5,7,9]
            assert len(sa.partial)==1 and sa.step==11


def test_future_observation_does_not_change_past():
    torch.manual_seed(9)
    m=BlockMemory(12,small()).eval()
    z=torch.randn(1,9,4,16)
    a=m(z,[m.empty()])["hidden"]
    z[:,5:]+=torch.randn_like(z[:,5:])*20
    b=m(z,[m.empty()])["hidden"]
    torch.testing.assert_close(a[:,:5],b[:,:5],atol=2e-6,rtol=2e-5)


def test_full_spec_boundary_31_32_and_127_128():
    c=small(window=128,block_steps=32,memory_tokens=8,memory_blocks=256)
    m=BlockMemory(12,c).eval()
    with torch.no_grad():
        r=m(torch.randn(1,130,4,16),[m.empty()])
    assert not r["memory_mask"][0,31*4].any()
    assert r["memory_mask"][0,32*4,:8].all()
    assert not r["local_mask"][0,128*4,0]
    assert r["local_mask"][0,127*4,0]
    assert [b.end_step for b in r["states"][0].blocks]==[31,63,95,127]


def test_fork_partial_block_and_reset():
    m=BlockMemory(12,small()).eval()
    with torch.no_grad():
        base=m(torch.randn(1,3,4,16),[m.empty()])["states"][0]
        a,b=base.fork(),base.fork()
        assert a.blocks[0].value is b.blocks[0].value
        assert a.partial.data_ptr()!=base.partial.data_ptr()
        before=base.partial.clone()
        ra=m(torch.arange(64,dtype=torch.float32).reshape(1,1,4,16),[a],version=2)["states"][0]
        rb=m(torch.zeros(1,1,4,16),[b],version=2)["states"][0]
        assert len(base.blocks)==1 and len(ra.blocks)==2 and len(rb.blocks)==2
        torch.testing.assert_close(base.partial,before)
        assert not torch.allclose(ra.blocks[-1].value,rb.blocks[-1].value)
        assert ra.blocks[-1].source_versions==(0,2)
        empty=m.empty()
        assert empty.step==0 and len(empty.blocks)==0 and len(empty.partial)==0


def test_257th_block_evicts_oldest_without_stopping():
    m=BlockMemory(12,small(memory_blocks=256)).eval()
    state=m.empty()
    blocks=tuple(MemoryBlock(2*i+1,torch.randn(2,16),0,(0,0)) for i in range(256))
    state=replace(state,step=512,blocks=blocks)
    with torch.no_grad():
        out=m(torch.randn(1,2,4,16),[state])["states"][0]
    assert len(out.blocks)==256 and out.blocks[0].end_step==3 and out.blocks[-1].end_step==513
    assert len(state.blocks)==256 and state.blocks[0].end_step==1


def test_real_historical_gradient_budget_and_no_cache_mutation():
    torch.manual_seed(22)
    m=BlockMemory(12,small(memory_blocks=5))
    old=m(torch.randn(1,8,4,16),[m.empty()])["states"][0]
    raw={block.end_step:torch.randn(2,4,16,requires_grad=True) for block in old.blocks}
    cached=[b.value.clone() for b in old.blocks]
    replacements,selected=recompute_history(m,[old],lambda b:raw[b.end_step],torch.Generator().manual_seed(5))
    assert len(selected[0])==2 and selected[0][0].end_step==7
    out=m(torch.randn(1,1,4,16),[old],replacements=replacements)
    # Actual CF loss: no current raw encoder in this graph can explain past-source gradients.
    readout=torch.nn.Linear(16,6)
    heads=readout(out["hidden"].flatten(0,1))
    logp=torch.log_softmax(heads[:,:5],-1)[:,2]
    value=heads[:,5]
    loss,_=teacher_v2_loss(logp,value,torch.ones(1),torch.ones(1),1.,.2,.5)
    loss.backward()
    chosen={b.end_step for b in selected[0]}
    for key,x in raw.items():
        assert (x.grad is not None and x.grad.abs().sum()>1e-6) if key in chosen else x.grad is None
    for block,before in zip(old.blocks,cached): torch.testing.assert_close(block.value,before)
    assert all(not k.requires_grad and not v.requires_grad for k,v in old.kv)
    assert m.writer.queries.grad.abs().sum()>0


def test_teacher_generation_pin_backpressure_and_release():
    p=TeacherGenerations()
    m=torch.nn.Linear(2,2)
    assert p.publish(m,0)
    i0,v0,a=p.acquire()
    first=a.weight.clone()
    with torch.no_grad(): m.weight.add_(1)
    assert p.publish(m,1)
    i1,v1,b=p.acquire()
    assert (v0,v1)==(0,1) and i0!=i1
    assert not p.publish(m,2)
    torch.testing.assert_close(a.weight,first)
    p.release(i0)
    assert p.publish(m,2)
    assert p.versions[i1]==1
    p.release(i1)


def make_policy(device="cpu"):
    return TokPolicy(width=32,layers=1,heads=4,items=True,item_tokens=True,global_token=True,
                     block_memory=True,block_config=asdict(small())).to(device)


def test_policy_packed_actions_checkpoint_and_aux_interfaces():
    torch.manual_seed(15)
    rng=np.random.default_rng(15)
    m=make_policy()
    rows=synth_rows(rng,3); rows["first"]=0
    pending=synth_pending(rng,3,m.heads)
    b=to_batch(rows,pending,"cpu")
    z=m.encode_block(b); zp=m.encode_block(b,packed=True)
    assert z.shape==(3,4,16)
    torch.testing.assert_close(z,zp,atol=3e-6,rtol=3e-5)
    ctx={"states":[m.block_memory.empty() for _ in range(3)]}
    out=m(b,need_h=True,need_vg=True,temporal=ctx)
    assert [x.shape for x in out]==[torch.Size([3,20]),torch.Size([3]),torch.Size([3,9]),torch.Size([3,32]),torch.Size([3])]
    assert ctx["o_seq"].shape==(3,16) and ctx["z_out"][0].shape==(3,32)
    actions,lp,v,vg=m.act(b,u=torch.rand(3,20),need_vg=True,
                        temporal={"states":[m.block_memory.empty() for _ in range(3)]})
    assert actions.shape==(3,5) and torch.isfinite(lp).all()
    stream=io.BytesIO()
    torch.save({"config":m.config,"model":m.state_dict()},stream)
    stream.seek(0); ck=torch.load(stream,weights_only=True)
    copy=TokPolicy(**ck["config"]); copy.load_state_dict(ck["model"])
    torch.testing.assert_close(copy.encode_block(b),z)
    with pytest.raises(ValueError,match="explicit episode states"): m(b)


def test_cold_first_record_resets_supplied_history():
    m=make_policy().eval()
    rng=np.random.default_rng(8)
    rows=synth_rows(rng,1); rows["first"]=1
    b=to_batch(rows,synth_pending(rng,1,m.heads),"cpu")
    with torch.no_grad():
        old=m.block_memory(torch.randn(1,5,4,16),[m.block_memory.empty()])["states"][0]
        a=m(b,temporal={"states":[old]})
        b0=m(b,temporal={"states":[m.block_memory.empty()]})
    for x,y in zip(a,b0): torch.testing.assert_close(x,y)


def test_warmstart_never_loads_temporal_or_heads():
    old=TokPolicy(width=32,layers=1,heads=4,items=True,item_tokens=True,global_token=True,temporal=True)
    new=make_policy()
    pi=new.pi.weight.clone()
    with pytest.raises(ValueError,match="fresh random"):
        warm_start_spatial(new,old.state_dict())
    torch.testing.assert_close(new.pi.weight,pi)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_bfloat16_forward_and_gradients():
    m=BlockMemory(12,small()).cuda()
    z=torch.randn(2,7,4,16,device="cuda",requires_grad=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=m(z,[m.empty(),m.empty()])
        loss=out["hidden"].square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum()>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_full_v1_cuda_bank_and_scale_families():
    import time
    c=BlockConfig()
    m=BlockMemory(192,c).cuda()
    state=m.empty()
    blocks=tuple(MemoryBlock((i+1)*32-1,torch.randn(8,256,device="cuda"),0,(0,)*32)
                 for i in range(256))
    prefix=tuple((torch.randn(4,512,64,device="cuda",dtype=torch.bfloat16),
                  torch.randn(4,512,64,device="cuda",dtype=torch.bfloat16)) for _ in range(4))
    state=replace(state,step=8192,blocks=blocks,kv=prefix,
                  kv_steps=torch.arange(8064,8192,device="cuda").repeat_interleave(4),
                  kv_versions=torch.zeros(512,device="cuda",dtype=torch.long))
    z=torch.randn(1,128,4,256,device="cuda",requires_grad=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=m(z,[state]); loss=(out["hidden"]*torch.randn(256,device="cuda")).mean()
    loss.backward(); torch.cuda.synchronize()
    print("V1_FULL_BANK_GPU",{"seconds":round(time.perf_counter()-start,3),
          "peak_allocated_MiB":round(torch.cuda.max_memory_allocated()/2**20,2),
          "chunk_steps":128,"existing_blocks":256,"finite":bool(torch.isfinite(z.grad).all())})
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>1e-5
    assert len(out["states"][0].blocks)==256
    sizes=[sum(p.numel() for p in m.parameters())]
    for width,layers,heads in ((384,6,6),(512,8,8)):
        bigger=BlockMemory(192,BlockConfig(width=width,layers=layers,heads=heads)).cuda()
        with torch.no_grad():
            result=bigger(torch.randn(1,1,4,256,device="cuda"),[bigger.empty()])
        assert result["hidden"].shape==(1,1,width)
        sizes.append(sum(p.numel() for p in bigger.parameters()))
    assert sizes[0]<sizes[1]<sizes[2]
    print("MEMORY_MODULE_PARAMETER_COUNTS",sizes)

