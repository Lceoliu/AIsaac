"""Tensor slot math against the same fixed-weight object reference."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from test_block_memory import small
from isaac_bridge.block_memory import BlockMemory
from isaac_bridge.block_slots import BlockSlotState


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_dense_slot_step_reset_fifo_and_fork_match_reference(device,monkeypatch):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(191)
    m=BlockMemory(12,small()).to(device).eval()
    arena=BlockSlotState(m.config,8,device=device)
    slots=[0,2,1];states=[m.empty(device=device) for _ in slots]
    for step in range(21):
        if step==6:
            arena.reset([2]);states[1]=m.empty(device=device)
        if step==11:
            arena.fork([0],[4]);frozen=states[0].fork();frozen_data=arena.gather(arena._ids([4])[1])
        z=torch.randn(3,4,16,device=device)
        with torch.no_grad():reference=m(z[:,None],states,version=step)
        with monkeypatch.context() as patch:
            patch.setattr(m,'forward',lambda *a,**k:(_ for _ in ()).throw(AssertionError('OBJECT_FORWARD_ON_SLOT_PATH')))
            result=arena.step(m,z,slots,version=step,source_ids=np.arange(3)+step*3)
        torch.testing.assert_close(result['hidden'],reference['hidden'],rtol=2e-5,atol=2e-6)
        for i in range(3):
            keys=len(states[i].blocks)*m.config.memory_tokens
            torch.testing.assert_close(result['memory_mask'][i,:,:keys],reference['memory_mask'][i,:,:keys],rtol=0,atol=0)
            assert not result['memory_mask'][i,:,keys:].any()
            assert not reference['memory_mask'][i,:,keys:].any()
        states=reference['states']
        batch=arena.gather(arena._ids(slots)[1])
        for i,state in enumerate(states):
            assert arena.step_host[slots[i]]==state.step
            n=len(state.kv_steps)
            torch.testing.assert_close(batch['kv_steps'][i,:n],state.kv_steps,rtol=0,atol=0)
            torch.testing.assert_close(batch['kv_versions'][i,:n],state.kv_versions,rtol=0,atol=0)
            for layer in range(m.config.layers):
                for kv in (0,1):torch.testing.assert_close(batch['kv'][layer,kv,i,:,:n],state.kv[layer][kv],rtol=2e-5,atol=2e-6)
            for j,block in enumerate(state.blocks):
                assert batch['ends'][i,j]==block.end_step
                torch.testing.assert_close(batch['bank'][i,j],block.value,rtol=2e-5,atol=2e-6)
                assert batch['bank_source_versions'][i,j].tolist()==list(block.source_versions)
    after=arena.gather(arena._ids([4])[1])
    for key in frozen_data:torch.testing.assert_close(after[key],frozen_data[key],rtol=0,atol=0,msg=key)
    arena.fork([4],[5]);z=torch.randn(1,4,16,device=device)
    with torch.no_grad():reference=m(z[:,None],[frozen],version=77)
    result=arena.step(m,z,[5],version=77)
    torch.testing.assert_close(result['hidden'],reference['hidden'],rtol=2e-5,atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='slot CUDA graph and weight publication')
@pytest.mark.parametrize('bf16',[False,True])
def test_slot_reader_graph_matches_eager_with_published_relative_bias(bf16):
    from isaac_bridge.block_training import ReaderStepGraph
    torch.manual_seed(92)
    memory=BlockMemory(12,small()).cuda().eval()
    graph=ReaderStepGraph(SimpleNamespace(block_memory=memory,parameters=memory.parameters),bf16)
    eager=BlockSlotState(memory.config,4,device='cuda')
    captured=BlockSlotState(memory.config,4,device='cuda')
    for step in range(12):
        slots=[0,1,2] if step%2 else [2,0]
        if step==5:
            eager.reset([1]);captured.reset([1])
            with torch.no_grad():memory.short_relative.normal_(std=.2);memory.memory_age.normal_(std=.2)
        z=torch.randn(len(slots),4,16,device='cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):
            a=eager.step(memory,z,slots,version=step)
            b=captured.step(memory,z,slots,version=step,reader=graph)
            for name in ('hidden','tokens'):
                torch.testing.assert_close(a[name],b[name],rtol=.02 if bf16 else 3e-5,atol=.015625 if bf16 else 3e-6)
            torch.testing.assert_close(a['memory_mask'],b['memory_mask'][...,:a['memory_mask'].shape[-1]],rtol=0,atol=0)
    assert graph.replays==12 and graph.captures==2
