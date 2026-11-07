"""Window-external history budgets and time-origin-independent Reader semantics."""
from types import SimpleNamespace
from dataclasses import replace
import torch
import pytest
import copy
from test_block_memory import small
from isaac_bridge.block_memory import BlockConfig,BlockMemory,historical_blocks,long_mask


def test_history_budget_uniform_without_replacement_outside_short_window():
    assert BlockConfig().history_recompute_blocks==4
    state=SimpleNamespace(step=1024,blocks=tuple(SimpleNamespace(end_step=n) for n in range(31,1024,32)))
    for budget in (0,4,8,64):
        config=BlockConfig(history_recompute_blocks=budget)
        selected=historical_blocks(state,torch.Generator().manual_seed(82),config=config)
        eligible=[b for b in state.blocks if b.end_step<=state.step-config.window]
        order=torch.randperm(len(eligible),generator=torch.Generator().manual_seed(82))[:budget].tolist()
        assert [b.end_step for b in selected]==[eligible[i].end_step for i in order]
        assert len({b.end_step for b in selected})==min(budget,len(eligible))
    edge=SimpleNamespace(step=256,blocks=(SimpleNamespace(end_step=128),SimpleNamespace(end_step=129)))
    assert [b.end_step for b in historical_blocks(edge,config=BlockConfig())]==[128]
    assert historical_blocks(SimpleNamespace(step=8,blocks=()),config=BlockConfig())==()


def test_reader_and_all_gradients_are_invariant_to_absolute_time_origin():
    torch.manual_seed(7)
    m=BlockMemory(12,small(window=8,block_steps=4,memory_blocks=8))
    with torch.no_grad():m.memory_age.normal_(std=.1);m.short_relative.normal_(std=.1)
    z=torch.randn(2,4,4,16)
    position=torch.arange(128,132)[None].repeat(2,1)
    kp=torch.arange(124,128)[None].repeat_interleave(4,1).repeat(2,1)
    ends=torch.tensor([[95,95,111,111,123,123,-1,-1]]).repeat(2,1)
    bank=torch.randn(2,8,16)
    prefixes=tuple(tuple(torch.randn(2,2,16,8) for _ in (0,1)) for _ in range(2))
    expected=None
    for shift in (0,1,17,32,10**8):
        m.zero_grad(set_to_none=True)
        out=m.read(z,position+shift,prefixes,bank,torch.where(ends>=0,ends+shift,ends),kp+shift)
        out[0].square().mean().backward()
        grads={n:None if p.grad is None else p.grad.clone() for n,p in m.named_parameters()}
        if expected is None:expected=(out,grads)
        else:
            for i in (0,1,3,4):torch.testing.assert_close(out[i],expected[0][i],atol=0,rtol=0)
            for a,b in zip(out[2],expected[0][2]):
                for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=0,rtol=0)
            for n,g in grads.items():
                assert (g is None)==(expected[1][n] is None),n
                if g is not None:torch.testing.assert_close(g,expected[1][n],atol=0,rtol=0,msg=n)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Flex and BF16 relative-bias GPU gradients')
@pytest.mark.parametrize('amp',[False,True])
def test_flex_matches_sdpa_relative_bias_gradients_and_shift(amp,record_property):
    torch.manual_seed(824)
    cfg=small(width=32,window=8,block_steps=4,memory_blocks=8,attention='sdpa')
    reference=BlockMemory(12,cfg).cuda()
    with torch.no_grad():reference.short_relative.normal_(std=.1);reference.memory_age.normal_(std=.1)
    flex=copy.deepcopy(reference);flex.config=replace(cfg,attention='flex')
    z=torch.randn(2,8,4,16,device='cuda')
    position=torch.arange(128,136,device='cuda')[None].repeat(2,1)
    kp=torch.arange(120,128,device='cuda')[None].repeat_interleave(4,1).repeat(2,1)
    kp[1,:8]=-1
    ends=torch.tensor([[103,103,111,111,123,123,-1,-1],[99,99,115,115,127,127,-1,-1]],device='cuda')
    bank=torch.randn(2,8,16,device='cuda')
    prefixes=tuple(tuple(torch.randn(2,2,32,16,device='cuda') for _ in (0,1)) for _ in range(2))
    expected=None;shift_reference={}
    for model,shift in ((m,s) for m in (reference,flex) for s in (0,0,1,17,10**8)):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
            out=model.read(z,position+shift,prefixes,bank,torch.where(ends>=0,ends+shift,ends),
                           torch.where(kp>=0,kp+shift,kp))
            loss=out[0].float().square().mean()
        loss.backward()
        grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        assert grads['short_relative'].abs().sum()>0 and grads['memory_age'].abs().sum()>0
        if expected is None:expected=(out,grads)
        elif model is flex and shift==0:
            # Backend BF16 roundoff is distinct from exact origin invariance.
            record_property('hidden_max_abs_error',float((out[0]-expected[0][0]).abs().max()))
            record_property('all_gradient_max_abs_error',max(float((g-expected[1][n]).abs().max()) for n,g in grads.items()))
            for i in (0,1):torch.testing.assert_close(out[i],expected[0][i],rtol=.03 if amp else 3e-4,atol=.015625 if amp else 3e-6)
            for i in (3,4):torch.testing.assert_close(out[i],expected[0][i],rtol=0,atol=0)
            for a,b in zip(out[2],expected[0][2]):
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=.03 if amp else 3e-4,atol=.015625 if amp else 3e-6)
            assert grads.keys()==expected[1].keys()
            for n,g in grads.items():torch.testing.assert_close(g,expected[1][n],rtol=.04 if amp else 5e-4,atol=.0003 if amp else 3e-6,msg=n)
        if model not in shift_reference:shift_reference[model]=(out,grads)
        else:
            original=shift_reference[model]
            for i in (0,1,3,4):torch.testing.assert_close(out[i],original[0][i],rtol=0,atol=0)
            for a,b in zip(out[2],original[0][2]):
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            for n,g in grads.items():
                # Flex accumulates the captured age-table gradient atomically:
                # identical-input repeats already differ by ~4.5e-13 on 4090.
                atomic=model is flex and n=='memory_age'
                if atomic:record_property(f'flex_age_gradient_error_shift_{shift}',float((g-original[1][n]).abs().max()))
                torch.testing.assert_close(g,original[1][n],rtol=1e-5 if atomic else 0,atol=1e-10 if atomic else 0,
                    msg=lambda message:f'{model.config.attention} shift={shift} {n}: {message}')
