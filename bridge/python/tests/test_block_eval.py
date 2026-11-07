"""The real eval entrypoint must preserve the block checkpoint and worker state."""
import numpy as np
import torch

from eval_tok import build_eval_model,policy_actions
from test_block_training import history_actor
from test_visible_input import synth_rows,synth_pending


def test_eval_checkpoint_and_reordered_workers_match_actor():
    model,actor,rng=history_actor(steps=0)
    evaluated,changed=build_eval_model(dict(config=model.config,model=model.state_dict()),True,'cpu')
    assert not changed
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,evaluated.state_dict()[name],atol=0,rtol=0)
    states=[evaluated.block_memory.empty() for _ in range(2)]
    for step in range(13):
        rows=synth_rows(rng,2);rows['first']=int(step==0)
        rows['episode']=1;rows['done']=0
        if step>=7:
            rows['episode'][0]=2;rows['first'][0]=int(step==7)
        pending=synth_pending(rng,2,model.heads)
        expected=actor.step(rows,pending,[0,1],greedy=True)[0].numpy()
        got=policy_actions(evaluated,rows[::-1].copy(),pending[::-1].copy(),[1,0],'cpu',True,block_state=states)
        np.testing.assert_array_equal(got[::-1],expected)
        for i,state in enumerate(states):
            assert state.step==actor.memory[i].step
            for (ak,av),(bk,bv) in zip(state.kv,actor.memory[i].kv):
                torch.testing.assert_close(ak,bk,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(av,bv,atol=2e-6,rtol=2e-5)


def test_eval_rejects_missing_block_context():
    import pytest
    model,actor,rng=history_actor(steps=0)
    with pytest.raises(ValueError,match='explicit episode states'):
        policy_actions(model,synth_rows(rng,1),synth_pending(rng,1,model.heads),[0],'cpu',True)
