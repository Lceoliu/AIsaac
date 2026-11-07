"""PyTorch 2.7+ FlexAttention long-memory backend (CUDA learner chunks).

The learnable age table is indexed once in score_mod, so its captured-tensor
gradient is preserved. No B x H x Q x M age-bias tensor is materialised.
Actor single-step calls keep SDPA: compiling masks for four queries costs more
than the small dense attention. Backend errors propagate rather than falling back.
"""
import torch
from torch.nn.attention.flex_attention import create_block_mask,flex_attention

_compiled_flex=None


def plan(allowed,query_steps,end_steps,age_weights,block_steps):
    b,qn,kn=allowed.shape

    def mask(batch,head,q,k):
        return (q<qn)&(k<kn)&allowed[batch,q.clamp(max=qn-1),k.clamp(max=kn-1)]

    def score(value,batch,head,q,k):
        qs=query_steps[batch,q.clamp(max=qn-1)]
        end=end_steps[batch,k.clamp(max=kn-1)]
        age=((qs-end-1)//block_steps).clamp(0,age_weights.shape[1]-1)
        return value+age_weights[head,age]

    block_mask=create_block_mask(mask,B=b,H=None,Q_LEN=qn,KV_LEN=kn,device=str(allowed.device),
                                 BLOCK_SIZE=128,_compile=True)
    return block_mask,score


def attend(q,k,v,prepared):
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex=torch.compile(flex_attention,dynamic=False)
    block_mask,score=prepared
    return _compiled_flex(q,k,v,block_mask=block_mask,score_mod=score)
