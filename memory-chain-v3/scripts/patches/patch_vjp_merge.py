"""--source-vjp-merge / --source-fused-cast / ISAAC_SOURCE_ENCODE_PROFILE (2026-10-27). usage: python patch_vjp_merge.py <checkout>"""
import sys
root = sys.argv[1] + '/bridge/python'


def patch(path, pairs):
    s = open(path, encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:80], s.count(a))
        s = s.replace(a, b)
    open(path, 'w', encoding='utf-8', newline='').write(s)


patch(root + '/isaac_bridge/source_pipeline.py', [
("""class SourcePipeline:
    def __init__(""",
 '''class ImmutableSourceCastFused(torch.autograd.Function):
    """(--source-fused-cast) ImmutableSourceCast for all of an encode's parameters in one node.

    The same per-forward FP32 edges (each parameter's BF16 gradient converted to its FP32 snapshot before it is
    accumulated), but one backward node and one flat conversion per encode instead of one node per parameter.
    Outputs that receive no gradient stay None (no zero tensors)."""
    @staticmethod
    def forward(ctx, n, *tensors):
        ctx.n = n
        ctx.dtypes = [t.dtype for t in tensors[:n]]
        ctx.set_materialize_grads(False)
        return tuple(tensors[n:])

    @staticmethod
    def backward(ctx, *grads):
        out = [None] * ctx.n
        present = [i for i, g in enumerate(grads) if g is not None]
        by_dtype = {}
        for i in present: by_dtype.setdefault(ctx.dtypes[i], []).append(i)
        for dtype, index in by_dtype.items():
            flat = torch.cat([grads[i].reshape(-1) for i in index]).to(dtype)
            for i, part in zip(index, flat.split([grads[i].numel() for i in index])):
                out[i] = part.view(grads[i].shape)
        return (None, *out, *([None] * ctx.n))


class SourcePipeline:
    def __init__('''),
("""balance=False, checkpoint_blocks=None, streams=0):""",
 """balance=False, checkpoint_blocks=None, streams=0, vjp_merge=False, fused_cast=False):"""),
("""        self.streams = int(streams or 0)""",
 """        self.streams = int(streams or 0)
        # (launch overhead, 2026-10-27; both opt-in, same gradients up to float summation order)
        # vjp_merge: one autograd.backward per card and VJP over every episode group (default: one per episode);
        # fused_cast: one cast node per encode for all its BF16 parameter aliases (default: one per parameter)
        self.vjp_merge = bool(vjp_merge)
        self.fused_cast = bool(fused_cast)"""),
("""                         layer_checkpoint=layer_checkpoint,checkpoint_blocks=checkpoint_blocks,streams=streams)""",
 """                         layer_checkpoint=layer_checkpoint,checkpoint_blocks=checkpoint_blocks,streams=streams,
                         vjp_merge=vjp_merge,fused_cast=fused_cast)"""),
("""        params=dict(replica.named_parameters())
        return _reparametrize_module(replica,{n:ImmutableSourceCast.apply(params[n],value)
            for n,value in replica._source_cast_values.items()})""",
 """        params=dict(replica.named_parameters())
        if self.fused_cast and replica._source_cast_values:
            names=list(replica._source_cast_values)
            casts=ImmutableSourceCastFused.apply(len(names),*[params[n] for n in names],
                                                 *[replica._source_cast_values[n] for n in names])
            return _reparametrize_module(replica,dict(zip(names,casts)))
        return _reparametrize_module(replica,{n:ImmutableSourceCast.apply(params[n],value)
            for n,value in replica._source_cast_values.items()})"""),
("""            hook=getattr(self,'vjp_group_hook',None)   # (test aid: a shuffled schedule between episode groups)
            for outputs,gradients in groups.values():""",
 """            hook=getattr(self,'vjp_group_hook',None)   # (test aid: a shuffled schedule between episode groups)
            if self.vjp_merge and len(groups)>1:   # (one engine call over every episode group of this card)
                groups={None:([o for os_,_ in groups.values() for o in os_],[g for _,gs in groups.values() for g in gs])}
            for outputs,gradients in groups.values():"""),
])

patch(root + '/isaac_bridge/source_worker.py', [
("""    profile_at=int(os.environ.get('ISAAC_SOURCE_VJP_PROFILE','0') or 0)""",
 """    profile_at=int(os.environ.get('ISAAC_SOURCE_VJP_PROFILE','0') or 0)
    # (ISAAC_SOURCE_ENCODE_PROFILE=first:count, measurement only: the encodes first .. first+count-1 under one
    # profiler, summarised like a VJP: kernels, idle share, autograd-free)
    encode_profile=[int(x) for x in os.environ.get('ISAAC_SOURCE_ENCODE_PROFILE','0:0').split(':')]
    encode_index=[0];encode_prof=[None,[],0.]"""),
("""                with new_work():
                    if shuffle is not None:shuffle()
                    result=local._encode(0,parts,version,keys=keys)""",
 """                encode_index[0]+=1
                first_,count_=encode_profile
                if count_ and encode_index[0]==first_ and device.type=='cuda':
                    import time as time_
                    from torch.profiler import profile as profile_, ProfilerActivity
                    torch.cuda.synchronize(device)
                    encode_prof[0]=profile_(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]);encode_prof[0].__enter__()
                    encode_prof[2]=time_.perf_counter()
                with new_work():
                    if shuffle is not None:shuffle()
                    result=local._encode(0,parts,version,keys=keys)"""),
("""                for offset,key in enumerate(keys): local.frames[key]=(cid,offset)
                exports['encode',cid]=reply""",
 """                for offset,key in enumerate(keys): local.frames[key]=(cid,offset)
                exports['encode',cid]=reply
                if encode_prof[0] is not None:
                    encode_prof[1].append((dict(rows=len(keys),uid=local.chunks[cid]['uid']),{}))
                    if encode_index[0]==first_+count_-1:
                        import time as time_
                        torch.cuda.synchronize(device);encode_prof[0].__exit__(None,None,None)
                        vjp_profile_summary(encode_prof[0],encode_prof[1],time_.perf_counter()-encode_prof[2],device,
                                            f'encode {first_}..{first_+count_-1}',tag='SOURCE_ENCODE_PROFILE')
                        encode_prof[0]=None"""),
("""def vjp_profile_summary(prof, decoded, wall_s, device, index):""",
 """def vjp_profile_summary(prof, decoded, wall_s, device, index, tag='SOURCE_VJP_PROFILE'):"""),
("""    print('SOURCE_VJP_PROFILE ' + json.dumps(out), flush=True)""",
 """    print(tag + ' ' + json.dumps(out), flush=True)"""),
])

patch(root + '/train_tok.py', [
("""            streams=args.source_streams,""",
 """            streams=args.source_streams, vjp_merge=bool(args.source_vjp_merge), fused_cast=bool(args.source_fused_cast),"""),
("""    p.add_argument('--source-streams', type=int, default=0,""",
 """    p.add_argument('--source-vjp-merge', type=int, choices=(0, 1), default=0,
                   help='1: one autograd.backward per source card and VJP over all its episode groups (0: one per '
                        'episode); the same gradients up to float summation order')
    p.add_argument('--source-fused-cast', type=int, choices=(0, 1), default=0,
                   help='1: one BF16-cast autograd node per encode for all its parameters (0: one per parameter); '
                        'the same per-forward FP32 gradient conversion')
    p.add_argument('--source-streams', type=int, default=0,"""),
])
print('ok')
