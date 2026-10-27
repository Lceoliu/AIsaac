"""Source-process cProfile hooks (2026-10-27, measurement only): ISAAC_SOURCE_VJP_CPROFILE=n profiles the n-th source
VJP; ISAAC_SOURCE_ENCODE_CPROFILE=first:count the encode requests first .. first+count-1. Each prints one
SOURCE_CPROFILE line: top functions by cumulative and own time, and named functions' cumulative seconds.
usage: python patch_cprofile.py <checkout>"""
import sys
root = sys.argv[1] + '/bridge/python'


def patch(path, pairs):
    s = open(path, encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:80], s.count(a))
        s = s.replace(a, b)
    open(path, 'w', encoding='utf-8', newline='').write(s)


patch(root + '/isaac_bridge/source_worker.py', [
("""def source_worker(connection, model, device, options, vjp_connection=None):""",
 '''# (named functions whose cumulative time the SOURCE_CPROFILE line reports: (file suffix, function name))
CPROFILE_NAMED = [('source_pipeline.py', '_encode'), ('source_replay.py', 'encode_values'), ('source_replay.py', 'backward_replay'),
                  ('source_pipeline.py', '_backward'), ('source_pipeline.py', '_write'), ('tok_policy.py', 'to_batch'),
                  ('tok_policy.py', 'learner_plan'), ('tok_policy.py', 'plan_tokens'), ('tok_policy.py', 'item_count'),
                  ('tok_policy.py', 'encode_rebuild'), ('tok_policy.py', 'write'), ('source_replay.py', '_packed_indices'),
                  ('source_pipeline.py', '_cast_context'), ('source_worker.py', 'ipc_export'), ('rebuild_memory.py', 'write'),
                  ('graph.py', '_engine_run_backward'), ('~', "<method 'to' of 'torch._C.TensorBase' objects>"),
                  ('~', "<built-in method torch.as_tensor>"), ('~', "<built-in method torch.from_numpy>"),
                  ('~', "<method 'copy_' of 'torch._C.TensorBase' objects>"), ('~', "<method 'synchronize' of 'torch._C._CudaEventBase' objects>"),
                  ('~', "<built-in method torch._C._cuda_synchronize>"), ('~', "<built-in method numpy.concatenate>")]


def cprofile_summary(prof, label, frames, device):
    """(measurement) one SOURCE_CPROFILE line from a cProfile.Profile."""
    import json, pstats
    st = pstats.Stats(prof).stats
    name = lambda k: f"{k[0].rsplit('/', 1)[-1].rsplit(chr(92), 1)[-1]}:{k[1]}({k[2]})"
    total = max((v[3] for v in st.values()), default=0.)
    cum = sorted(st.items(), key=lambda kv: -kv[1][3])[:25]
    own = sorted(st.items(), key=lambda kv: -kv[1][2])[:25]
    named = {}
    for f, fn in CPROFILE_NAMED:
        hits = [v for k, v in st.items() if k[2] == fn and (f == '~' or k[0].endswith(f))]
        if hits: named[f'{f}:{fn}'] = dict(calls=sum(h[1] for h in hits), cum_s=max(h[3] for h in hits))
    print('SOURCE_CPROFILE ' + json.dumps(dict(label=label, device=str(device), frames=frames, total_s=total,
          top_cumulative=[[name(k), v[1], round(v[3], 4)] for k, v in cum],
          top_own=[[name(k), v[1], round(v[2], 4)] for k, v in own], named=named)), flush=True)


def source_worker(connection, model, device, options, vjp_connection=None):'''),
("""    profile_at=int(os.environ.get('ISAAC_SOURCE_VJP_PROFILE','0') or 0)""",
 """    profile_at=int(os.environ.get('ISAAC_SOURCE_VJP_PROFILE','0') or 0)
    cprofile_vjp=int(os.environ.get('ISAAC_SOURCE_VJP_CPROFILE','0') or 0)
    cprofile_encode=[int(x) for x in os.environ.get('ISAAC_SOURCE_ENCODE_CPROFILE','0:0').split(':')]
    cprofile_state=dict(requests=0,prof=None,frames=0)"""),
("""            if command in ('encode','encode_many'):""",
 """            if command in ('encode','encode_many'):
                cprofile_state['requests']+=1
                first_c,count_c=cprofile_encode
                if count_c and cprofile_state['requests']==first_c:
                    import cProfile
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    cprofile_state['prof']=cProfile.Profile();cprofile_state['prof'].enable()
                cprofile_state['frames']+=sum(len(k) for _,_,k in ([payload] if command=='encode' else payload)) if cprofile_state['prof'] else 0"""),
("""                else:reply=[encode_one(cid,parts,keys,version) for cid,parts,keys in payload]""",
 """                else:reply=[encode_one(cid,parts,keys,version) for cid,parts,keys in payload]
                if cprofile_state['prof'] is not None and cprofile_state['requests']==first_c+count_c-1:
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    cprofile_state['prof'].disable()
                    cprofile_summary(cprofile_state['prof'],f'encode requests {first_c}..{first_c+count_c-1}',cprofile_state['frames'],device)
                    cprofile_state['prof']=None"""),
("""                else:
                    grads=local._backward(0,decoded,retain)""",
 """                elif retain and cprofile_vjp and vjp_index[0]==cprofile_vjp:
                    import cProfile
                    serve_=local.__dict__.pop('service_forward',None)   # (the VJP alone)
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    prof_c=cProfile.Profile();prof_c.enable()
                    grads=local._backward(0,decoded,retain)
                    if device.type=='cuda':torch.cuda.synchronize(device)
                    prof_c.disable()
                    if serve_ is not None:local.service_forward=serve_
                    replay_=[e for e in local.events if e.get('replay')]
                    cprofile_summary(prof_c,f'vjp {vjp_index[0]}',replay_[-1]['replayed_rows'] if replay_ else
                                     sum(c.get('rows',0) for c,_ in decoded),device)
                else:
                    grads=local._backward(0,decoded,retain)"""),
])
print('ok')
