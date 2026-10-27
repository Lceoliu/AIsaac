"""--source-vjp-merge N: pack episode groups into calls of about N frames (1: all in one) (2026-10-27).
usage: python patch_vjp_pack.py <checkout>"""
import sys
root = sys.argv[1] + '/bridge/python'


def patch(path, pairs):
    s = open(path, encoding='utf-8').read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:80], s.count(a))
        s = s.replace(a, b)
    open(path, 'w', encoding='utf-8', newline='').write(s)


patch(root + '/isaac_bridge/source_pipeline.py', [
("""        # vjp_merge: one autograd.backward per card and VJP over every episode group (default: one per episode);""",
 """        # vjp_merge: 1 = one autograd.backward per card and VJP over every episode group, N > 1 = episode groups
        # packed into calls of at least N encoded frames (between calls the VJP still serves waiting encodes);
        # 0 (default) = one call per episode group;"""),
("""        self.vjp_merge = bool(vjp_merge)""",
 """        self.vjp_merge = int(vjp_merge or 0)"""),
("""                    owner=chunk.get('uid') if self.ipc_async else None
                    outputs,gradients=groups.setdefault(owner,([],[]))""",
 """                    owner=chunk.get('uid') if self.ipc_async else None
                    outputs,gradients=groups.setdefault(owner,([],[]))
                    frames[owner]=frames.get(owner,0)+chunk.get('rows',0)"""),
("""            groups={}
            for chunk, grads in packets:""",
 """            groups={};frames={}
            for chunk, grads in packets:"""),
("""            if self.vjp_merge and len(groups)>1:   # (one engine call over every episode group of this card)
                groups={None:([o for os_,_ in groups.values() for o in os_],[g for _,gs in groups.values() for g in gs])}""",
 """            if self.vjp_merge and len(groups)>1:   # (fewer engine calls: whole episode groups packed together)
                limit=float('inf') if self.vjp_merge==1 else self.vjp_merge
                packed,cur,n=[],([],[]),0
                for owner,(os_,gs) in groups.items():
                    cur[0].extend(os_);cur[1].extend(gs);n+=frames.get(owner,0)
                    if n>=limit:packed.append(cur);cur,n=([],[]),0
                if cur[0]:packed.append(cur)
                groups=dict(enumerate(packed))"""),
])
patch(root + '/train_tok.py', [
("""vjp_merge=bool(args.source_vjp_merge),""", """vjp_merge=args.source_vjp_merge,"""),
("""    p.add_argument('--source-vjp-merge', type=int, choices=(0, 1), default=0,
                   help='1: one autograd.backward per source card and VJP over all its episode groups (0: one per '
                        'episode); the same gradients up to float summation order')""",
 """    p.add_argument('--source-vjp-merge', type=int, default=0,
                   help='1: one autograd.backward per source card and VJP over all its episode groups; N > 1: episode '
                        'groups packed into calls of at least N encoded frames (waiting encodes are served between '
                        'calls); 0: one call per episode. The same gradients up to float summation order')"""),
])
print('ok')
