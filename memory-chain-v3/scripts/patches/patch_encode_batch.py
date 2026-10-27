"""--source-encode-batch 1: one RPC per source process and schedule() call for all its encodes (2026-10-27).
usage: python patch_encode_batch.py <checkout>"""
import sys
root = sys.argv[1] + '/bridge/python'

p = root + '/isaac_bridge/source_worker.py'
s = open(p, encoding='utf-8').read()
start = s.index("            if command=='encode':\n")
end = s.index("            elif command=='write':\n")
body = s[start:end].split('\n')[1:]   # (without the 'if' line)
assert body[0].startswith("                cid,parts,keys=payload"), body[0]
body = body[1:]   # (cid, parts, keys become the function's parameters)
func = ["    def encode_one(cid,parts,keys,version):",
        "        # (one encode: its graph registered as chunk cid, its compact IPC export returned)"]
for line in body:
    if not line.strip(): continue
    assert line.startswith(' ' * 16), repr(line)
    func.append(line[8:])
func.append("        return reply")
s = s[:start] + """            if command in ('encode','encode_many'):
                # ('encode_many', --source-encode-batch: the encodes of one schedule() call in one message)
                if command=='encode':reply=encode_one(*payload,version)
                else:reply=[encode_one(cid,parts,keys,version) for cid,parts,keys in payload]
""" + s[end:]
anchor = "    def dispatch(connection,message):\n"
assert s.count(anchor) == 1
s = s.replace(anchor, '\n'.join(func) + '\n' + anchor)
a = "            if message[0] not in ('encode','write'):"
assert s.count(a) == 1
s = s.replace(a, "            if message[0] not in ('encode','encode_many','write'):")
open(p, 'w', encoding='utf-8', newline='').write(s)


def patch(path, pairs):
    t = open(path, encoding='utf-8').read()
    for a, b in pairs:
        assert t.count(a) == 1, (path, a[:80], t.count(a))
        t = t.replace(a, b)
    open(path, 'w', encoding='utf-8', newline='').write(t)


patch(root + '/isaac_bridge/source_pipeline.py', [
("""checkpoint_blocks=None, streams=0, vjp_merge=False, fused_cast=False):""",
 """checkpoint_blocks=None, streams=0, vjp_merge=False, fused_cast=False, encode_batch=False):"""),
("""        self.fused_cast = bool(fused_cast)""",
 """        self.fused_cast = bool(fused_cast)
        # encode_batch (source processes): the encodes one schedule() call gives a process go in one RPC, not one
        # request and reply each (the same encodes, in the same order)
        self.encode_batch = bool(encode_batch) and bool(processes)"""),
("""        self.replicas
        for uid,worker,indices in groups:""",
 """        self.replicas
        batches={}   # (encode_batch: worker -> [(cid, parts, keys, the chunk's future)])
        for uid,worker,indices in groups:"""),
("""                future = self.workers[worker].submit(self._encode, worker, selected, self.version,cid,[keys[i] for i in ix])""",
 """                if self.encode_batch:
                    from concurrent.futures import Future
                    future=Future();batches.setdefault(worker,[]).append((cid,selected,[keys[i] for i in ix],future))
                else:
                    future = self.workers[worker].submit(self._encode, worker, selected, self.version,cid,[keys[i] for i in ix])"""),
("""                self.encoded_rows += len(ix)
        return True""",
 """                self.encoded_rows += len(ix)
        for worker,items in batches.items():
            batch=self.workers[worker].submit(self._encode_many,worker,[(c,p,k) for c,p,k,_ in items],self.version)
            def settle(done,futures=[f for *_,f in items]):
                try:
                    for f,r in zip(futures,done.result()):f.set_result(r)
                except BaseException as error:
                    for f in futures:
                        if not f.done():f.set_exception(error)
            batch.add_done_callback(settle)
        return True

    def _encode_many(self, worker, items, version):
        results=self._remote(worker,'encode_many',items,version)
        for result in results:result['grads']={}
        return results"""),
("""                         vjp_merge=vjp_merge,fused_cast=fused_cast)""",
 """                         vjp_merge=vjp_merge,fused_cast=fused_cast,encode_batch=False)"""),
])

patch(root + '/train_tok.py', [
("""vjp_merge=args.source_vjp_merge, fused_cast=bool(args.source_fused_cast),""",
 """vjp_merge=args.source_vjp_merge, fused_cast=bool(args.source_fused_cast),
            encode_batch=bool(args.source_encode_batch),"""),
("""    p.add_argument('--source-fused-cast', type=int, choices=(0, 1), default=0,""",
 """    p.add_argument('--source-encode-batch', type=int, choices=(0, 1), default=0,
                   help='1 (source processes): the encodes of one scheduling call go to each source process in one '
                        'request (0: one request and reply per encode); the same encodes')
    p.add_argument('--source-fused-cast', type=int, choices=(0, 1), default=0,"""),
])
print('ok')
