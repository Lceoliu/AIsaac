import sys
root = sys.argv[1] + '/bridge/python'
p = root + '/isaac_bridge/source_pipeline.py'; s = open(p, encoding='utf-8').read()
for a, b in [("""        self.encode_batch = bool(encode_batch) and bool(processes)""",
              """        self.encode_batch = bool(encode_batch) and bool(processes)
        self.encode_requests = 0   # (accounting: encode requests sent to source processes)"""),
             ("""            result=self._remote(worker,'encode',(cid,parts,keys),version)""",
              """            self.encode_requests+=1
            result=self._remote(worker,'encode',(cid,parts,keys),version)"""),
             ("""        results=self._remote(worker,'encode_many',items,version)""",
              """        self.encode_requests+=1
        results=self._remote(worker,'encode_many',items,version)""")]:
    assert s.count(a) == 1, a; s = s.replace(a, b)
open(p, 'w', encoding='utf-8', newline='').write(s)
p = root + '/tests/test_source_vjp_merge.py'; s = open(p, encoding='utf-8').read()
for a, b in [("""        runs[name] = dict(grads=grads,""", """        runs[name] = dict(requests=pipe.encode_requests, grads=grads,"""),
             ("""    assert all(r['dropped'] == 0 for r in runs.values())""",
              """    if processes: assert runs['batch']['requests'] < runs['both']['requests'], (runs['batch']['requests'], runs['both']['requests'])
    if processes: report['encode_requests'] = {k: r['requests'] for k, r in runs.items()}
    assert all(r['dropped'] == 0 for r in runs.values())""")]:
    assert s.count(a) == 1, a; s = s.replace(a, b)
open(p, 'w', encoding='utf-8', newline='').write(s)
print('ok')
