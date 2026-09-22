"""Native bomb trajectory/fuse/boundary regression; no engine launch."""
import argparse,gzip,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
rows=json.load(gzip.open(a.fixture,'rt',encoding='utf8'));checks=0;maximum=0.;cases=[]
for case in ('stationary','moving','held','free_velocity'):
 selected=[r for r in rows if r['case']==case]
 prefix=['reset 320 280 0']+(['3 0 0']*12 if case=='moving' else [])
 if case=='free_velocity':prefix=['bomb 250 280 5 0 45 100 380']
 inputs=prefix+[' '.join(map(str,r['action'][:3])) for r in selected]
 proc=subprocess.run([str(a.exe.resolve())],input='\n'.join(inputs)+'\n',capture_output=True,text=True,check=True)
 actual=[json.loads(s) for s in proc.stdout.splitlines()][len(prefix):]
 for ref,got in zip(selected,actual):
  assert len(ref['bombs'])==len(got['bombs']),(case,ref['tick'],'count')
  assert got['hp']==ref['hp'],(case,ref['tick'],'hp',got['hp'],ref['hp'])
  for expected,b in zip(ref['bombs'],got['bombs']):
   for key in ('pos','vel'):
    error=max(abs(x-y) for x,y in zip(expected[key],b[key]));maximum=max(maximum,error)
    assert error<0.001,(case,ref['tick'],key,error)
   assert expected['dead']==b['dead'],(case,ref['tick'],'explosion')
   checks+=1
 cases.append(case)
for ref in rows:
 if not ref['case'].startswith('player_radius_') or ref['tick']!=4:continue
 distance=int(ref['case'].rsplit('_',1)[1])
 proc=subprocess.run([str(a.exe.resolve())],input=f'bomb 320 280 0 0 1 {320+distance} 280\n'+'tick\n'*5,capture_output=True,text=True,check=True)
 assert json.loads(proc.stdout.splitlines()[-1])['hp']==ref['hp'];checks+=1
report=dict(passed=True,checks=checks,cases=cases,max_position_velocity_error=maximum,scope='ordinary bomb free motion, birth, fuse, self damage and strict player radius; not full contact/terrain equality')
a.report.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
