"""Native conditional one-tick law checks, plus open-loop projectile tracks.

This checks measured states, not global RNG identity. Position/height tolerance
is 0.001 game units. Boss contact and grid clipping are excluded from motion laws.
"""
import argparse,collections,gzip,json,math,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('trace',type=Path);p.add_argument('--exe',type=Path,required=True)
p.add_argument('--report',type=Path,required=True);p.add_argument('--save-fixture',type=Path)
a=p.parse_args()
op=gzip.open if a.trace.suffix=='.gz' else open
with op(a.trace,'rt',encoding='utf8') as f: rows=[json.loads(l) for l in f]
if 'info' in rows[0]:
 rows=[dict(case=r['case'],tick=r['tick'],player=r['obs']['players'][0]['pos'],**r['info']['monstro']) for r in rows]
if a.save_fixture:
 with gzip.open(a.save_fixture,'wt',encoding='utf8') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
proc=subprocess.Popen([str(a.exe.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf8')
def cmd(s):
 proc.stdin.write(s+'\n');proc.stdin.flush();return json.loads(proc.stdout.readline())
stats=collections.Counter();errors=[];maxima=collections.defaultdict(float)
for old,new in zip(rows,rows[1:]):
 if old['case']!=new['case'] or not old['bosses'] or not new['bosses']:continue
 b,c=old['bosses'][0],new['bosses'][0]
 if b['target']==[0,0] and c['target']!=[0,0] and math.dist(c['pos'],new['player'])>65:
  expected=[int((new['player'][0]-40)/40+0.5)*40+40,int((new['player'][1]-120)/40+0.5)*40+120]
  assert c['target']==expected,(c,expected)
  stats['target_locks']+=1
 if b['state']==c['state']==7 and b['target']!=[0,0] and c['frame']>b['frame']:
  assert c['target']==b['target'],(b,c)
  stats['locked_target_frames']+=1
 if c['state']==4 and 2<=c['frame']<=43:
  assert c['collision']==(0 if 7<=c['frame']<23 else 4),c
  stats['hop_collision_frames']+=1
 if c['state']==7 and 2<=c['frame']<=62:
  assert c['collision']==(0 if c['frame']<33 else 4),c
  stats['high_jump_collision_frames']+=1
 born=[q for q in new['projectiles'] if q['age']==0]
 if born:
  expected_count=13 if c['state']==8 and c['frame']==22 else 18 if c['state']==7 and c['frame']==35 else 0
  assert len(born)==expected_count,(c,len(born))
  stats['native_volleys']+=1
  for q in born:
   assert q['height']==-23 and abs(q['accel']-0.32)<1e-6 and -19<=q['fall']<=5,q
   if c['state']==7:assert abs(math.hypot(*q['vel'])-7)<1e-5,q
   elif math.dist(q['pos'],new['player'])>65:
    d=[new['player'][j]-q['pos'][j] for j in range(2)];length=math.hypot(*d)
    assert math.hypot(*(q['vel'][j]-7*d[j]/length for j in range(2)))<=3.5001,q
   stats['native_projectile_births']+=1
 if b['state'] not in (4,6,7,8) or b['state']!=c['state'] or b['frame']>=43:continue
 if b['state']!=8 and (b['target']==[0,0] or c['target']!=b['target']):continue
 if any(math.dist(q['pos'],new['player'])<65 or not (110<q['pos'][0]<530 and 190<q['pos'][1]<360) for q in (b,c)):continue
 values=[b['state'],b['frame'],*b['pos'],*b['vel'],*b['target'],*new['player']]
 cmd('boss '+' '.join(map(str,values))); got=cmd('tick')['bosses'][0]
 err=max(abs(x-y) for x,y in zip(got['pos']+got['vel'],c['pos']+c['vel']))
 maxima['boss_motion']=max(maxima['boss_motion'],err);stats['boss_motion']+=1
 if err>0.001 and len(errors)<12:errors.append(dict(case=old['case'],tick=old['tick'],state=b['state'],frame=b['frame'],error=err,expected=c['pos'],got=got['pos']))
tracks=collections.defaultdict(list)
for r in rows:
 for q in r['projectiles']:tracks[r['case'],q['id']].append(q)
for tr in tracks.values():
 if len(tr)<2 or tr[0]['age']!=0:continue
 b=tr[0]
 cmd('projectile '+' '.join(map(str,[*b['pos'],*b['vel'],b['height'],b['fall'],b['accel'],0])))
 for c in tr[1:]:
  # Only interior live segments: wall/contact death tested independently.
  if c['dead'] or not (100<c['pos'][0]<540 and 180<c['pos'][1]<380):break
  got=cmd('tick')['projectiles']
  if not got:
   errors.append({'missing_projectile':c});break
  g=got[0];err=max(abs(x-y) for x,y in zip(g['pos']+[g['height'],g['fall']],c['pos']+[c['height'],c['fall']]))
  maxima['projectile_track']=max(maxima['projectile_track'],err);stats['projectile_points']+=1
  if err>0.001 and len(errors)<12: errors.append(dict(projectile=c['id'],age=c['age'],error=err))
collision_path=Path(__file__).resolve().parents[1]/'tests/fixtures/j460_monstro_collision.json'
collision=json.loads(collision_path.read_text(encoding='utf8'))
for case,height,pos in [('low',-30,[320,380]),('overhead',-70,[320,380]),('floor',-5,[200,280])]:
 cmd('projectile '+' '.join(map(str,[*pos,0,0,height,0,0,0,320,380])))
 for expected in (r for r in collision if r['case']==case):
  got=cmd('tick')
  assert got['hp']==expected['hearts'],(case,got,expected)
  assert len(got['projectiles'])==len(expected['projectiles']),(case,got,expected)
  for g,c in zip(got['projectiles'],expected['projectiles']):
   assert g['dead']==c['dead'] and abs(g['height']-c['height'])<0.001,(g,c)
  stats['native_collision_frames']+=1
proc.stdin.close();code=proc.wait();proc.stdout.close()
report=dict(counts=dict(stats),max_error=dict(maxima),errors=errors,engine_exit=code)
a.report.write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
assert code==0 and not errors,report
