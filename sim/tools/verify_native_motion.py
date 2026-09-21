"""Independent native-oracle replay. No model parameters are fit by this tool.

Player/birth velocities replay open-loop from inputs. Random birth position/fall
are checked against their native bounds; subsequent tear tracks replay from the
observed birth state, without any further corrections. Also checks removal time.
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import subprocess


def read_records(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f]
    if 'obs' not in rows[0]:
        return rows
    return [dict(case=r['case'], initial=r.get('initial', False), action=r.get('action'),
                 player={**{k:r['obs']['players'][0][k] for k in ('pos','vel')},
                         'fire_delay':r['info']['players'][0]['fire_delay']},
                 falling_stat=r['info']['calibration']['tear_falling_speed'],
                 tears=[{k:t[k] for k in ('id','age','pos','vel','height','fall','dead')}
                        for t in r['info']['calibration']['tears']]) for r in rows]


def replay(exe, commands):
    result = subprocess.run([str(exe)], input='\n'.join(commands)+'\n',
                            text=True, capture_output=True, check=True)
    assert not result.stderr, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == len(commands)
    return rows


def verify(exe, records):
    commands=[]
    for r in records:
        p=r['player']
        commands.append(f"reset {p['pos'][0]} {p['pos'][1]} {p['fire_delay']}" if r['initial']
                        else f"{r['action'][0]} {r['action'][1]}")
    outputs=replay(exe,commands)
    errors=dict(player_pos=0., player_vel=0., fire_delay=0., birth_velocity=0.,
                birth_longitudinal=0., tear_pos=0., tear_vel=0., tear_height=0., tear_fall=0.)
    counts=dict(frames=0, shots=0, tear_samples=0, birth_count_mismatches=0,
                death_mismatches=0, removal_mismatches=0)
    failures=[]
    tracks=[]; active={}; previous_sim=set(); eye_draws=[]; falling_draws=[]
    case_stats={}
    for r,s in zip(records,outputs):
        if r['initial']:
            active={}; previous_sim=set(); shot_number=0
            continue
        counts['frames']+=1
        pe=math.dist(r['player']['pos'],s['player']['pos'])
        case_stats[r['case']]=max(case_stats.get(r['case'],0.),pe)
        errors['player_pos']=max(errors['player_pos'],pe)
        errors['player_vel']=max(errors['player_vel'],math.dist(r['player']['vel'],s['player']['vel']))
        errors['fire_delay']=max(errors['fire_delay'],abs(r['player']['fire_delay']-s['player']['fire_delay']))
        born=[t for t in r['tears'] if t['age']==0]
        sborn=[t for t in s['tears'] if t['id'] not in previous_sim]
        previous_sim={t['id'] for t in s['tears']}
        counts['birth_count_mismatches']+=int(len(born)!=len(sborn))
        for t,st in zip(born,sborn):
            counts['shots']+=1
            v=t['vel']; length=math.hypot(*v)
            errors['birth_velocity']=max(errors['birth_velocity'],math.dist(v,st['vel']))
            d=[t['pos'][i]-st['pos'][i] for i in (0,1)]
            errors['birth_longitudinal']=max(errors['birth_longitudinal'],abs(sum(d[i]*v[i] for i in (0,1)))/length)
            if math.hypot(*d)>length*0.2001:
                failures.append(('eye_position_bound',r['case'],t['id']))
            u=((t['fall']-0.1)/0.9+r['falling_stat'])/0.2
            falling_draws.append(u)
            if not -0.00001<=u<=1.00001: failures.append(('fall_draw_bound',r['case'],u))
            # Unambiguous stationary spawn origin, away from collision correction.
            if math.hypot(*r['player']['vel'])<1e-8:
                delta=[t['pos'][i]-r['player']['pos'][i]-v[i] for i in (0,1)]
                eye=(delta[1]*v[0]-delta[0]*v[1])/(length*length)
                sign=-1 if shot_number%2==0 else 1
                eye_draws.append(abs(eye))
                if not 0.29999<=sign*eye<=0.50001:
                    failures.append(('eye_alternation_or_bound',r['case'],eye))
            shot_number+=1
        present={t['id'] for t in r['tears']}
        for tid in list(active):
            if tid not in present:
                tracks[active.pop(tid)]['removed']=True
        for t in r['tears']:
            if t['age']==0:
                tracks.append(dict(case=r['case'],samples=[],removed=False))
                active[t['id']]=len(tracks)-1
            tracks[active[t['id']]]['samples'].append(t)
    commands=[]; expected=[]
    for track in tracks:
        samples=track['samples']; t=samples[0]
        commands.append('tear '+' '.join(map(str,[*t['pos'],*t['vel'],t['height'],t['fall']])))
        expected.append(t)
        for t in samples[1:]: commands.append('tick'); expected.append(t)
        if track['removed']: commands.append('tick'); expected.append(None)
    outputs=replay(exe,commands)
    for t,s in zip(expected,outputs):
        if t is None:
            counts['removal_mismatches']+=bool(s['tears'])
            continue
        counts['tear_samples']+=1
        if not s['tears']:
            counts['removal_mismatches']+=1
            continue
        st=s['tears'][0]
        counts['death_mismatches']+=t['dead']!=st['dead']
        for k in ('pos','vel','height','fall'):
            d=math.dist(t[k],st[k]) if k in ('pos','vel') else abs(t[k]-st[k])
            errors['tear_'+k]=max(errors['tear_'+k],d)
    thresholds={k:(0.001 if 'pos' in k or 'longitudinal' in k else 0.00001) for k in errors}
    passed=all(errors[k]<=thresholds[k] for k in errors) and not failures and not any(v for k,v in counts.items() if 'mismatch' in k)
    def stats(xs):
        return dict(n=len(xs),minimum=min(xs),maximum=max(xs),mean=sum(xs)/len(xs))
    return dict(passed=passed,counts=counts,max_errors=errors,thresholds=thresholds,
                native_eye_offset_fraction=stats(eye_draws),native_falling_uniform=stats(falling_draws),
                player_error_by_case=case_stats,failures=failures)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture',type=Path)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--save-fixture',type=Path)
    args=parser.parse_args()
    records=read_records(args.fixture)
    if args.save_fixture:
        payload=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records).encode()
        args.save_fixture.write_bytes(gzip.compress(payload,mtime=0))
    report=verify(args.exe.resolve(),records)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='player_error_by_case'},indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__': main()
