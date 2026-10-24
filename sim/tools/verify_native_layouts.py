"""Compare extracted room presets to live J460 goto s.boss.* terrain."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
layouts=json.loads((Path(__file__).resolve().parents[1]/'tests/fixtures/monstro_layouts.json').read_text())['rooms']
rows=[json.loads(l) for l in a.trace.read_text().splitlines()];report=[]
for expected in layouts:
 obs=next(r['obs'] for r in rows if r['case']==f'layout_{expected["variant"]}' and r['tick']==3)
 assert obs['room']['variant']==expected['variant']
 rocks=[c[0] for c in obs['terrain']['cells'] if c[9]]
 assert rocks==expected['rocks'],(expected['variant'],rocks,expected['rocks'])
 report.append(dict(variant=expected['variant'],rock_cells=len(rocks),grid_dimensions=[obs['terrain']['width'],obs['terrain']['height']],native_door_slots=[d['slot'] for d in obs['doors']],passed=True))
a.report.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
