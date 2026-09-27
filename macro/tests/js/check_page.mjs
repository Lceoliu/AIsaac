// Checks of the page's own logic (web/static/js) on floors the generators made: node check_page.mjs FLOORS.json
// Run by tests/test_js.py.
import { readFileSync } from 'node:fs';
import { aiExpected, aiReplay } from '../../web/static/js/ai.js';
import { hiddenRooms, keyTargets, newPlay, useBomb } from '../../web/static/js/play.js';
import { floorScore } from '../../web/static/js/score.js';

const runs = JSON.parse(readFileSync(process.argv[2], 'utf8'));
let floors = 0;
for (const run of runs) {
  const rep = run.game === 'repplus';
  for (const f of run.floors) {
    floors += 1;
    const where = `${run.game} ${run.seed.text} ${f.name_en}`;
    // the AI finds every hidden room, and on average it always gets there
    const r = aiReplay(f, rep);
    const found = new Set(r.final.values());
    for (const h of hiddenRooms(f)) if (!found.has(h.kind)) throw new Error(`${where}: the AI missed the ${h.kind} room`);
    const e = aiExpected(f, rep);
    if (!e || e.unfinished > 1e-9 || e.bombs < 1 - 1e-9) throw new Error(`${where}: expected ${JSON.stringify(e)}`);
    if (!rep && e.keys !== 0) throw new Error(`${where}: AB+ has no Red Key`);
    // a wall bombed for nothing is still a door slot for the Red Key
    if (rep && f.door_targets.length) {
      const hidden = new Set(hiddenRooms(f).map((h) => h.cell));
      const slot = f.door_targets.find((c) => !hidden.has(c));
      if (slot !== undefined) {
        const play = newPlay();
        useBomb(f, play, slot);
        if (!keyTargets(f, play).includes(slot)) throw new Error(`${where}: bombed slot ${slot} lost to the Red Key`);
      }
    }
  }
}
// points: a find, a miss, the clean bonus
const floor = runs[0].floors[0];
const [first] = hiddenRooms(floor);
const miss = floorScore(floor, { log: [{ found: [] }, { found: [first.kind] }], revealed: false, obs: new Map() });
if (miss.total !== 100 - 25 && first.kind === 'secret') throw new Error(`score ${JSON.stringify(miss)}`);
console.log(`ok ${floors} floors`);
