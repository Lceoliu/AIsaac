// Checks of the page's own logic (web/static/js) on floors the generators made: node check_page.mjs FLOORS.json
// Run by tests/test_js.py.
import { readFileSync } from 'node:fs';
import { aiExpected, aiReplay } from '../../web/static/js/ai.js';
import { hiddenRooms, keyTargets, newPlay, undo, useBomb, useKey } from '../../web/static/js/play.js';
import { floorScore, maxScore, rulesFor, singleLinked } from '../../web/static/js/score.js';

const runs = JSON.parse(readFileSync(process.argv[2], 'utf8'));
let floors = 0;
let keyOnHidden = 0;
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
    // the Red Key right on a secret or super secret room counts as a bomb (and undoes as one)
    if (rep) {
      const play = newPlay();
      const targets = new Set(keyTargets(f, play));
      const h = hiddenRooms(f).find((x) => (x.kind === 'secret' || x.kind === 'super') && targets.has(x.cell));
      if (h) {
        const entry = useKey(f, play, h.cell);
        if (entry.tool !== 'bomb' || play.bombs !== 1 || play.keys !== 0 || play.obs.get(h.cell) !== h.kind) {
          throw new Error(`${where}: the key on the ${h.kind} room gave ${JSON.stringify({ tool: entry.tool, bombs: play.bombs, keys: play.keys })}`);
        }
        undo(play);
        if (play.bombs !== 0 || play.keys !== 0 || play.obs.has(h.cell)) throw new Error(`${where}: undo after the key on the ${h.kind} room`);
        keyOnHidden += 1;
      }
    }
  }
}
// points: a find, a miss, the clean bonus
const floor = runs[0].floors[0];
const [first] = hiddenRooms(floor);
const miss = floorScore(floor, { log: [{ found: [] }, { found: [first.kind] }], revealed: false, obs: new Map() });
if (miss.total !== 100 - 25 && first.kind === 'secret') throw new Error(`score ${JSON.stringify(miss)}`);
if (!keyOnHidden) throw new Error('no floor had a secret room behind a door slot');

// the 2026-09-29 rules (v2): floors 1-2 give half the room points; each room found on the first try
// +25 (+50 on floors 6-7); single-link bonuses; a floor with no miss +25 / +50 / +125
const v1 = rulesFor('2026-09-28'), v2 = rulesFor('2026-09-29');
if (v1.v !== 1 || v2.v !== 2) throw new Error('rules by day');
const check = (ok, what) => { if (!ok) throw new Error(what); };
const HALF = { secret: 50, super: 75, ultra: 100 }, FULL = { secret: 100, super: 150, ultra: 200 };
for (const run of runs) {
  const f = run.floors[0];
  const hid = hiddenRooms(f);
  const links = singleLinked(f);
  const bonusOf = (k) => (links.has(k) ? { secret: 25, ultra: 50 }[k] || 0 : 0);
  // floor 1, every room on its first try: halves + 25 each + the links + 25 for no miss
  const clean = newPlay();
  for (const h of hid) useBomb(f, clean, h.cell);
  const want1 = hid.reduce((a, h) => a + HALF[h.kind] + 25 + bonusOf(h.kind), 0) + 25;
  const got1 = floorScore(f, clean, 0, v2);
  check(got1.total === want1 && got1.clean && got1.firsts === hid.length, `v2 floor 1 clean: ${got1.total} != ${want1}`);
  check(maxScore(f, 0, v2) === hid.reduce((a, h) => a + HALF[h.kind] + 25, 0) + 25, 'v2 max floor 1');
  // floor 6 with a miss first: no first-try bonus for the first room, none for the floor
  const g = run.floors[5];
  const hg = hiddenRooms(g);
  const lg = singleLinked(g);
  const missy = newPlay();
  const taken = new Set(hg.map((h) => h.cell));
  const spare = g.door_targets.find((c) => !taken.has(c) && !g.rooms.some((r) => r.cells.includes(c)));
  if (spare === undefined) continue;
  useBomb(g, missy, spare);
  for (const h of hg) useBomb(g, missy, h.cell);
  const want6 = hg.reduce((a, h, i) => a + FULL[h.kind] + (i === 0 ? 0 : 50) + (lg.has(h.kind) ? { secret: 25, ultra: 50 }[h.kind] || 0 : 0), 0) - 25;
  const got6 = floorScore(g, missy, 5, v2);
  check(got6.total === want6 && !got6.clean && got6.per[0] === -25, `v2 floor 6 after a miss: ${got6.total} != ${want6}`);
  // v1 is unchanged: rooms at full points, +50 for no miss, whatever the floor
  check(floorScore(g, missy, 5, v1).total === hg.reduce((a, h) => a + FULL[h.kind], 0) - 25, 'v1 unchanged');
}
// single-link detection on a small made-up floor (13 wide): explored rooms at 40 and 42
const room = (index, cell, type = 1, hidden = false) => ({ index, cells: [cell], type, hidden });
const one = { rooms: [room(0, 40), room(1, 42), room(2, 27, 7, true), room(3, 56, 29, true)], door_targets: [27, 55] };
check(singleLinked(one).has('secret'), 'a secret room next to one room is single-linked');          // 27 touches 40 only
check(singleLinked(one).has('ultra'), 'an ultra room next to one door slot is single-linked');      // 56: 55 is its only slot
const two = { rooms: [room(0, 40), room(1, 42), room(2, 41, 7, true), room(3, 56, 29, true)], door_targets: [55, 57] };   // 41 touches 40 and 42
check(!singleLinked(two).has('secret') && !singleLinked(two).has('ultra'), 'two links are not single');
console.log(`ok ${floors} floors, ${keyOnHidden} keys on a secret room`);
