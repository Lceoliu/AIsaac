// The AI's way through a floor: the hint policy played out. It knows what the player knows (the
// explored map, the room layouts, which kinds of hidden room the floor has) and nothing else. It bombs
// the wall most likely to hold the secret or the super secret room until both are found, then (on
// Repentance+) opens with the Red Key the door most likely to reach the ultra secret room.
//
// aiReplay(floor)  its moves on this floor, with the probabilities it saw at each step
// aiExpected(floor) the bombs and keys it needs on average: exact over the floor's joint posterior
//                   (each move splits the hypotheses by what it would reveal)
import { consistent, infer, neighbours, nextBomb, nextRedKey } from './infer.js';
import { canBomb, hiddenRooms, keyTargets, newPlay, useBomb, useKey } from './play.js';

const COL = { secret: 0, super: 1, ultra: 2 };

// the joint, restricted to the kinds of hidden room the floor shows the player it has
function knownJoint(floor) {
  const kinds = new Set(hiddenRooms(floor).map((h) => h.kind));
  const rows = floor.joint.filter((r) => (r[1] >= 0) === kinds.has('super') && (r[2] >= 0) === kinds.has('ultra'));
  return rows.reduce((a, r) => a + r[3], 0) > 0 ? rows : floor.joint;
}

// the next move for these observations: ['bomb' | 'key', cell, p] or null when done
function nextMove(floor, rows, obs, kinds, rep) {
  const found = new Set(obs.values());
  const inf = infer(rows, obs);
  if (!inf.ok) return null;
  const play = { obs };
  if (!found.has('secret') || (kinds.has('super') && !found.has('super'))) {
    const pick = nextBomb(inf, obs, (c) => canBomb(floor, play, c));
    if (pick) return ['bomb', pick[0], pick[1], inf];
  }
  if (rep && kinds.has('ultra') && !found.has('ultra')) {
    const pick = nextRedKey(inf, obs, keyTargets(floor, play));
    if (pick) return ['key', pick[0], pick[1], inf];
  }
  return null;
}

export function aiReplay(floor, rep) {
  const kinds = new Set(hiddenRooms(floor).map((h) => h.kind));
  const rows = knownJoint(floor);
  const play = newPlay();
  const steps = [];
  for (let guard = 0; guard < 60; guard++) {
    const move = nextMove(floor, rows, play.obs, kinds, rep);
    if (!move) break;
    const [tool, cell, p, inf] = move;
    const before = new Map(play.obs);
    const entry = tool === 'bomb' ? useBomb(floor, play, cell) : useKey(floor, play, cell);
    steps.push({ tool, cell, p, found: entry.found, added: entry.added, before, after: new Map(play.obs), inf });
  }
  return { steps, bombs: play.bombs, keys: play.keys, final: play.obs, rows };
}

// what a move reveals under one hypothesis, as a key to group the hypotheses by
function outcome(tool, cell, row, obs) {
  const at = (c) => (row[0] === c ? 'secret' : row[1] === c ? 'super' : row[2] === c ? 'ultra' : null);
  const kind = at(cell);
  if (tool === 'bomb' || kind) return [[cell, kind || 'empty']];
  const out = [[cell, 'red']];
  for (const n of neighbours(cell)) {
    const k = at(n);
    if (k && !obs.has(n)) out.push([n, k]);
  }
  return out;
}

export function aiExpected(floor, rep) {
  const kinds = new Set(hiddenRooms(floor).map((h) => h.kind));
  const rows = knownJoint(floor);
  const total = rows.reduce((a, r) => a + r[3], 0);
  if (!total) return null;
  let bombs = 0, keys = 0, stuck = 0;
  const solve = (subset, obs, nb, nk, depth) => {
    const mass = subset.reduce((a, r) => a + r[3], 0);
    const move = depth < 60 ? nextMove(floor, subset, obs, kinds, rep) : null;
    if (!move) {
      bombs += mass * nb;
      keys += mass * nk;
      const found = new Set(obs.values());
      if ([...kinds].some((k) => !found.has(k))) stuck += mass;
      return;
    }
    const [tool, cell] = move;
    const groups = new Map();
    for (const r of subset) {
      const add = outcome(tool, cell, r, obs);
      const key = add.map(([c, k]) => `${c}:${k}`).join(',');
      if (!groups.has(key)) groups.set(key, { add, rows: [] });
      groups.get(key).rows.push(r);
    }
    for (const { add, rows: part } of groups.values()) {
      const next = new Map(obs);
      for (const [c, k] of add) next.set(c, k);
      solve(part.filter((r) => consistent(r, next)), next, nb + (tool === 'bomb' ? 1 : 0), nk + (tool === 'key' ? 1 : 0), depth + 1);
    }
  };
  solve(rows, new Map(), 0, 0, 0);
  return { bombs: bombs / total, keys: keys / total, unfinished: stuck / total };
}

export { COL };
