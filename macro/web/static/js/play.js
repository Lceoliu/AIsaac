// Find mode: the player sees the explored floor without its hidden rooms and uses bombs (and, on
// Repentance+, the Red Key) on empty cells; the answer comes from the generated floor itself.
import { GRID, TYPE_KIND, neighbours } from './infer.js';

export function newPlay() {
  return { obs: new Map(), bombs: 0, keys: 0, log: [], revealed: false };
}

const ORDER = { secret: 0, super: 1, ultra: 2 };

// the hidden rooms of this floor: [{kind, cell, room}], secret room first
export function hiddenRooms(floor) {
  return floor.rooms.filter((r) => r.hidden).map((r) => ({ kind: TYPE_KIND[r.type], cell: r.cells[0], room: r }))
    .sort((a, b) => ORDER[a.kind] - ORDER[b.kind]);
}

function truthAt(floor, cell) {
  const r = floor.rooms.find((q) => q.hidden && q.cells.includes(cell));
  return r ? TYPE_KIND[r.type] : null;
}

function visibleCells(floor) {
  const out = new Set();
  for (const r of floor.rooms) if (!r.hidden) for (const c of r.cells) out.add(c);
  return out;
}

// cells the player can stand in: explored rooms, hidden rooms found, red rooms opened
export function knownCells(floor, play) {
  const out = visibleCells(floor);
  for (const [c, kind] of play.obs) if (kind !== 'empty') out.add(c);
  return out;
}

export function isFree(floor, play, cell) {
  return cell >= 0 && cell < GRID * GRID && !visibleCells(floor).has(cell) && !play.obs.has(cell);
}

// a bomb opens a wall of a room the player can reach
export function canBomb(floor, play, cell) {
  if (!isFree(floor, play, cell)) return false;
  const known = knownCells(floor, play);
  return neighbours(cell).some((n) => known.has(n));
}

// the Red Key opens a door slot: those of the explored rooms' layouts (floor.door_targets), and every
// side of a hidden or red room found so far (1x1 rooms). A wall already bombed for nothing is still a
// door slot: the key makes a red room there all the same.
export function keyTargets(floor, play) {
  const out = new Set(floor.door_targets || []);
  for (const [c, kind] of play.obs) {
    if (kind !== 'empty') for (const n of neighbours(c)) out.add(n);
  }
  const visible = visibleCells(floor);
  return [...out].filter((c) => c >= 0 && c < GRID * GRID && !visible.has(c) && (!play.obs.has(c) || play.obs.get(c) === 'empty'));
}

export function canKey(floor, play, cell) {
  return keyTargets(floor, play).includes(cell);
}

// returns {added: [[cell, kind]], found: [kind]} and records it (with what it replaced) for undo
export function useBomb(floor, play, cell) {
  const kind = truthAt(floor, cell);
  const added = [[cell, kind || 'empty']];
  const prev = added.map(([c]) => [c, play.obs.get(c)]);
  play.obs.set(cell, kind || 'empty');
  play.bombs += 1;
  const entry = { tool: 'bomb', cell, added, prev, found: kind ? [kind] : [] };
  play.log.push(entry);
  return entry;
}

export function useKey(floor, play, cell) {
  const kind = truthAt(floor, cell);
  const added = [];
  const found = [];
  if (kind) {                       // the door opens into a hidden room
    added.push([cell, kind]);
    found.push(kind);
  } else {                          // a red room: it connects to every adjacent room
    added.push([cell, 'red']);
    for (const n of neighbours(cell)) {
      const k = truthAt(floor, n);
      if (k && !play.obs.has(n)) {
        added.push([n, k]);
        found.push(k);
      }
    }
  }
  const prev = added.map(([c]) => [c, play.obs.get(c)]);
  for (const [c, k] of added) play.obs.set(c, k);
  play.keys += 1;
  const entry = { tool: 'key', cell, added, prev, found };
  play.log.push(entry);
  return entry;
}

export function undo(play) {
  const entry = play.log.pop();
  if (!entry) return null;
  for (const [c, k] of entry.prev || entry.added.map(([c]) => [c, undefined])) {
    if (k === undefined) play.obs.delete(c); else play.obs.set(c, k);
  }
  if (entry.tool === 'bomb') play.bombs -= 1;
  else play.keys -= 1;
  return entry;
}

export function foundKinds(play) {
  const out = new Set();
  for (const kind of play.obs.values()) if (kind === 'secret' || kind === 'super' || kind === 'ultra') out.add(kind);
  return out;
}

export function isComplete(floor, play) {
  const found = foundKinds(play);
  return hiddenRooms(floor).every((h) => found.has(h.kind));
}
