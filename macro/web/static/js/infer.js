// Linked hidden-room inference on the page. A floor's joint rows are [secret cell, super secret cell,
// ultra secret cell, p] (-1: no such room), the posterior given the visible map (web/api.py). What the
// player learns only removes rows, so every update is exact: keep the consistent rows, renormalise.
export const GRID = 13;
const TRAVEL = [[-1, 0], [0, -1], [1, 0], [0, 1]];
export const KINDS = ['secret', 'super', 'ultra'];          // the joint's columns
export const KIND_TYPE = { secret: 7, super: 8, ultra: 29 };
export const TYPE_KIND = { 7: 'secret', 8: 'super', 29: 'ultra' };
export const KIND_NAME = { secret: '隐藏房', super: '超级隐藏房', ultra: '究极隐藏房' };

export function neighbours(c) {
  const x = c % GRID, y = Math.floor(c / GRID);
  return TRAVEL.map(([dx, dy]) => [x + dx, y + dy])
    .filter(([a, b]) => a >= 0 && a < GRID && b >= 0 && b < GRID).map(([a, b]) => a + b * GRID);
}

// obs: Map(cell -> 'empty' | 'red' | 'secret' | 'super' | 'ultra'). A red room (Level::MakeRedRoomDoor,
// RVA 0x34D010) opens a door to every adjacent room, so its neighbours hold no hidden room that has
// not been found through it.
export function consistent(row, obs) {
  const [c, h, u] = row;
  for (const [cell, kind] of obs) {
    if (kind === 'secret') { if (c !== cell) return false; continue; }
    if (kind === 'super') { if (h !== cell) return false; continue; }
    if (kind === 'ultra') { if (u !== cell) return false; continue; }
    if (c === cell || h === cell || u === cell) return false;
    if (kind === 'red') {
      const nb = neighbours(cell);
      if (c >= 0 && nb.includes(c) && obs.get(c) !== 'secret') return false;
      if (h >= 0 && nb.includes(h) && obs.get(h) !== 'super') return false;
      if (u >= 0 && nb.includes(u) && obs.get(u) !== 'ultra') return false;
    }
  }
  return true;
}

export function infer(joint, obs, extra = null) {
  let checks = obs;
  if (extra) {
    checks = new Map(obs);
    checks.set(extra[0], extra[1]);
  }
  const rows = joint.filter((row) => consistent(row, checks));
  let total = 0;
  for (const row of rows) total += row[3];
  const out = { ok: total > 0, rows, secret: new Map(), super: new Map(), ultra: new Map() };
  if (!out.ok) return out;
  for (const row of rows) {
    for (let k = 0; k < 3; k++) {
      if (row[k] >= 0) out[KINDS[k]].set(row[k], (out[KINDS[k]].get(row[k]) || 0) + row[3] / total);
    }
  }
  return out;
}

export function best(map) {
  let top = null;
  for (const [c, p] of map) {
    if (!top || p > top[1] + 1e-12 || (Math.abs(p - top[1]) <= 1e-12 && c < top[0])) top = [c, p];
  }
  return top;
}

// the wall to bomb: the cell most likely to hold the secret or the super secret room
export function nextBomb(inf, obs, allowed = null) {
  const score = new Map();
  for (const k of ['secret', 'super']) {
    for (const [c, p] of inf[k]) {
      if (!obs.has(c) && (!allowed || allowed(c))) score.set(c, (score.get(c) || 0) + p);
    }
  }
  return best(score);
}

// the cell to open with the Red Key: a red room there reaches the ultra secret room if it is adjacent.
// When no door slot is next to a place the ultra secret room could be, the one with most of it two
// cells away (a red room there opens the way for a second one).
export function nextRedKey(inf, obs, targets) {
  if (!inf.ultra.size || [...obs.values()].includes('ultra')) return null;
  const open = (c) => !obs.has(c) || obs.get(c) === 'empty';
  const score = new Map(), reach = new Map();
  for (const e of targets) {
    if (!open(e)) continue;
    let p = 0, q = 0;
    const near = neighbours(e);
    for (const n of near) p += inf.ultra.get(n) || 0;
    for (const n of near) {
      if (!open(n)) continue;
      for (const m of neighbours(n)) if (m !== e && !near.includes(m)) q += inf.ultra.get(m) || 0;
    }
    if (p > 1e-9) score.set(e, p);
    else if (q > 1e-9) reach.set(e, q);
  }
  return best(score) || best(reach);
}

// Expected bombs to find the secret and super secret rooms when always bombing the nextBomb cell,
// averaged exactly over the joint (a decision tree: each bomb splits the rows by what it reveals).
export function expectedBombs(joint) {
  let total = 0;
  for (const r of joint) total += r[3];
  if (!total) return null;
  const solve = (rows, tested, foundC, foundH, depth) => {
    if (!rows.length) return 0;
    let mass = 0;
    for (const r of rows) mass += r[3];
    if ((foundC && (foundH || rows.every((r) => r[1] < 0))) || depth > 40) return mass * depth;
    const score = new Map();
    for (const r of rows) {
      if (!foundC && !tested.has(r[0])) score.set(r[0], (score.get(r[0]) || 0) + r[3]);
      if (!foundH && r[1] >= 0 && !tested.has(r[1])) score.set(r[1], (score.get(r[1]) || 0) + r[3]);
    }
    const pick = best(score);
    if (!pick) return mass * depth;
    const x = pick[0];
    const next = new Set(tested).add(x);
    const c = [], h = [], e = [];
    for (const r of rows) {
      if (!foundC && r[0] === x) c.push(r);
      else if (!foundH && r[1] === x) h.push(r);
      else e.push(r);
    }
    return solve(c, next, true, foundH, depth + 1) + solve(h, next, foundC, true, depth + 1)
      + solve(e, next, foundC, foundH, depth + 1);
  };
  return solve(joint, new Set(), false, false, 0) / total;
}

export function pct(p) {
  if (p >= 0.995) return '100%';
  if (p > 0 && p < 0.01) return '<1%';
  return `${Math.round(p * 100)}%`;
}

// a cell relative to the start room, the way players give directions
export function where(c, start) {
  const dx = c % GRID - start % GRID, dy = Math.floor(c / GRID) - Math.floor(start / GRID);
  const parts = (dx ? (dx > 0 ? '右' : '左') + Math.abs(dx) : '') + (dy ? (dy > 0 ? '下' : '上') + Math.abs(dy) : '');
  return parts ? '起点' + parts : '起点';
}
