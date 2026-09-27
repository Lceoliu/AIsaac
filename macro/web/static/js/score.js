// The quiz's points for a floor. Which rules apply goes by the day a run began (China time), so a run
// keeps its rules to the end and a day's board never mixes two:
// - v1, to 2026-09-28: a hidden room found +100 / +150 / +200 (secret / super / ultra), −25 for each
//   bomb or Red Key that found nothing, +50 for a floor with no miss.
// - v2, from 2026-09-29 (the site owner's design): the rooms are worth half on floors 1-2; each room
//   found on the first try since the last find +25 (+50 on floors 6-7); a secret room next to one
//   room only +25, an ultra secret room one door slot only can reach +50; a floor with no miss
//   +25 / +50 / +125 (floors 1-2 / 3-5 / 6-7); still −25 a miss. At most 800 on a floor.
// A floor never goes below 0. Challenges have no hints, so nothing else counts.
import { neighbours } from './infer.js';
import { hiddenRooms, isComplete } from './play.js';

export const FIND_POINTS = { secret: 100, super: 150, ultra: 200 };
export const MISS_POINTS = 25;
export const V2_FROM = '2026-09-29';

const RULES = {
  1: { v: 1, room: (k) => FIND_POINTS[k], first: () => 0, clean: () => 50, link: {} },
  2: {
    v: 2,
    room: (k, tier) => (tier === 0 ? FIND_POINTS[k] / 2 : FIND_POINTS[k]),
    first: (tier) => (tier === 2 ? 50 : 25),
    clean: (tier) => [25, 50, 125][tier],
    link: { secret: 25, ultra: 50 },
  },
};
export function rulesFor(day) { return day && day >= V2_FROM ? RULES[2] : RULES[1]; }
export function rulesOf(version) { return RULES[version] || RULES[1]; }

// floors 1-2, 3-5, 6 and deeper, by the floor's place in the run
export function tierOf(index) { return index < 2 ? 0 : index < 5 ? 1 : 2; }

// The hidden rooms with one link only: a secret room next to a single explored room, an ultra secret
// room next to a single door slot of the explored rooms (the one place a red room reaches it from).
// Rare and hard (on 7,000 challenge floors: 1.6% and 6.4%, the AI needs 3.9 tries against 1.6 and
// 5.4 against 3.8). Shown only once the room is found: before that it would tell where to look.
const LINKS = new WeakMap();
export function singleLinked(floor) {
  if (LINKS.has(floor)) return LINKS.get(floor);
  const owner = new Map();
  for (const r of floor.rooms) if (!r.hidden) for (const c of r.cells) owner.set(c, r.index);
  const slots = new Set(floor.door_targets || []);
  const out = new Set();
  for (const h of hiddenRooms(floor)) {
    if (h.kind === 'secret' && new Set(neighbours(h.cell).map((n) => owner.get(n)).filter((x) => x !== undefined)).size === 1) out.add('secret');
    if (h.kind === 'ultra' && neighbours(h.cell).filter((n) => slots.has(n)).length === 1) out.add('ultra');
  }
  LINKS.set(floor, out);
  return out;
}

// what one find is worth, part by part: [[label, points]]
function findParts(floor, kind, tier, first, rules) {
  const parts = [[kind, rules.room(kind, tier)]];
  if (first && rules.first(tier)) parts.push(['first', rules.first(tier)]);
  if (rules.link[kind] && singleLinked(floor).has(kind)) parts.push(['link', rules.link[kind], kind]);
  return parts;
}

// The floor's points: `per` has each action's points in log order (a miss is −25), `parts` the finds'
// breakdown; `index` is the floor's place in the run.
export function floorScore(floor, play, index = 0, rules = RULES[1]) {
  const tier = tierOf(index);
  const per = [], parts = [];
  let gain = 0, misses = 0, firsts = 0, firstPts = 0, linkPts = 0, since = 0;
  for (const e of play.log) {
    if (!e.found.length) {
      per.push(-MISS_POINTS);
      parts.push(null);
      misses += 1;
      since += 1;
      continue;
    }
    const got = e.found.flatMap((k) => findParts(floor, k, tier, since === 0, rules));
    const pts = got.reduce((a, [, v]) => a + v, 0);
    for (const [label, v] of got) {
      if (label === 'first') { firsts += 1; firstPts += v; }
      if (label === 'link') linkPts += v;
    }
    per.push(pts);
    parts.push(got);
    gain += pts;
    since = 0;
  }
  const complete = isComplete(floor, play);
  const clean = complete && !play.revealed && misses === 0;
  const bonus = clean ? rules.clean(tier) : 0;
  const total = Math.max(0, Math.round(gain - misses * MISS_POINTS + bonus));
  return { per, parts, gain, misses, firsts, firstPts, linkPts, clean, bonus, total, complete, done: complete || play.revealed, tier };
}

// the most a floor gives without its single-link bonuses (those would give away where the room is)
export function maxScore(floor, index = 0, rules = RULES[1]) {
  const tier = tierOf(index);
  return hiddenRooms(floor).reduce((a, h) => a + rules.room(h.kind, tier) + rules.first(tier), 0) + rules.clean(tier);
}
