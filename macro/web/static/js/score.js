// The quiz's points for a floor: each hidden room found, minus the bombs and Red Keys that found
// nothing, plus a bonus for a clean floor. Challenges have no hints, so nothing else counts.
import { hiddenRooms, isComplete } from './play.js';

export const FIND_POINTS = { secret: 100, super: 150, ultra: 200 };
export const MISS_POINTS = 25;
export const CLEAN_BONUS = 50;

// the points one action earned: what it found, or the miss
export function actionPoints(entry) {
  if (!entry.found.length) return -MISS_POINTS;
  return entry.found.reduce((a, k) => a + FIND_POINTS[k], 0);
}

export function floorScore(floor, play) {
  let gain = 0, misses = 0;
  for (const e of play.log) {
    if (e.found.length) gain += actionPoints(e);
    else misses += 1;
  }
  const complete = isComplete(floor, play);
  const clean = complete && !play.revealed && misses === 0;
  const total = Math.max(0, Math.round(gain - misses * MISS_POINTS + (clean ? CLEAN_BONUS : 0)));
  return { gain, misses, clean, total, complete, done: complete || play.revealed };
}

export function maxScore(floor) {
  return hiddenRooms(floor).reduce((a, h) => a + FIND_POINTS[h.kind], 0) + CLEAN_BONUS;
}
