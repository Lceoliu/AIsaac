// The score queue (web/static/js/scoreboard.js) against a fake Supabase: node check_scoreboard.mjs
// A score the database refuses (a SQLSTATE code, here the daily-date rule) is dropped and the rest
// still go out; a network failure or a gateway error (no code) keeps everything for later.
// Run by tests/test_js.py.
const kept = new Map();
globalThis.localStorage = {
  getItem: (k) => (kept.has(k) ? kept.get(k) : null),
  setItem: (k, v) => { kept.set(k, String(v)); },
  removeItem: (k) => { kept.delete(k); },
};
globalThis.window = { MAPGEN_CONFIG: { supabase: { url: 'https://example.supabase.co', key: 'sb_publishable_test' } } };

const STALE = '2026-09-20';            // a daily score too old for the insert policy
const sent = [];
let net = 'up';
globalThis.fetch = async (url, opts = {}) => {
  if (net === 'down') throw new TypeError('Failed to fetch');
  if (net === 'badkey') return new Response(JSON.stringify({ message: 'Invalid API key' }), { status: 401 });
  const entry = JSON.parse(opts.body);
  if (entry.day === STALE) {
    return new Response(JSON.stringify({ code: '42501', message: 'new row violates row-level security policy' }), { status: 401 });
  }
  sent.push(entry);
  return new Response(null, { status: 201 });
};

const { createScoreboard } = await import('../../web/static/js/scoreboard.js');
const sb = createScoreboard();
const score = (floor, day) => ({ client_id: 'c', player: 'p', game: 'abplus', mode: 'daily', day, seed: 7, floor, points: 100,
  bombs: 1, keys: 0, hints: false });
const check = (ok, what) => { if (!ok) throw new Error(what); };

// offline: both wait in the queue
net = 'down';
let r = await sb.submit(score(0, STALE));
r = await sb.submit(score(1, '2026-09-27'));
check(r.pending === 2 && r.error && !r.refused, `offline: ${JSON.stringify(r)}`);
// a gateway error (no SQLSTATE) is not a refusal: still queued
net = 'badkey';
r = await sb.flush();
check(r.pending === 2 && r.error && r.refused.length === 0, `bad key: ${JSON.stringify(r)}`);
// back online: the refused one is dropped instead of blocking the one behind it
net = 'up';
r = await sb.flush();
check(r.pending === 0 && !r.error && r.refused.length === 1 && sent.length === 1 && sent[0].floor === 1, `online: ${JSON.stringify(r)}`);
// a new score the database refuses says so
r = await sb.submit(score(2, STALE));
check(r.refused === true && r.pending === 0 && !r.error, `refused: ${JSON.stringify(r)}`);
// and a good one goes straight out
r = await sb.submit(score(3, '2026-09-27'));
check(!r.refused && r.pending === 0 && sent.length === 2, `sent: ${JSON.stringify(r)}`);
console.log('ok scoreboard queue');
