// The score queue and the boards (web/static/js/scoreboard.js) against a fake Supabase: node check_scoreboard.mjs
// A score the database refuses (a SQLSTATE code, here the daily-date rule) is dropped and the rest
// still go out; a network failure or a gateway error (no code) keeps everything for later. A board
// is read in the order points, [time,] who finished first, and a player outside the top rows gets
// their exact place; a database without the time column yet still takes scores and shows boards.
// A daily floor's moves wait for the network like the scores but never block them, and a day's
// replays are handed out only for a token that sent that day's 7 floors (hr_replay's rule).
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
let noMs = false;                      // a database from before the ms column
let views = [];                        // hr_board_daily's rows
const moves = [];                      // hr_moves
let noMovesTable = false;

// ---- a little PostgREST: eq filters, an or=(...) tree of gt/lt/eq/is, order, limit, count=exact
function parseTree(src) {
  let i = 0;
  const item = () => {
    for (const op of ['and(', 'or(']) {
      if (src.startsWith(op, i)) { i += op.length; return { op: op.slice(0, -1), items: list() }; }
    }
    let j = i, quoted = false;
    while (j < src.length && (quoted || (src[j] !== ',' && src[j] !== ')'))) { if (src[j] === '"') quoted = !quoted; j += 1; }
    const m = src.slice(i, j).match(/^(\w+)\.(not\.)?(\w+)\.(.*)$/);
    i = j;
    return { col: m[1], not: !!m[2], cmp: m[3], val: m[4].replace(/^"(.*)"$/, '$1') };
  };
  const list = () => {
    const items = [];
    for (;;) {
      items.push(item());
      if (src[i] === ',') { i += 1; continue; }
      if (src[i] === ')') { i += 1; return items; }
      throw new Error(`bad tree ${src}`);
    }
  };
  if (src[i] !== '(') throw new Error(`bad tree ${src}`);
  i += 1;
  return { op: 'or', items: list() };
}
function holds(node, row) {
  if (node.op) return node.op === 'and' ? node.items.every((n) => holds(n, row)) : node.items.some((n) => holds(n, row));
  const v = row[node.col];
  const x = typeof v === 'number' ? Number(node.val) : node.val;
  let r;
  if (node.cmp === 'is') r = node.val === 'null' ? v === null : false;
  else if (v === null) return false;                 // SQL: a comparison with null is not true
  else r = { gt: v > x, lt: v < x, eq: v === x }[node.cmp];
  return node.not ? !r : r;
}
function serve(url) {
  const q = new URL(url).searchParams;
  const cols = q.get('select').split(',');
  if (noMs && (cols.includes('ms') || /ms\./.test(q.get('order') || '') || /ms\./.test(q.get('or') || ''))) {
    return new Response(JSON.stringify({ code: '42703', message: 'column hr_board_daily.ms does not exist' }), { status: 400 });
  }
  let rows = views.filter((r) => ['game', 'day', 'client_id'].every((k) => !q.has(k) || `eq.${r[k]}` === q.get(k)));
  if (q.has('or')) { const tree = parseTree(q.get('or')); rows = rows.filter((r) => holds(tree, r)); }
  if (q.has('order')) {
    const keys = q.get('order').split(',').map((o) => o.split('.'));
    rows = [...rows].sort((a, b) => {
      for (const [col, dir, nulls] of keys) {
        const x = a[col], y = b[col];
        if (x === y) continue;
        if (x === null || y === null) return (x === null ? 1 : -1) * (nulls === 'nullslast' ? 1 : -1);
        return (x < y ? -1 : 1) * (dir === 'desc' ? -1 : 1);
      }
      return 0;
    });
  }
  const total = rows.length;
  rows = rows.slice(0, Number(q.get('limit') || rows.length)).map((r) => Object.fromEntries(cols.map((c) => [c, r[c]])));
  return new Response(JSON.stringify(rows), { status: 200, headers: { 'content-range': `0-${rows.length - 1}/${total}` } });
}
globalThis.fetch = async (url, opts = {}) => {
  if (net === 'down') throw new TypeError('Failed to fetch');
  if (net === 'badkey') return new Response(JSON.stringify({ message: 'Invalid API key' }), { status: 401 });
  if ((opts.method || 'GET') === 'GET') return serve(url);
  const entry = JSON.parse(opts.body);
  if (url.endsWith('/hr_moves')) {
    if (noMovesTable) return new Response(JSON.stringify({ code: 'PGRST205', message: "Could not find the table 'public.hr_moves'" }), { status: 404 });
    if (moves.some((m) => m.client_id === entry.client_id && m.game === entry.game && m.day === entry.day && m.floor === entry.floor)) {
      return new Response(JSON.stringify({ code: '23505' }), { status: 409 });
    }
    moves.push(entry);
    return new Response(null, { status: 201 });
  }
  if (url.endsWith('/rpc/hr_replay')) {                // the SQL function's rule
    const at = (m) => m.game === entry.p_game && m.day === entry.p_day;
    if (moves.filter((m) => at(m) && m.client_id === entry.p_client && m.token === entry.p_token).length < 7) {
      return new Response(JSON.stringify({ code: 'P0001', message: "finish this day's challenge first" }), { status: 400 });
    }
    const out = moves.filter((m) => at(m) && m.client_id === entry.p_player).sort((a, b) => a.floor - b.floor)
      .map(({ floor, moves: mv, gave_up }) => ({ floor, moves: mv, gave_up }));
    return new Response(JSON.stringify(out), { status: 200 });
  }
  if (noMs && 'ms' in entry) {
    return new Response(JSON.stringify({ code: 'PGRST204', message: "Could not find the 'ms' column of 'hr_scores' in the schema cache" }), { status: 400 });
  }
  if (entry.day === STALE) {
    return new Response(JSON.stringify({ code: '42501', message: 'new row violates row-level security policy' }), { status: 401 });
  }
  sent.push(entry);
  return new Response(null, { status: 201 });
};

const { TIME_FROM, createScoreboard } = await import('../../web/static/js/scoreboard.js');
const check = (ok, what) => { if (!ok) throw new Error(what); };

// ---- the queue
{
  const sb = createScoreboard();
  const score = (floor, day) => ({ client_id: 'c', player: 'p', game: 'abplus', mode: 'daily', day, seed: 7, floor, points: 100,
    bombs: 1, keys: 0, hints: false, ms: 61000 });
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
  // and a good one goes straight out, with its time
  r = await sb.submit(score(3, '2026-09-27'));
  check(!r.refused && r.pending === 0 && sent.length === 2 && sent[1].ms === 61000, `sent: ${JSON.stringify(r)}`);
  // a database without the ms column takes the score without it
  noMs = true;
  r = await sb.submit(score(4, '2026-09-27'));
  check(!r.refused && r.pending === 0 && sent.length === 3 && !('ms' in sent[2]), `no ms column: ${JSON.stringify(sent[2])}`);
  noMs = false;
}

// ---- the boards: 60 players over few point totals (ties), some without a time
const timedDay = TIME_FROM, oldDay = '2026-10-19';
let seed = 12345;
const rnd = (n) => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed % n; };
for (const day of [timedDay, oldDay]) {
  for (let i = 0; i < 60; i++) {
    views.push({ day, game: 'repplus', client_id: `p${i}`, player: `玩家${i}`, points: [3875, 3850, 3600, 0][rnd(4)], floors: 7,
      last_at: `2026-10-21T0${rnd(10)}:${String(rnd(60)).padStart(2, '0')}:00.${String(rnd(1000)).padStart(3, '0')}+00:00`,
      ms: rnd(5) === 0 ? null : 60000 * (3 + rnd(3)) });   // few distinct times: ties on time too
  }
}
const expected = (day, timed) => views.filter((r) => r.day === day).sort((a, b) => b.points - a.points
  || (timed ? (a.ms === null ? Infinity : a.ms) - (b.ms === null ? Infinity : b.ms) : 0)
  || (a.last_at < b.last_at ? -1 : a.last_at > b.last_at ? 1 : 0)).map((r) => r.client_id);
for (const [day, timed, missing] of [[timedDay, true, false], [oldDay, false, false], [timedDay, false, true]]) {
  noMs = missing;
  const sb = createScoreboard();
  const want = expected(day, timed);
  for (const me of want) {
    const b = await sb.board({ scope: 'today', game: 'repplus', day, limit: 20, me });
    check(b.timed === timed, `${day}: timed ${b.timed}`);
    check(b.rows.map((r) => r.client_id).join() === want.slice(0, 20).join(), `${day}: the top rows`);
    check(b.rank === want.indexOf(me) + 1 && b.mine.client_id === me, `${day} ${me}: place ${b.rank}, expected ${want.indexOf(me) + 1}`);
    check(missing ? !('ms' in b.mine) : 'ms' in b.mine, `${day}: ms ${JSON.stringify(b.mine)}`);
  }
}
noMs = false;

// ---- the moves and the replays
{
  const sb = createScoreboard();
  const day = '2026-10-21';
  const mv = (client, token, floor) => ({ client_id: client, token, game: 'repplus', day, floor, moves: `b${40 + floor} k${60 + floor}`, gave_up: floor === 6 });
  const ask = (token) => sb.replay({ game: 'repplus', day, player: 'pro', client: 'me', token });
  const refused = async (token) => { try { await ask(token); return false; } catch (err) { return !!err.notFinished; } };
  for (let f = 0; f < 7; f++) sb.queueMoves(mv('pro', 'tp', f));
  for (let f = 0; f < 6; f++) sb.queueMoves(mv('me', 'tm', f));
  sb.queueMoves(mv('me', 'tm', 0));                                  // queued once only
  net = 'down';
  check(await sb.flushMoves() === 13, 'offline: the moves wait');
  net = 'up';
  await sb.flush();
  check(moves.length === 13 && await sb.flushMoves() === 0, `sent: ${moves.length}`);
  check(await refused('tm'), 'six floors are not the day');
  sb.queueMoves(mv('me', 'tm', 6));
  await sb.flush();
  check(await refused('nope'), 'another token is not mine');
  const got = await ask('tm');
  check(got.length === 7 && got[0].moves === 'b40 k60' && got[6].gave_up === true && got.every((r, i) => r.floor === i), `replay ${JSON.stringify(got)}`);
  // a database without the table yet: the moves are dropped, the scores behind them still go
  noMovesTable = true;
  sb.queueMoves(mv('late', 'tl', 0));
  const before = sent.length;
  const r = await sb.submit({ client_id: 'late', player: 'p', game: 'repplus', mode: 'daily', day, seed: 9, floor: 0, points: 50, bombs: 1, keys: 0, hints: false, ms: 1000 });
  check(r.pending === 0 && sent.length === before + 1 && await sb.flushMoves() === 0, 'no hr_moves table');
  noMovesTable = false;
}
console.log('ok scoreboard queue, boards, moves and replays');
