// Where scores go. With MAPGEN_CONFIG.supabase = {url, key} (the project's URL and publishable key,
// see web/scoreboard.sql) they go to the shared Supabase table and everyone sees the same boards;
// without it they stay in this browser. One score per player, game, seed and floor: the first one.
// A board is ordered by points; a daily board from TIME_FROM on breaks ties by the time the floors
// took (ms, summed when every floor has one), and then, like every board, by who finished first.
// A daily floor's moves go along too (hr_moves), for the replays: a day's moves are handed out only to
// a player who has finished that day's challenge, proved by the secret token their own moves carry.
const CONFIG = window.MAPGEN_CONFIG || {};
const LOCAL = 'hr-scores-v1';
const OUTBOX = 'hr-outbox-v1';
const LOCAL_MOVES = 'hr-moves-v1';
const MOVES_OUT = 'hr-moves-outbox-v1';
export const DAILY_FLOORS = 7;
export const TIME_FROM = '2026-10-21';
export const timedDay = (day) => !!day && day >= TIME_FROM;

function readJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || '') ?? fallback; } catch { return fallback; }
}
function writeJson(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
const same = (a, b) => a.client_id === b.client_id && a.game === b.game && a.seed === b.seed && a.floor === b.floor;
const sameMoves = (a, b) => a.client_id === b.client_id && a.game === b.game && a.day === b.day && a.floor === b.floor;
const notFinished = () => Object.assign(new Error('这一天的挑战还没完成'), { notFinished: true });
// a day's moves for one player, if the asker has finished that day: [{floor, moves, gave_up}]
function movesFor(rows, { game, day, player, client, token }) {
  const mine = rows.filter((r) => r.game === game && r.day === day && r.client_id === client && r.token === token);
  if (mine.length < DAILY_FLOORS) throw notFinished();
  return rows.filter((r) => r.game === game && r.day === day && r.client_id === player).sort((a, b) => a.floor - b.floor)
    .map(({ floor, moves, gave_up }) => ({ floor, moves, gave_up }));
}

// rows -> [{client_id, player, points, floors, ms, last_at}] best first; the latest name per player
function aggregate(rows, timed) {
  const by = new Map();
  for (const r of rows) {
    const e = by.get(r.client_id) || { client_id: r.client_id, player: r.player, points: 0, floors: 0, ms: 0, last_at: '' };
    e.points += r.points;
    e.floors += 1;
    e.ms = e.ms === null || typeof r.ms !== 'number' ? null : e.ms + r.ms;
    if ((r.created_at || '') >= e.last_at) { e.player = r.player; e.last_at = r.created_at || ''; }
    by.set(r.client_id, e);
  }
  const ms = (e) => (e.ms === null ? Infinity : e.ms);
  return [...by.values()].sort((a, b) => b.points - a.points || (timed ? ms(a) - ms(b) : 0) || a.last_at.localeCompare(b.last_at));
}

// The PostgREST filter for the players ahead of `m` on a board ordered points desc, [ms asc nulls
// last,] last_at asc: their count + 1 is m's place
export function aheadOf(m, timed) {
  const at = `last_at.lt."${m.last_at}"`;
  const tie = (more) => `and(points.eq.${m.points},${more})`;
  const parts = [`points.gt.${m.points}`];
  if (!timed) parts.push(tie(at));
  else if (typeof m.ms === 'number') parts.push(tie(`ms.lt.${m.ms}`), tie(`ms.eq.${m.ms},${at}`));
  else parts.push(tie('ms.not.is.null'), tie(`ms.is.null,${at}`));
  return `(${parts.join(',')})`;
}

function localStore() {
  return {
    remote: false,
    async submit(entry) {
      const all = readJson(LOCAL, []);
      if (all.some((r) => same(r, entry))) return { duplicate: true };
      all.push({ ...entry, created_at: new Date().toISOString() });
      writeJson(LOCAL, all);
      return { duplicate: false };
    },
    async board({ scope, game, day, limit = 20 }) {
      const daily = scope !== 'total';
      const timed = daily && timedDay(day);
      const rows = readJson(LOCAL, []).filter((r) => r.game === game && (!daily || (r.mode === 'daily' && r.day === day)));
      const all = aggregate(rows, timed);
      return { rows: all.slice(0, limit), total: all.length, all, timed };
    },
    async submitMoves(entry) {
      const all = readJson(LOCAL_MOVES, []);
      if (!all.some((r) => sameMoves(r, entry))) all.push(entry);
      writeJson(LOCAL_MOVES, all);
    },
    async replay(q) { return movesFor(readJson(LOCAL_MOVES, []), q); },
  };
}

function supabaseStore({ url, key }) {
  const base = url.replace(/\/+$/, '') + '/rest/v1/';
  // a publishable key (sb_publishable_...) goes in apikey only; an old anon key is a JWT, sent as both
  const headers = key.startsWith('sb_') ? { apikey: key } : { apikey: key, Authorization: `Bearer ${key}` };
  async function get(path, extra = {}) {
    const res = await fetch(base + path, { headers: { ...headers, ...extra } });
    if (!res.ok) throw Object.assign(new Error(`排行榜读取失败（HTTP ${res.status}）`), { status: res.status });
    return res;
  }
  // false once the database turns out not to have the ms column yet (web/scoreboard.sql): scores then
  // go without it and the boards are read without it
  let hasMs = true;
  async function post(entry) {
    const res = await fetch(base + 'hr_scores', {
      method: 'POST', headers: { ...headers, 'Content-Type': 'application/json', Prefer: 'return=minimal' },
      body: JSON.stringify(entry),
    });
    if (res.status === 409) return { duplicate: true };
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      if (body.code === 'PGRST204' && 'ms' in entry) {          // no such column (yet)
        hasMs = false;
        const { ms, ...rest } = entry;
        return post(rest);
      }
      // an error from the database itself (a SQLSTATE code: a check, or the daily-date rule) refuses
      // this row for good; anything else (the network, the gateway, a key) may pass later
      const err = new Error(`分数提交失败（HTTP ${res.status}）`);
      err.refused = /^[0-9A-Z]{5}$/.test(String(body.code || ''));
      throw err;
    }
    return { duplicate: false };
  }
  async function read({ scope, game, day, limit, me }) {
    const daily = scope !== 'total';          // today's or yesterday's challenge: one day's board
    const view = daily ? 'hr_board_daily' : 'hr_board_total';
    const timed = daily && hasMs && timedDay(day);
    const select = `client_id,player,points,floors,last_at${daily && hasMs ? ',ms' : ''}`;
    const where = (q) => { q.set('game', `eq.${game}`); if (daily) q.set('day', `eq.${day}`); return q; };
    const order = timed ? 'points.desc,ms.asc.nullslast,last_at.asc' : 'points.desc,last_at.asc';
    const res = await get(`${view}?${where(new URLSearchParams({ select, order, limit: String(limit) }))}`, { Prefer: 'count=exact' });
    const rows = await res.json();
    const range = res.headers.get('content-range') || '';
    const total = parseInt(range.split('/')[1], 10) || rows.length;
    let mine = rows.find((r) => r.client_id === me) || null;
    let rank = mine ? rows.indexOf(mine) + 1 : null;
    if (me && !mine) {                     // outside the top rows: my total, then how many are ahead
      [mine] = await (await get(`${view}?${where(new URLSearchParams({ select, client_id: `eq.${me}` }))}`)).json();
      if (mine) {
        const aq = where(new URLSearchParams({ select: 'client_id', or: aheadOf(mine, timed), limit: '1' }));
        const ahead = await get(`${view}?${aq}`, { Prefer: 'count=exact' });
        rank = (parseInt((ahead.headers.get('content-range') || '').split('/')[1], 10) || 0) + 1;
      }
    }
    return { rows, total, mine: mine || null, rank, timed };
  }
  // moves: kept for later only when the network failed; anything the server says ends them
  async function postMoves(entry) {
    let res;
    try {
      res = await fetch(base + 'hr_moves', {
        method: 'POST', headers: { ...headers, 'Content-Type': 'application/json', Prefer: 'return=minimal' },
        body: JSON.stringify(entry),
      });
    } catch (err) {
      throw Object.assign(err, { network: true });
    }
    if (!res.ok && res.status !== 409) throw new Error(`操作记录提交失败（HTTP ${res.status}）`);
  }
  async function replay({ game, day, player, client, token }) {
    const res = await fetch(base + 'rpc/hr_replay', {
      method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' },
      body: JSON.stringify({ p_game: game, p_day: day, p_player: player, p_client: client, p_token: token }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      if (body.code === 'P0001') throw notFinished();
      throw new Error(`回看读取失败（HTTP ${res.status}）`);
    }
    return res.json();
  }
  return {
    remote: true,
    submit: post,
    submitMoves: postMoves,
    replay,
    async board({ scope, game, day, limit = 20, me = null }) {
      try {
        return await read({ scope, game, day, limit, me });
      } catch (err) {
        if (!(hasMs && scope !== 'total' && err.status === 400)) throw err;
        hasMs = false;                     // the daily view has no ms column yet
        return read({ scope, game, day, limit, me });
      }
    },
  };
}

export function createScoreboard() {
  const sb = CONFIG.supabase && CONFIG.supabase.url && CONFIG.supabase.key ? supabaseStore(CONFIG.supabase) : localStore();
  const store = {
    remote: sb.remote,
    // a score to send: queued first, so a failed send is retried the next time
    async submit(entry) {
      const outbox = readJson(OUTBOX, []);
      if (!outbox.some((r) => same(r, entry))) outbox.push(entry);
      writeJson(OUTBOX, outbox);
      const out = await store.flush();
      return { ...out, refused: out.refused.some((r) => same(r, entry)) };
    },
    // send what is queued, in order; one the database refuses is dropped (it would block the rest
    // forever), anything else stops the round and waits for the next. Then the moves.
    async flush() {
      let outbox = readJson(OUTBOX, []);
      let error = null;
      const refused = [];
      for (const entry of [...outbox]) {
        try {
          await sb.submit(entry);
        } catch (err) {
          if (!err.refused) { error = err; break; }
          refused.push(entry);
        }
        outbox = outbox.filter((r) => !same(r, entry));
        writeJson(OUTBOX, outbox);
      }
      await store.flushMoves();
      return { pending: outbox.length, error, refused };
    },
    // a daily floor's moves, queued like the scores
    queueMoves(entry) {
      const out = readJson(MOVES_OUT, []);
      if (!out.some((r) => sameMoves(r, entry))) out.push(entry);
      writeJson(MOVES_OUT, out);
    },
    async flushMoves() {
      let out = readJson(MOVES_OUT, []);
      for (const entry of [...out]) {
        try {
          await sb.submitMoves(entry);
        } catch (err) {
          if (err.network) break;
        }
        out = out.filter((r) => !sameMoves(r, entry));
        writeJson(MOVES_OUT, out);
      }
      return out.length;
    },
    replay: (q) => sb.replay(q),
    async board(opts) {
      const out = await sb.board(opts);
      if (!sb.remote && opts.me) {
        const i = out.all.findIndex((r) => r.client_id === opts.me);
        out.mine = i >= 0 ? out.all[i] : null;
        out.rank = i >= 0 ? i + 1 : null;
      }
      return out;
    },
  };
  return store;
}
