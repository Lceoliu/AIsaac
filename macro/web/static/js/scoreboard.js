// Where scores go. With MAPGEN_CONFIG.supabase = {url, key} (the project's URL and publishable key,
// see web/scoreboard.sql) they go to the shared Supabase table and everyone sees the same boards;
// without it they stay in this browser. One score per player, game, seed and floor: the first one.
const CONFIG = window.MAPGEN_CONFIG || {};
const LOCAL = 'hr-scores-v1';
const OUTBOX = 'hr-outbox-v1';

function readJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || '') ?? fallback; } catch { return fallback; }
}
function writeJson(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
const same = (a, b) => a.client_id === b.client_id && a.game === b.game && a.seed === b.seed && a.floor === b.floor;

// rows -> [{client_id, player, points, floors}] best first; the latest name per player
function aggregate(rows) {
  const by = new Map();
  for (const r of rows) {
    const e = by.get(r.client_id) || { client_id: r.client_id, player: r.player, points: 0, floors: 0, at: '' };
    e.points += r.points;
    e.floors += 1;
    if ((r.created_at || '') >= e.at) { e.player = r.player; e.at = r.created_at || ''; }
    by.set(r.client_id, e);
  }
  return [...by.values()].sort((a, b) => b.points - a.points || a.at.localeCompare(b.at));
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
      const rows = readJson(LOCAL, []).filter((r) => r.game === game && (scope === 'total' || (r.mode === 'daily' && r.day === day)));
      const all = aggregate(rows);
      return { rows: all.slice(0, limit), total: all.length, all };
    },
  };
}

function supabaseStore({ url, key }) {
  const base = url.replace(/\/+$/, '') + '/rest/v1/';
  // a publishable key (sb_publishable_...) goes in apikey only; an old anon key is a JWT, sent as both
  const headers = key.startsWith('sb_') ? { apikey: key } : { apikey: key, Authorization: `Bearer ${key}` };
  async function get(path, extra = {}) {
    const res = await fetch(base + path, { headers: { ...headers, ...extra } });
    if (!res.ok) throw new Error(`排行榜读取失败（HTTP ${res.status}）`);
    return res;
  }
  return {
    remote: true,
    async submit(entry) {
      const res = await fetch(base + 'hr_scores', {
        method: 'POST', headers: { ...headers, 'Content-Type': 'application/json', Prefer: 'return=minimal' },
        body: JSON.stringify(entry),
      });
      if (res.status === 409) return { duplicate: true };
      if (!res.ok) {
        // an error from the database itself (a SQLSTATE code: a check, or the daily-date rule) refuses
        // this row for good; anything else (the network, the gateway, a key) may pass later
        const body = await res.json().catch(() => ({}));
        const err = new Error(`分数提交失败（HTTP ${res.status}）`);
        err.refused = /^[0-9A-Z]{5}$/.test(String(body.code || ''));
        throw err;
      }
      return { duplicate: false };
    },
    async board({ scope, game, day, limit = 20, me = null }) {
      const daily = scope !== 'total';          // today's or yesterday's challenge: one day's board
      const view = daily ? 'hr_board_daily' : 'hr_board_total';
      const q = new URLSearchParams({ select: 'client_id,player,points,floors', game: `eq.${game}`, order: 'points.desc,last_at.asc', limit: String(limit) });
      if (daily) q.set('day', `eq.${day}`);
      const res = await get(`${view}?${q}`, { Prefer: 'count=exact' });
      const rows = await res.json();
      const range = res.headers.get('content-range') || '';
      const total = parseInt(range.split('/')[1], 10) || rows.length;
      let mine = rows.find((r) => r.client_id === me) || null;
      let rank = mine ? rows.indexOf(mine) + 1 : null;
      if (me && !mine) {                     // outside the top rows: my total, then how many are ahead
        const mq = new URLSearchParams({ select: 'client_id,player,points,floors', game: `eq.${game}`, client_id: `eq.${me}` });
        if (daily) mq.set('day', `eq.${day}`);
        [mine] = await (await get(`${view}?${mq}`)).json();
        if (mine) {
          const aq = new URLSearchParams({ select: 'client_id', game: `eq.${game}`, points: `gt.${mine.points}`, limit: '1' });
          if (daily) aq.set('day', `eq.${day}`);
          const ahead = await get(`${view}?${aq}`, { Prefer: 'count=exact' });
          rank = (parseInt((ahead.headers.get('content-range') || '').split('/')[1], 10) || 0) + 1;
        }
      }
      return { rows, total, mine, rank };
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
    // forever), anything else stops the round and waits for the next
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
      return { pending: outbox.length, error, refused };
    },
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
