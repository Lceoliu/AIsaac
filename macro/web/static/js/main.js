// 找隐藏房: the page. A home screen (player name, the challenges, the scoreboard) and the game: a run's
// floors (backend.js) whose hidden rooms the player finds on the game's minimap (play.js), with the
// linked posterior as optional hints (infer.js), points (score.js) and a shared board (scoreboard.js).
import { aiExpected, aiReplay, playerReplay } from './ai.js';
import { createBackend } from './backend.js';
import { celebrate, isMuted, isaacSprite, animateIsaac, loadSounds, perfectIcon, playSound, setMuted, stampPerfect } from './fx.js';
import {
  KIND_NAME, KIND_TYPE, best, infer, nextBomb, nextRedKey, pct, where,
} from './infer.js';
import {
  HEAT, boundsOf, css, drawLayout, drawMap, drawThumb, iconOnly, layoutArt, layoutSvg, openRoom, playEffect, roomSkeleton,
  tileIcon,
} from './minimap.js';
import {
  canBomb, canKey, foundKinds, hiddenRooms, isComplete, keyTargets, knownCells, movesText, newPlay, onlyUltraLeft, parseMoves, undo,
  useBomb, useKey,
} from './play.js';
import { loadArt, prepare } from './roomart.js';
import { MISS_POINTS, floorScore, maxScore, rulesFor, rulesOf, singleLinked, tierOf } from './score.js';
import { TIME_FROM, createScoreboard } from './scoreboard.js';

const $ = (id) => document.getElementById(id);
const GAMES = { abplus: '胎衣†', repplus: '忏悔+' };
const MODES = {
  daily: { name: '今日挑战', ranked: true, floors: 7 },
  random: { name: '随机挑战', ranked: true, floors: 7 },
  practice: { name: '自由练习', ranked: false, floors: null },
};
const PARAMS = ['mode', 'route', 'last', 'coins', 'keys', 'hearts', 'max_hearts', 'soul'];
const DEFAULTS = { mode: 'normal', route: 'sheol', last: '11', coins: '0', keys: '0', hearts: '6', max_hearts: '6', soul: '0' };
const STORE_TEXT = 'isaac-mapgen-text-labels';
const ONE_COLUMN = window.matchMedia('(max-width: 980px)');   // the panel sits below the map (style.css)
const FLOORS_OPEN = 'hr-floors-open';
const COACH = 'hr-coach-room';     // set once the player has opened a room's layout from the map
const CONFIRM = 'hr-confirm';      // '1' / '0': 二次确认 on or off; without a choice, on for touch screens
const COARSE = window.matchMedia('(pointer: coarse)');
const FLOOR_MS = 3600e3;           // a floor's clock stops at an hour (the database's cap)
const PROFILE = 'hr-profile-v1';
const PLAYS = 'hr-play-v1:';     // + game:seed:floor -> a ranked floor's play, kept across reloads
const RUNS = 'hr-run-v1:';       // + game:seed -> {mode, day, points: {floor: points}}
const WARM = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 13];   // room files, most used first
const LEGEND_TYPES = [5, 4, 2, 7, 8, 29, 10, 6, 11, 12, 13, 9, 20, 21, 18, 19, 24];
const TYPE_NAME = {
  2: '商店', 4: '宝箱房', 5: '头目房', 6: '小头目房', 7: '隐藏房', 8: '超级隐藏房', 9: '赌博房', 10: '诅咒房',
  11: '挑战房', 12: '图书馆', 13: '献祭房', 18: '卧室', 19: '卧室', 20: '宝库', 21: '骰子房', 24: '星象房', 29: '究极隐藏房',
};
const TYPE_COLOR = {
  2: '--t-shop', 4: '--t-treasure', 5: '--t-boss', 6: '--t-miniboss', 7: '--t-secret', 8: '--t-supersecret',
  9: '--t-arcade', 10: '--t-curse', 11: '--t-challenge', 12: '--t-library', 13: '--t-sacrifice', 18: '--t-bedroom',
  19: '--t-bedroom', 20: '--t-vault', 21: '--t-dice', 24: '--t-planetarium', 29: '--t-ultrasecret',
};

const state = {
  screen: 'home', mode: 'practice', backend: null, game: 'abplus', data: null, floor: 0, view: 'find',
  plays: new Map(), tool: 'bomb', hints: { heat: false, next: false }, text: false, sheets: new Map(), sheet: null,
  selected: null, layouts: new Map(), layoutJobs: new Map(), expected: new Map(), progress: new Map(), pending: 0,
  geo: null, k: 6, hover: null, tipSig: null, art: undefined, profile: null, scoreboard: null, scope: 'today',
  board: null, boardJob: 0, warmed: new Set(), engineReady: false, ai: null, aiCache: new Map(), aiTimer: null,
  day: null,        // a daily run's own day: its seed and its scores keep it after midnight
  started: null,    // the day this run was opened, and the scoring rules it plays by (score.js)
  rules: null,
  floorsOpen: false, // in one column the floor thumbnails fold into a row of numbers
  confirm: false,    // 二次确认: a tap on the map picks the cell, the button (or the same cell again) acts
  pendingAct: null,  // the cell picked: {cell, tool, floor}
  nudge: null,       // the tool the page has just switched to, its button pulses once
  watch: null,       // another player's replay: {id, name, floors: Map(floor -> {moves, gaveUp}), cache, paused}
};

// ---------------------------------------------------------------------------- small helpers
function html(tag, attrs = {}, text = null) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v === null || v === undefined) continue;
    e.setAttribute(k, v === true ? '' : v);
  }
  if (text !== null) e.textContent = text;
  return e;
}
function kv(rows) {
  const dl = html('dl', { class: 'kv' });
  for (const [k, v, cls] of rows) {
    dl.appendChild(html('dt', {}, k));
    const dd = html('dd', cls ? { class: cls } : {});
    if (v instanceof Node) dd.appendChild(v); else dd.textContent = v;
    dl.appendChild(dd);
  }
  return dl;
}
function readJson(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key) || '') ?? fallback; } catch { return fallback; }
}
function writeJson(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}
let statusTimer = null;
function flash(text) {
  setStatus(text);
  clearTimeout(statusTimer);
  statusTimer = setTimeout(() => setStatus(''), 2600);
}
function floor() { return state.data.floors[state.floor]; }
function rep() { return state.data && state.data.game === 'repplus'; }
function ranked() { return MODES[state.mode].ranked; }
function useIcons() { return !state.text && !!state.sheet; }
function ui() { return state.art ? state.art.ui : null; }
// the probabilities and the next-step hint: practice only (a challenge shows the AI once a floor is over)
function hintsOn() { return !ranked() && (state.hints.heat || state.hints.next); }

// ---------------------------------------------------------------------------- player, days, seeds
function uuid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  const b = crypto.getRandomValues(new Uint8Array(16));
  b[6] = (b[6] & 15) | 64;
  b[8] = (b[8] & 63) | 128;
  const h = [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}
function loadProfile() {
  const p = readJson(PROFILE, null) || {};
  // token: never shown; it proves to the replays which moves are this browser's own (scoreboard.js)
  const out = { id: p.id || uuid(), name: cleanName(p.name), token: p.token || uuid() };
  writeJson(PROFILE, out);
  return out;
}
function cleanName(s) {
  return String(s || '').replace(/[\u0000-\u001f\u007f<>]/g, '').trim().slice(0, 16);
}
function setName(name) {
  state.profile.name = cleanName(name);
  writeJson(PROFILE, state.profile);
  renderMe();
}
// the challenge day, China time: everyone gets the same daily seed
function today() {
  return new Date(Date.now() + 8 * 3600e3).toISOString().slice(0, 10);
}
// the day a daily run in the address belongs to: today's, or yesterday's while it can still be sent
function yesterday() {
  return new Date(Date.parse(`${today()}T00:00:00Z`) - 86400e3).toISOString().slice(0, 10);
}
function challengeDay(day) {
  return day === yesterday() ? day : today();
}
function dayName(day) { const [, mm, dd] = day.split('-'); return `${+mm} 月 ${+dd} 日`; }
function dailySeed(day, game) {
  let h = 0x811c9dc5;
  for (const b of new TextEncoder().encode(`找隐藏房:${day}:${game}`)) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h || 1;
}
function randomSeed() {
  const a = new Uint32Array(1);
  do crypto.getRandomValues(a); while (!a[0]);
  return a[0];
}

// ---------------------------------------------------------------------------- backend
function onProgress(m) {
  if (m.state === 'error') {
    setStatus('运行环境加载失败：' + m.text, true);
    $('engine').textContent = '运行环境加载失败：' + m.text;
    return;
  }
  const prev = state.progress.get(m.step) || {};
  state.progress.set(m.step, { state: m.state, text: m.text || prev.text });
  if (m.step === 'code' && m.state === 'done') state.engineReady = true;
  renderEngine();
  renderLoading();
}
function renderEngine() {
  const e = $('engine');
  if (state.backend.kind !== 'pyodide') { e.textContent = ''; return; }
  const running = [...state.progress.values()].find((p) => p.state === 'run');
  e.textContent = state.engineReady ? '' : running ? `准备中：${running.text}…` : '准备中…';
  e.classList.toggle('working', !state.engineReady);   // not 'busy': that is the board's badge
}
function renderLoading() {
  // the first run shows every step; later, only while a game's data is being loaded
  const loadingData = [...state.progress.entries()].some(([step, p]) => p.state === 'run' && step !== 'generate');
  const show = state.pending > 0 && (!state.data || loadingData);
  $('loading').hidden = !show;
  if (show) $('welcome').hidden = true;
  const steps = [...state.progress.entries()].sort(([a], [b]) => (a === 'generate') - (b === 'generate'));
  $('loading-steps').replaceChildren(...steps.map(([, p]) => html('li', { class: p.state }, p.text)));
}
async function call(method, params) {
  state.pending += 1;
  renderLoading();
  try {
    return await state.backend.call(method, params);
  } finally {
    state.pending -= 1;
    renderLoading();
  }
}
function loadSheet(game) {
  if (!state.sheets.has(game)) state.sheets.set(game, call('sprites', { game }).catch(() => null));
  return state.sheets.get(game);
}
// parse the game's room files one at a time while nothing else is asked of the worker
async function warmup(game) {
  if (state.backend.kind !== 'pyodide' || state.warmed.has(game)) return;
  state.warmed.add(game);
  for (const stage of WARM) {
    while (state.pending > 0) await sleep(250);
    try { await state.backend.call('warmup', { game, stage }); } catch { return; }
  }
}

// ---------------------------------------------------------------------------- saved progress
function playKey(i) { return `${PLAYS}${state.data.game}:${state.data.seed.value}:${i}`; }
function runKey(game, seed) { return `${RUNS}${game}:${seed}`; }
function play(i = state.floor) {
  if (!state.plays.has(i)) {
    const saved = ranked() ? readJson(playKey(i), null) : null;
    state.plays.set(i, saved ? {
      obs: new Map(saved.obs), bombs: saved.bombs, keys: saved.keys, log: saved.log || [], revealed: !!saved.revealed,
      submitted: !!saved.submitted,
      // the floor's clock (ms); a floor begun before the page kept one has none
      ms: typeof saved.ms === 'number' ? saved.ms : ((saved.log || []).length || saved.revealed ? null : 0),
      movesSent: !!saved.movesSent,
    } : { ...newPlay(), submitted: false, ms: 0 });
  }
  return state.plays.get(i);
}
function savePlay(i = state.floor) {
  if (!ranked() || !state.plays.has(i)) return;
  const p = state.plays.get(i);
  writeJson(playKey(i), { obs: [...p.obs], bombs: p.bombs, keys: p.keys, log: p.log, revealed: p.revealed,
    submitted: p.submitted, ms: p.ms, movesSent: p.movesSent });
}
// The moves of every finished daily floor of today and yesterday not sent yet (a daily score goes with
// them; floors played before the replays go now), queued once each.
function sendMoves() {
  for (const game of Object.keys(GAMES)) {
    for (const day of [yesterday(), today()]) {
      const seed = dailySeed(day, game);
      for (let i = 0; i < MODES.daily.floors; i++) {
        const key = `${PLAYS}${game}:${seed}:${i}`;
        const saved = readJson(key, null);
        if (!saved || !saved.submitted || saved.movesSent) continue;
        state.scoreboard.queueMoves({ client_id: state.profile.id, token: state.profile.token, game, day, floor: i,
          moves: movesText(saved.log || []), gave_up: !!saved.revealed });
        saved.movesSent = true;
        writeJson(key, saved);
        const live = state.data && state.data.game === game && state.data.seed.value === seed ? state.plays.get(i) : null;
        if (live) live.movesSent = true;
      }
    }
  }
}
function recordRun(i, points) {
  const key = runKey(state.data.game, state.data.seed.value);
  const rec = readJson(key, null) || { mode: state.mode, day: state.mode === 'daily' ? state.day : null, rules: state.rules.v, points: {} };
  rec.points[i] = points;
  writeJson(key, rec);
}
// the rules this run is scored by: those of the day it began; a random run already on record keeps
// the ones it began with
function runRules() {
  if (state.mode === 'daily') return rulesFor(state.day);
  if (state.mode === 'random') {
    const rec = readJson(runKey(state.data.game, state.data.seed.value), null);
    if (rec) return rulesOf(rec.rules || 1);
  }
  return rulesFor(state.started || today());
}
function scoreOf(i, p) { return floorScore(state.data.floors[i], p, i, state.rules); }
function runPoints() {
  if (!state.data) return 0;
  let total = 0;
  state.data.floors.forEach((f, i) => {
    const p = state.plays.get(i) || (ranked() ? play(i) : null);
    if (p && (p.log.length || p.revealed)) total += scoreOf(i, p).total;
  });
  return total;
}
function floorDone(i) {
  const p = state.plays.get(i) || (ranked() ? play(i) : null);
  return !!p && (p.revealed || isComplete(state.data.floors[i], p));
}
// the complete map gives the answer away: in a challenge, only once the floor is over
function mapOpen(i = state.floor) { return !ranked() || floorDone(i); }

// ---------------------------------------------------------------------------- the floor's clock
// A challenge floor's time, the board's tie-break (scoreboard.js): it runs while the floor is on screen,
// the page in front and the floor not done, and is kept with the floor's play across reloads. Stopped
// before the run's data changes (savePlay writes under the run's seed).
const clock = { i: null, at: 0 };
function clockWanted() {
  const on = state.screen === 'play' && !!state.data && ranked() && document.visibilityState === 'visible' && !floorDone(state.floor);
  return on ? state.floor : null;
}
function stopClock() {
  if (clock.i === null) return;
  const p = state.plays.get(clock.i);
  if (p && typeof p.ms === 'number') {
    p.ms = Math.min(FLOOR_MS, p.ms + (performance.now() - clock.at));
    savePlay(clock.i);
  }
  clock.i = null;
}
function syncClock() {
  const want = clockWanted();
  if (want === clock.i) return;
  stopClock();
  if (want !== null) { clock.i = want; clock.at = performance.now(); }
}
function clockText(ms) {
  const t = Math.round(ms / 1000);
  const h = Math.floor(t / 3600), m = Math.floor(t / 60) % 60, sec = String(t % 60).padStart(2, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
}
// the run's time so far: the floors done, when each has a clock
function runTime() {
  let total = 0;
  for (let i = 0; i < state.data.floors.length; i++) {
    const p = state.plays.get(i) || play(i);
    if (typeof p.ms !== 'number') return null;
    total += p.ms;
  }
  return total;
}

// ---------------------------------------------------------------------------- screens and URL
function showScreen(screen) {
  state.screen = screen;
  document.body.dataset.screen = screen;
  stopAi();
  $('home').hidden = screen !== 'home';
  $('app').hidden = screen !== 'play';
  hideTip();
  syncClock();
}
function writeUrl(push = false) {
  const q = new URLSearchParams();
  if (state.screen === 'home') {
    if (state.game !== 'abplus') q.set('game', state.game);
  } else {
    q.set('play', state.mode);
    q.set('game', state.game);
    if (state.mode !== 'daily') {
      const seed = state.data ? state.data.seed.text : $('seed').value.trim();
      if (seed) q.set('seed', seed);
    } else if (state.day) q.set('day', state.day);
    if (state.data) q.set('f', String(state.floor + 1));
    if (state.mode === 'practice') for (const k of PARAMS) if ($(k).value !== DEFAULTS[k]) q.set(k, $(k).value);
  }
  const url = q.toString() ? '?' + q.toString() : location.pathname;
  if (push) history.pushState(null, '', url); else history.replaceState(null, '', url);
}
function readUrl() {
  const q = new URLSearchParams(location.search);
  if (GAMES[q.get('game')]) state.game = q.get('game');
  const mode = q.get('play') || (q.get('seed') ? 'practice' : null);
  if (!mode || !MODES[mode]) return { screen: 'home' };
  for (const k of PARAMS) if (q.get(k) !== null) $(k).value = q.get(k);
  const f = parseInt(q.get('f') || '1', 10);
  return { screen: 'play', mode, seed: q.get('seed'), day: q.get('day'), floorIndex: Number.isFinite(f) ? f - 1 : 0 };
}
function goHome(push = true) {
  state.watch = null;
  showScreen('home');
  writeUrl(push);
  renderHome();
  loadBoard();
  window.scrollTo(0, 0);
}

// ---------------------------------------------------------------------------- start a run
// the name dialog: resolves true once a name is saved. Its buttons settle it directly; the close
// event (Esc) only as a fallback, since browsers may hold that event back in a background tab.
let nameAnswer = null;
function answerName(ok) {
  const done = nameAnswer;
  nameAnswer = null;
  if ($('namedlg').open) $('namedlg').close();
  if (done) done(ok);
}
async function askName() {
  const dlg = $('namedlg');
  $('dlgname').value = state.profile.name;
  $('dlgerror').textContent = '';
  dlg.showModal();
  $('dlgname').focus();
  return new Promise((resolve) => { nameAnswer = resolve; });
}
async function start(mode, { seed = null, day = null, floorIndex = 0, push = true } = {}) {
  if (MODES[mode].ranked && !state.profile.name && !(await askName())) return;
  stopClock();
  state.watch = null;
  state.mode = mode;
  state.view = 'find';
  state.tool = 'bomb';
  state.selected = null;
  state.pendingAct = null;
  state.data = null;
  showScreen('play');
  if (mode === 'practice') {
    if (seed) $('seed').value = seed;
    render();
    if ($('seed').value.trim()) await generate({ floorIndex, push });
    else writeUrl(push);
    return;
  }
  state.day = mode === 'daily' ? challengeDay(day) : null;
  state.started = today();
  const s = mode === 'daily' ? String(dailySeed(state.day, state.game)) : (seed || String(randomSeed()));
  await generate({ seed: s, floorIndex, push });
}
function runParams(seed) {
  if (state.mode === 'practice') {
    const out = { game: state.game, seed: $('seed').value.trim() };
    for (const k of PARAMS) out[k] = $(k).value;
    return out;
  }
  return { ...DEFAULTS, game: state.game, seed, mode: 'normal', floors: String(MODES[state.mode].floors) };
}
async function generate({ seed = null, keepFloor = false, floorIndex = null, push = false } = {}) {
  const params = runParams(seed);
  if (!params.seed) {
    setStatus('先输入种子，或者点骰子随机一个。', true);
    $('seed').focus();
    return;
  }
  $('go').disabled = true;
  $('busy').hidden = !state.data;
  $('welcome').hidden = true;
  setStatus('');
  state.progress.delete('generate');
  state.progress.set('generate', { state: 'run', text: '生成楼层' });   // listed last
  try {
    const [data, sheet] = await Promise.all([call('run', params), loadSheet(params.game)]);
    stopClock();
    state.data = data;
    state.rules = runRules();
    state.sheet = sheet;
    state.plays = new Map();
    state.expected = new Map();
    state.aiCache = new Map();
    stopAi();
    state.selected = null;
    const i = floorIndex !== null ? floorIndex : keepFloor ? state.floor : 0;
    state.floor = Math.max(0, Math.min(data.floors.length - 1, i));
    state.tool = floorTool();
    state.pendingAct = null;
    if (state.mode === 'practice') $('seed').value = data.seed.text;
    writeUrl(push);
    render();
  } catch (err) {
    setStatus('生成失败：' + err.message, true);
    render();
  } finally {
    state.progress.set('generate', { state: 'done', text: '生成楼层' });
    renderLoading();
    $('go').disabled = false;
    $('busy').hidden = true;
  }
}
async function randomRun() {
  $('seed').value = String(randomSeed());
  await generate({ push: true });
}

// ---------------------------------------------------------------------------- home
function renderMe() {
  $('me-name').textContent = state.profile.name || '起个名字';
  $('me').classList.toggle('unnamed', !state.profile.name);
  if (document.activeElement !== $('pname')) $('pname').value = state.profile.name;
}
function renderHome() {
  renderMe();
  for (const b of $('home-game').querySelectorAll('button')) b.setAttribute('aria-checked', String(b.dataset.game === state.game));
  for (const li of document.querySelectorAll('.rules-strip .rep-only')) li.hidden = state.game !== 'repplus';
  const v2 = rulesFor(today()).v === 2;                       // the rules of a run begun today
  $('rules-v1').hidden = v2;
  $('rules-v2').hidden = !v2;
  const day = today();
  const [, mm, dd] = day.split('-');
  $('daily-meta').textContent = `${+mm} 月 ${+dd} 日 · ${MODES.daily.floors} 层 · 所有人同一局`;
  $('random-meta').textContent = `${MODES.random.floors} 层 · 随机种子 · 计入总榜`;
  const rec = readJson(runKey(state.game, dailySeed(day, state.game)), null);
  const done = rec ? Object.keys(rec.points).length : 0;
  const pts = rec ? Object.values(rec.points).reduce((a, b) => a + b, 0) : 0;
  $('daily-state').textContent = !rec ? '' : done >= MODES.daily.floors ? `今天已完成 · ${pts} 分` : `已完成 ${done}/${MODES.daily.floors} 层 · ${pts} 分，点这里继续`;
  for (const t of $('board').parentElement.querySelectorAll('.tabs button')) t.setAttribute('aria-selected', String(t.dataset.scope === state.scope));
  $('time-note').textContent = `${today() < TIME_FROM ? `${dayName(TIME_FROM)}起，` : ''}只算页面在前台的时间`;
  renderEngine();
}
function drawIcons() {
  const icons = ui() && ui().icons;
  if (!icons) return;
  for (const el of document.querySelectorAll('[data-icon]')) {
    const ico = icons[el.dataset.icon];
    if (!ico || el.firstChild) continue;
    el.appendChild(html('img', { src: 'art/' + ico.file, alt: '', class: 'pixel', width: ico.w, height: ico.h }));
  }
}
function drawBrand() {
  if (!ui() || !ui().happy) return;
  $('brand-isaac').replaceChildren(isaacSprite(ui(), 1.5));
  const hero = isaacSprite(ui(), 4, 0);
  $('hero-isaac').replaceChildren(hero);
  let busy = false;
  const cheer = () => { if (!busy) { busy = true; animateIsaac(hero, ui(), 4).then(() => { busy = false; }); } };
  $('hero-isaac').onmouseenter = cheer;
  setTimeout(cheer, 500);
  $('loading-isaac').replaceChildren(isaacSprite(ui(), 3, 0));
  drawIcons();
}
async function loadBoard() {
  const job = ++state.boardJob;
  const list = $('board');
  list.setAttribute('aria-busy', 'true');
  if (!state.board) list.replaceChildren(...Array.from({ length: 5 }, () => html('li', { class: 'ghostrow' })));
  const day = state.scope === 'today' ? today() : state.scope === 'yesterday' ? yesterday() : null;
  $('board-sub').textContent = day ? `${day} · ${GAMES[state.game]} · ${state.scope === 'today' ? '今日' : '昨日'}挑战`
    : `${GAMES[state.game]} · 所有挑战的积分`;
  try {
    const res = await state.scoreboard.board({ scope: state.scope, game: state.game, day, limit: 20, me: state.profile.id });
    if (job !== state.boardJob) return;
    state.board = res;
    renderBoardList();
  } catch (err) {
    if (job !== state.boardJob) return;
    list.replaceChildren(html('li', { class: 'board-empty' }, '排行榜暂时连不上：' + err.message));
  } finally {
    list.removeAttribute('aria-busy');
  }
}
// one row of the board: the top three get the Challenge Trophy (gold, silver, bronze) after the name;
// a daily board that breaks ties by time shows it
function boardRow(r, rank, timed, watch = null) {
  const li = html('li', { class: r.client_id === state.profile.id ? 'me' : '' });
  if (watch) {                                   // a day this browser has finished: the row opens the player's replay
    li.classList.add('watch');
    Object.assign(li, { tabIndex: 0, title: `回看 ${r.player} 是怎么找的` });
    li.setAttribute('role', 'button');
    li.addEventListener('click', () => watchPlayer(r, watch));
    li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); watchPlayer(r, watch); } });
  }
  const who = html('span', { class: 'who' });
  who.appendChild(html('span', { class: 'name' }, r.player));
  const cup = rank <= 3 && r.points > 0 && ui() && ui().trophies ? ui().trophies[rank - 1] : null;
  if (cup) who.appendChild(html('img', { src: 'art/' + cup.file, alt: '', class: 'trophy pixel', width: cup.w, height: cup.h }));
  const meta = timed && typeof r.ms === 'number' ? `${r.floors} 层 · ${clockText(r.ms)}` : `${r.floors} 层`;
  li.append(html('span', { class: `rank r${rank}` }, String(rank)), who, html('span', { class: 'floors' }, meta),
    html('b', { class: 'pts num' }, String(r.points)));
  return li;
}
// the day a daily board is about, and whether this browser has finished that day's challenge (the
// replays of the day open then: before, they would give the answers away)
function boardDay() {
  return state.scope === 'today' ? today() : state.scope === 'yesterday' ? yesterday() : null;
}
function finishedDaily(day, game) {
  const rec = readJson(runKey(game, dailySeed(day, game)), null);
  return !!rec && Object.keys(rec.points).length >= MODES.daily.floors;
}
function renderBoardList() {
  const { rows, mine, rank, timed } = state.board;
  const list = $('board');
  const day = boardDay();
  const watch = day && finishedDaily(day, state.game) ? day : null;
  $('board-tip').textContent = !day || !rows.length ? '' : watch ? '点名字，回看 TA 每一层是怎么找的'
    : `完成${state.scope === 'today' ? '今天' : '昨天'}的今日挑战后，可以点名字回看大家是怎么找的`;
  if (!rows.length) {
    const empty = { today: '今天还没有人上榜，来当第一个。', yesterday: '昨天的今日挑战没有人上榜。' }[state.scope] || '还没有成绩，来当第一个。';
    list.replaceChildren(html('li', { class: 'board-empty' }, empty));
  } else {
    const items = rows.map((r, i) => boardRow(r, i + 1, timed, watch));
    // outside the top rows: my own row at the bottom, after a gap
    if (mine && rank > rows.length) items.push(html('li', { class: 'gap', 'aria-hidden': 'true' }, '⋮'), boardRow(mine, rank, timed, watch));
    list.replaceChildren(...items);
  }
  $('board-me').textContent = mine ? `你排第 ${rank} 名 · ${mine.points} 分`
    : !state.profile.name ? '起个名字，成绩就会上榜' : state.scope === 'yesterday' ? '你昨天没有成绩' : '你还没有上榜的成绩';
  $('board-note').textContent = state.scoreboard.remote ? '' : '在线排行榜还没接上，现在只显示这台设备上的成绩。';
}

// ---------------------------------------------------------------------------- render the game
function render() {
  const has = !!state.data;
  $('app').classList.toggle('ready', has);
  $('welcome').hidden = has || state.pending > 0 || state.mode !== 'practice';
  $('floors').hidden = !has;
  $('floorbar').hidden = !has;
  $('maptools').hidden = true;                     // renderMapTools shows it when it applies
  $('app').classList.toggle('floors-open', state.floorsOpen);
  $('boardhead').hidden = !has;
  $('boardfoot').hidden = !has;
  $('panel').hidden = !has;
  $('map').style.display = has ? '' : 'none';
  renderPlaybar();
  syncClock();                                     // first: a floor just done shows its whole time
  if (!has) return;
  if (state.view === 'map' && !mapOpen()) state.view = 'find';
  $('t-text').checked = state.text;
  $('t-text').disabled = !state.sheet;
  $('t-confirm').checked = state.confirm;
  renderFloors();
  renderFloorbar();
  renderBoard();
  renderPanel();
  renderMapTools();
  renderLegend();
  prefetchFloor();
  state.nudge = null;
}
function renderPlaybar() {
  const m = MODES[state.mode];
  const name = state.mode === 'daily' && state.day && state.day !== today() ? `${dayName(state.day)}挑战` : m.name;
  $('mode-badge').textContent = `${name} · ${state.data ? GAMES[state.data.game] : GAMES[state.game]}`;
  $('mode-badge').dataset.mode = state.mode;
  $('seedform').hidden = state.mode !== 'practice';
  $('seed-tag').hidden = state.mode === 'practice' || !state.data;
  $('seed-tag').textContent = state.data ? state.data.seed.text : '';
  for (const b of $('play-game').querySelectorAll('button')) b.setAttribute('aria-checked', String(b.dataset.game === state.game));
  const n = state.data ? state.data.floors.length : 0;
  const done = state.data ? state.data.floors.filter((_, i) => floorDone(i)).length : 0;
  $('run-progress').textContent = state.data && m.ranked ? `第 ${state.floor + 1}/${n} 层 · 完成 ${done}` : '';
  $('run-score').hidden = !m.ranked || !state.data;
  if (m.ranked && state.data) $('run-points').textContent = String(runPoints());
}

function floorStatus(i) {
  const f = state.data.floors[i];
  if (state.view === 'map' && !ranked()) {
    return { text: [`${f.rooms.length} 间`].concat(f.curses).join(' · '), done: false };
  }
  const p = state.plays.get(i) || (ranked() ? play(i) : null);
  const total = hiddenRooms(f).length;
  const found = p ? foundKinds(p).size : 0;
  if (ranked() && p && (p.revealed || found >= total)) return { text: `${scoreOf(i, p).total} 分`, done: true };
  if (p && p.revealed) return { text: '已揭晓', done: false };
  return { text: found >= total ? '全找到了' : `${found}/${total}`, done: found >= total };
}

function renderFloors() {
  const nav = $('floors');
  nav.replaceChildren(...state.data.floors.map((f, i) => {
    const btn = html('button', { type: 'button', class: 'floor', 'aria-current': String(i === state.floor), title: f.name });
    const mini = html('span', { class: 'mini' });
    const p = state.plays.get(i) || (ranked() ? play(i) : null);
    const view = state.view === 'map' && mapOpen(i) ? 'map' : 'find';
    mini.appendChild(drawThumb(f, state.sheet, useIcons() ? 'icons' : 'text', view, p ? p.obs : null, p && p.revealed));
    const st = floorStatus(i);
    btn.append(mini, html('span', { class: 'fname' }, f.name), html('span', { class: st.done ? 'fmeta done' : 'fmeta' }, st.text));
    btn.addEventListener('click', () => goFloor(i));
    return btn;
  }));
  const cur = nav.children[state.floor];            // keep the current floor in view, horizontally only
  if (cur) {
    const left = cur.offsetLeft - nav.offsetLeft, right = left + cur.offsetWidth;
    if (left < nav.scrollLeft) nav.scrollLeft = left - 8;
    else if (right > nav.scrollLeft + nav.clientWidth) nav.scrollLeft = right - nav.clientWidth + 8;
  }
}

function viewBounds(f) {
  if (state.view === 'map') return boundsOf(f.rooms.flatMap((r) => r.cells), 1);
  // the explored rooms only, with room for candidates: 2 cells for the ultra secret room
  return boundsOf(f.rooms.filter((r) => !r.hidden).flatMap((r) => r.cells), rep() ? 2 : 1);
}

function scaleFor(b) {
  const frame = $('frame');
  const pad = window.innerWidth <= 560 ? 20 : 36;
  const w = Math.max(200, frame.clientWidth - pad);
  const h = Math.max(260, Math.min(window.innerHeight * 0.74, 760));
  return Math.max(2, Math.min(8, Math.floor(w / (b.cols * 9)), Math.floor(h / (b.rows * 8))));
}
function redKeyIcon() {
  const k = ui() && ui().redkey;
  return k ? { url: 'art/' + k.file, w: k.w, h: k.h } : null;
}

function renderBoard() {
  for (const el of document.querySelectorAll('.coach, .confirm')) el.remove();
  const f = floor();
  const pend = state.pendingAct;
  if (pend && (pend.floor !== state.floor || pend.tool !== state.tool || state.view !== 'find' || aiShown())) state.pendingAct = null;
  const title = $('floortitle');
  title.replaceChildren(document.createTextNode(f.name), html('small', {}, f.name_en));
  for (const b of document.querySelectorAll('.board-head .tabs button')) {
    b.setAttribute('aria-selected', String(b.dataset.view === state.view));
    if (b.dataset.view === 'map') {
      const locked = !mapOpen();
      b.disabled = locked;
      b.title = locked ? '找完或放弃这一层后才能看完整地图' : '';
      b.classList.toggle('locked', locked);
    }
  }
  const b = viewBounds(f);
  const k = scaleFor(b);
  state.k = k;
  const base = { k, style: useIcons() ? 'icons' : 'text', sheet: state.sheet, bounds: b, onHover: showTip, onLeave: hideTip,
    redKey: redKeyIcon() };
  hideTip();
  if (state.view === 'map') {
    state.geo = drawMap($('map'), f, { ...base, view: 'map', selected: state.selected, onRoom: pickRoom });
    return;
  }
  if (replayShown()) { renderAiBoard(f, base); return; }
  const p = play();
  const done = isComplete(f, p) || p.revealed;
  if (done) state.pendingAct = null;
  const inf = infer(f.joint, p.obs);
  const suggest = hintsOn() && state.hints.next && !done ? {
    bomb: state.tool === 'bomb' ? nextBomb(inf, p.obs, (c) => canBomb(f, p, c)) : null,
    key: state.tool === 'key' ? nextRedKey(inf, p.obs, keyTargets(f, p)) : null,
  } : null;
  state.geo = drawMap($('map'), f, {
    ...base, view: 'find', obs: p.obs, revealed: p.revealed, selected: state.selected,
    heat: hintsOn() && state.hints.heat && !done && inf.ok ? inf : null, suggest,
    actionable: done ? () => false : (c) => (state.tool === 'bomb' ? canBomb(f, p, c) : canKey(f, p, c)),
    onCell: (c) => act(c), onRoom: pickRoom,
  });
  if (state.pendingAct) renderConfirm(f);
  else renderCoach(f);
}

// Folded floors (one column): the floors as a row of numbers, the thumbnails one tap away.
function renderFloorbar() {
  const pills = html('div', { class: 'fpills' });
  state.data.floors.forEach((f, i) => {
    const st = floorStatus(i);
    const b = html('button', { type: 'button', class: st.done ? 'fpill done' : 'fpill', 'aria-current': String(i === state.floor),
      title: `${f.name} · ${st.text}`, 'aria-label': `第 ${i + 1} 层 ${f.name}，${st.text}` }, String(i + 1));
    b.addEventListener('click', () => goFloor(i));
    pills.appendChild(b);
  });
  const more = html('button', { type: 'button', class: 'fmore', 'aria-expanded': String(state.floorsOpen), 'aria-controls': 'floors' },
    state.floorsOpen ? '收起 ▴' : '缩略图 ▾');
  more.addEventListener('click', () => {
    state.floorsOpen = !state.floorsOpen;
    try { localStorage.setItem(FLOORS_OPEN, state.floorsOpen ? '1' : '0'); } catch { /* storage unavailable */ }
    render();
  });
  $('floorbar').replaceChildren(pills, more);
  const cur = pills.children[state.floor];         // keep the current floor in view, horizontally only
  if (cur && (cur.offsetLeft < pills.scrollLeft || cur.offsetLeft + cur.offsetWidth > pills.scrollLeft + pills.clientWidth)) {
    pills.scrollLeft = cur.offsetLeft - 8;
  }
}

// The bomb and the Red Key right under the map, so switching needs no scrolling (one column, Rep+;
// style.css hides it on wide screens, where the panel with the tools is beside the map).
function toolButton(id, label, count, disabled) {
  const b = html('button', { type: 'button', 'aria-pressed': String(state.tool === id), title: `${label}（${id === 'bomb' ? 'B' : 'K'}）`, disabled,
    class: state.nudge === id ? 'nudge' : false });
  b.appendChild(toolIcon(id));
  b.append(html('span', {}, label), html('b', { class: 'count', title: '已用' }, String(count)));
  b.addEventListener('click', () => { state.tool = id; render(); });
  return b;
}
// the tool a floor starts with: the bomb, or the Red Key once only the ultra secret room is left
function floorTool() {
  return rep() && onlyUltraLeft(floor(), play()) ? 'key' : 'bomb';
}
function renderMapTools() {
  const box = $('maptools');
  const f = floor(), p = play();
  const show = state.view === 'find' && !aiShown() && rep() && !isComplete(f, p) && !p.revealed;
  box.hidden = !show;
  $('app').classList.toggle('has-maptools', show);
  box.replaceChildren(...(show ? [toolButton('bomb', '炸弹', p.bombs, false), toolButton('key', '红钥匙', p.keys, false)] : []));
}

// The first time on a phone: a ring on a room and a note that a tap shows its layout (the panel that
// says so is below the map). Gone for good once a room is opened, or the note is tapped.
function renderCoach(f) {
  for (const el of document.querySelectorAll('.coach')) el.remove();
  let seen = true;
  try { seen = localStorage.getItem(COACH) === '1'; } catch { /* storage unavailable: never nag */ }
  if (seen || !ONE_COLUMN.matches || !state.geo) return;
  const room = f.rooms.find((r) => !r.hidden && r.type === 5) || f.rooms.find((r) => r.start);
  if (!room) return;
  const map = $('map'), frame = $('frame');
  const mr = map.getBoundingClientRect(), fr = frame.getBoundingClientRect();
  const scale = mr.width / (parseFloat(map.getAttribute('width')) || mr.width);
  const pts = room.cells.map((c) => state.geo.pos(c));
  const cx = pts.reduce((a, q) => a + q[0], 0) / pts.length + state.geo.Sx / 2;
  const cy = pts.reduce((a, q) => a + q[1], 0) / pts.length + state.geo.Sy / 2;
  const x = mr.left - fr.left + cx * scale, y = mr.top - fr.top + cy * scale;
  const ring = html('span', { class: 'coach ring', style: `left:${x}px;top:${y}px` });
  const below = y < 90;
  const note = html('button', { type: 'button', class: below ? 'coach note below' : 'coach note',
    style: `left:${Math.max(110, Math.min(fr.width - 110, x))}px;top:${below ? y + 30 : y - 30}px` }, '点房间，看它的布局和门位');
  note.appendChild(html('b', { 'aria-hidden': 'true' }, '✕'));
  note.addEventListener('click', dismissCoach);
  frame.append(ring, note);
}
function dismissCoach() {
  try { localStorage.setItem(COACH, '1'); } catch { /* storage unavailable */ }
  for (const el of document.querySelectorAll('.coach')) el.remove();
}

// 二次确认: the picked cell framed on the map, and next to it what will happen with a button to do it.
// The same cell tapped again does it too; another cell moves the pick; Esc or 取消 drops it.
function renderConfirm(f) {
  const pend = state.pendingAct;
  const map = $('map'), frame = $('frame');
  const [x, y] = state.geo.pos(pend.cell);
  const k = state.k, key = pend.tool === 'key';
  const mark = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  for (const [a, v] of Object.entries({ x: x + k * 0.4, y: y + k * 0.4, width: state.geo.Sx - k * 0.8, height: state.geo.Sy - k * 0.8,
    rx: k, 'stroke-width': Math.max(2, k * 0.6), 'stroke-dasharray': `${k * 1.6} ${k}`, class: key ? 'pending key' : 'pending' })) {
    mark.setAttribute(a, v);
  }
  map.appendChild(mark);
  const box = html('div', { class: 'confirm', role: 'group', 'aria-label': '确认这一步' });
  const q = html('span', { class: 'q' });
  q.append(html('b', {}, key ? '用红钥匙开这个门？' : '炸这里？'), html('small', {}, where(pend.cell, f.start)));
  const no = html('button', { type: 'button' }, '取消');
  no.addEventListener('click', () => { state.pendingAct = null; renderBoard(); });
  const yes = html('button', { type: 'button', class: 'primary' }, key ? '开门' : '炸');
  yes.addEventListener('click', () => act(pend.cell, true));
  box.append(q, no, yes);
  frame.appendChild(box);
  // above the cell, or below it when there is no room above; inside the frame
  const mr = map.getBoundingClientRect(), fr = frame.getBoundingClientRect();
  const scale = mr.width / (parseFloat(map.getAttribute('width')) || mr.width);
  const cx = mr.left - fr.left + (x + state.geo.Sx / 2) * scale;
  const top = mr.top - fr.top + y * scale, bottom = top + state.geo.Sy * scale;
  const w = box.offsetWidth, h = box.offsetHeight;
  const above = top - h - 8 >= 4;
  box.style.left = `${Math.max(4, Math.min(fr.width - w - 4, cx - w / 2))}px`;
  box.style.top = `${above ? top - h - 8 : Math.min(fr.height - h - 4, bottom + 8)}px`;
  box.classList.toggle('below', !above);
  box.style.setProperty('--arrow', `${Math.max(14, Math.min(w - 14, cx - (parseFloat(box.style.left) || 0)))}px`);
  yes.focus({ preventScroll: true });
}

// What a room's door slots show. In the map view every door is known. In the find view a door is
// seen only into a room the player knows (explored, found, or a red room); with the Red Key in hand,
// the slots it can open show as red doorways. Nothing else: a slot without a door is wall in the game.
function doorState(r) {
  if (state.view === 'map') return (bit) => ((r.doors & bit) ? 'open' : 'wall');
  const f = floor(), p = play();
  const known = knownCells(f, p);
  const keyable = state.tool === 'key' && rep() && !(p.revealed || isComplete(f, p)) ? new Set(keyTargets(f, p)) : null;
  return (bit) => {
    const target = r.slot_targets ? r.slot_targets[31 - Math.clz32(bit)] : -1;
    if (target >= 0 && known.has(target)) return 'open';
    if (keyable && target >= 0 && keyable.has(target)) return 'key';
    return 'wall';
  };
}

// ---------------------------------------------------------------------------- find mode
function act(cell, confirmed = false) {
  if (!state.data || aiShown()) return;         // a map from before a new run, or the AI's replay
  const f = floor();
  const p = play();
  if (p.revealed || isComplete(f, p)) return;
  const bomb = state.tool === 'bomb';
  if (bomb ? !canBomb(f, p, cell) : !canKey(f, p, cell)) {
    flash(bomb ? '这里挨不着任何房间，炸不到。' : '红钥匙只能用在房间的门位外面。');
    if (state.pendingAct) { state.pendingAct = null; renderBoard(); }
    return;
  }
  // 二次确认: the first tap picks the cell; the button, or the same cell again, does it
  const pend = state.pendingAct;
  if (state.confirm && !confirmed && !(pend && pend.cell === cell && pend.tool === state.tool && pend.floor === state.floor)) {
    state.pendingAct = { cell, tool: state.tool, floor: state.floor };
    hideTip();
    renderBoard();
    return;
  }
  state.pendingAct = null;
  let entry, note = null;
  if (bomb) {
    entry = useBomb(f, p, cell);
  } else {
    entry = useKey(f, p, cell);
    if (entry.tool === 'bomb') note = `${KIND_NAME[entry.found[0]]}按炸弹算：记一颗炸弹`;   // right on a (super) secret room (play.js)
  }
  // the tool for what is left: the Red Key once only the ultra secret room is; else a key that hit a
  // (super) secret room goes back to the bomb
  if (entry.found.length && rep() && onlyUltraLeft(f, p)) {
    if (state.tool !== 'key') {
      state.tool = 'key';
      state.nudge = 'key';
      note = [note, '只剩究极隐藏房了，已换成红钥匙'].filter(Boolean).join('；');
    } else if (note) {
      note += '；只剩究极隐藏房了，继续用红钥匙';
    }
  } else if (entry.tool === 'bomb' && state.tool === 'key') {
    state.tool = 'bomb';
    note += '，已换回炸弹';
  }
  savePlay();
  render();
  playEffect($('map'), state.geo, entry.cell, entry.found.length > 0 || entry.tool === 'key', state.k);
  feedback(entry, note);
  if (isComplete(f, p)) finishFloor();
}

// the found room pops with Isaac's thumbs up and the jingle; a miss shows what it cost. The status line
// says what this action alone earned besides (and clears what an earlier one said).
function feedback(entry, note = null) {
  const map = $('map'), frame = $('frame');
  const r = map.getBoundingClientRect(), fr = frame.getBoundingClientRect();
  const scale = r.width / (parseFloat(map.getAttribute('width')) || r.width);
  const [x, y] = state.geo.pos(entry.cell);
  const px = r.left - fr.left + (x + state.geo.Sx / 2) * scale, py = r.top - fr.top + (y + state.geo.Sy / 2) * scale;
  const sc = scoreOf(state.floor, play());
  const pts = sc.per[sc.per.length - 1];
  if (entry.found.length) {
    playSound('secret');
    celebrate($('fx'), ui(), px, py, ranked() ? `+${pts}` : KIND_NAME[entry.found[0]]);
    const extra = [note, bonusText(sc.parts[sc.parts.length - 1])].filter(Boolean).join('；');
    if (extra) flash(extra); else setStatus('');
  } else {
    setStatus('');
    celebrate($('fx'), ui(), px, py, ranked() ? `${pts}` : '空的', false);
  }
}

async function finishFloor() {
  const f = floor(), p = play(), i = state.floor;
  const sc = scoreOf(i, p);
  if (sc.complete && !p.revealed) setTimeout(() => playSound('thumbsup', 0.9), 900);
  if (sc.clean) setTimeout(() => stampPerfect($('fx'), ui(), `一次不空 +${sc.bonus}`), 700);   // the "A+" paper
  if (ranked() && !p.submitted) {
    p.submitted = true;
    savePlay(i);
    recordRun(i, sc.total);
    if (state.mode === 'daily') sendMoves();       // sent with the score just below
    render();
    const res = await state.scoreboard.submit({
      client_id: state.profile.id, player: state.profile.name || '无名', game: state.data.game, mode: state.mode,
      day: state.mode === 'daily' ? state.day : null, seed: state.data.seed.value, floor: i, points: sc.total,
      bombs: p.bombs, keys: p.keys, hints: false, ms: typeof p.ms === 'number' ? Math.round(Math.min(FLOOR_MS, p.ms)) : null,
    });
    if (res.refused) flash('这一局已经过了提交期限，这层成绩没有计入排行榜。');
    else if (res.error) flash('成绩先存在本机，下次联网时会自动补交。');
    else if (state.mode === 'daily' && state.day !== today()) flash(`已过零点：这一局的成绩仍算在 ${dayName(state.day)}的挑战里。`);
    state.board = null;
  } else {
    render();
  }
}

async function giveUp() {
  const p = play();
  const sc = scoreOf(state.floor, p);
  const ok = !ranked() || confirm(`放弃这一层？按现在找到的计 ${sc.total} 分，之后不能再炸，并会揭晓答案。`);
  if (!ok) return;
  p.revealed = true;
  savePlay();
  if (ranked()) await finishFloor();
  else render();
}

function logText(entry, f) {
  const at = where(entry.cell, f.start);
  const found = entry.found.map((k) => KIND_NAME[k]).join('、');
  if (entry.tool === 'bomb') return [`炸 ${at}：`, found ? `找到${found}` : '空的', !!found];
  if (entry.added[0][1] !== 'red') return [`红钥匙 ${at}：`, `门后是${found}`, true];
  return [`红钥匙 ${at}：`, found ? `开出红房间，通到${found}` : '开出红房间，没有通到隐藏房', !!found];
}

function adviceText(f, p, inf) {
  if (p.revealed) return ['答案已揭晓。', true];
  if (isComplete(f, p)) return ['这一层的隐藏房全找到了。', true];
  if (!inf.ok) return ['这一层有推算模型没考虑到的情况，暂时给不出概率。', true];
  if (!hintsOn() || !state.hints.next) {
    return [state.tool === 'key' ? '选一个房间外的门位，用红钥匙开门。房间布局里的红色虚线门就是能开的地方。门后要是隐藏房或超级隐藏房，按一颗炸弹算。' : '点挨着房间的空格子，用炸弹炸开那面墙。', true];
  }
  if (state.tool === 'key') {
    if (foundKinds(p).has('ultra')) return ['究极隐藏房已经找到了。剩下的用炸弹找。', true];
    const s = nextRedKey(inf, p.obs, keyTargets(f, p));
    return s ? [`下一步：在${where(s[0], f.start)}开红房间，${pct(s[1])} 能通到究极隐藏房。`, false] : ['红钥匙暂时没有好的位置。', true];
  }
  const s = nextBomb(inf, p.obs, (c) => canBomb(f, p, c));
  if (!s) return ['没有可以炸的候选格了。', true];
  const left = ['secret', 'super'].filter((k) => !foundKinds(p).has(k) && hiddenRooms(f).some((h) => h.kind === k));
  return [`下一步：炸${where(s[0], f.start)}，${pct(s[1])} 能炸出${left.map((k) => KIND_NAME[k]).join('或')}。`, false];
}

// the bonuses of one find, as a line (null when it earned the room's points only)
function bonusText(parts) {
  const out = (parts || []).filter(([label]) => label === 'first' || label === 'link')
    .map(([label, v, kind]) => (label === 'first' ? `一次就找到 +${v}` : `单连${KIND_NAME[kind]} +${v}`));
  return out.length ? out.join(' · ') : null;
}
function scoreLine(f, p) {
  const sc = scoreOf(state.floor, p);
  const box = html('div', { class: 'scoreline' });
  box.append(html('span', { class: 'lbl' }, ranked() ? '本层得分' : '本层得分（练习）'), html('b', { class: 'num' }, String(sc.total)));
  const bits = [];
  if (sc.misses) bits.push(`空 ${sc.misses} 次 −${sc.misses * MISS_POINTS}`);
  if (sc.firstPts) bits.push(`一次就找到 ${sc.firsts} 个 +${sc.firstPts}`);
  if (sc.linkPts) bits.push(`单连 +${sc.linkPts}`);
  if (sc.clean) bits.push(`一次不空 +${sc.bonus}`);
  bits.push(`满分 ${maxScore(f, state.floor, state.rules)}`);
  if (ranked() && sc.done && typeof p.ms === 'number') bits.push(`用时 ${clockText(p.ms)}`);
  box.appendChild(html('span', { class: 'bits' }, bits.join(' · ')));
  return box;
}

function toolIcon(id) {
  if (id === 'key' && ui() && ui().redkey) {
    const k = ui().redkey;
    return html('img', { src: 'art/' + k.file, alt: '', class: 'pixel redkey', width: Math.round(k.w * 1.5), height: Math.round(k.h * 1.5) });
  }
  if (state.sheet) return iconOnly(state.sheet, id === 'bomb' ? 'IconBomb' : 'IconKey', 3);
  return html('span', {}, id === 'bomb' ? '●' : '⚷');
}

function renderFindPanel(panel) {
  const f = floor();
  const p = play();
  const hidden = hiddenRooms(f);
  const found = foundKinds(p);
  const inf = infer(f.joint, p.obs);
  const complete = isComplete(f, p);
  const done = complete || p.revealed;
  panel.appendChild(html('h3', {}, `这一层藏着 ${hidden.length} 个房间`));

  const ul = html('ul', { class: 'targets' });
  for (const h of hidden) {
    const got = found.has(h.kind);
    const li = html('li', { class: got ? 'found' : '' });
    li.appendChild(state.sheet ? tileIcon(state.sheet, KIND_TYPE[h.kind], 2, got ? 'RoomVisited' : 'RoomUnvisited')
      : html('i', { style: `display:block;width:18px;height:16px;border-radius:3px;background:${css(TYPE_COLOR[KIND_TYPE[h.kind]])}` }));
    li.appendChild(html('span', {}, `${KIND_NAME[h.kind]}`));
    const worth = state.rules.room(h.kind, tierOf(state.floor));
    const linked = state.rules.link[h.kind] && singleLinked(f).has(h.kind);   // only once found or shown
    const st = got ? `找到了 · +${worth}${linked ? ' · 单连' : ''}` : p.revealed ? `在${where(h.cell, f.start)}${linked ? ' · 单连' : ''}` : `${worth} 分`;
    li.appendChild(html('span', { class: 'state' }, st));
    ul.appendChild(li);
  }
  panel.appendChild(ul);
  panel.appendChild(scoreLine(f, p));

  const tools = html('div', { class: 'tools', role: 'group', 'aria-label': '工具' });
  tools.appendChild(toolButton('bomb', '炸弹', p.bombs, done));
  if (rep()) tools.appendChild(toolButton('key', '红钥匙', p.keys, done));
  panel.appendChild(tools);

  const [advice, quiet] = adviceText(f, p, inf);
  panel.appendChild(html('p', { class: quiet ? 'advice quiet' : 'advice' }, advice));

  const sw = html('div', { class: 'switches' });
  const toggle = (key, label, help) => {
    const wrap = html('span', { class: 'hintopt' });
    const l = html('label', { class: 'toggle' });
    const i = html('input', { type: 'checkbox', checked: state.hints[key] });
    i.addEventListener('change', () => { state.hints[key] = i.checked; render(); });
    l.append(i, label);
    const info = html('button', { type: 'button', class: 'info', 'aria-label': `${label}是怎么来的`,
      'aria-describedby': `help-${key}` }, '?');
    wrap.append(l, info, html('span', { class: 'info-tip', role: 'tooltip', id: `help-${key}` }, help));
    return wrap;
  };
  sw.append(
    toggle('heat', '显示概率', '按游戏的生成规则推算，只用你看得见的信息：地图的形状，每个房间的布局和门位，'
      + '以及你炸过、找到的格子。不会读取隐藏房的真实位置，所以概率高的地方也可能是空的。'),
    toggle('next', '提示下一步', '在推算出的概率里，挑最可能炸出隐藏房的那面墙；用红钥匙时，挑最可能通到'
      + '究极隐藏房的门位。同样只按规则推算，不看答案。'),
  );
  if (!ranked()) panel.appendChild(sw);
  else if (!done) panel.appendChild(html('p', { class: 'muted' }, '挑战里没有提示，也不能看完整地图。找完或放弃这一层后，可以看 AI 是怎么找的。'));

  const actions = html('div', { class: 'actions' });
  const button = (label, fn, disabled = false, title = '') => {
    const b = html('button', { type: 'button', disabled, title: title || false }, label);
    b.addEventListener('click', fn);
    actions.appendChild(b);
  };
  if (ranked()) {
    button('放弃这一层', giveUp, done, '按已经找到的计分，并揭晓答案');
  } else {
    button('撤销', () => { undo(p); render(); }, !p.log.length || p.revealed, '撤销上一步（Z）');
    button('重来', () => { state.plays.set(state.floor, { ...newPlay(), submitted: false, ms: 0 }); state.tool = 'bomb'; render(); }, !p.log.length && !p.revealed);
    button('揭晓答案', giveUp, complete || p.revealed);
  }
  panel.appendChild(actions);

  if (done) panel.appendChild(winCard(f, p));

  const room = state.selected !== null ? f.rooms.find((q) => q.index === state.selected) : null;
  if (room && (!room.hidden || p.revealed || p.obs.has(room.cells[0]))) {
    panel.appendChild(html('h4', {}, `${room.type_name} · ${where(room.cells[0], f.start)}`));
    panel.appendChild(html('div', { class: 'layoutbox', id: 'layoutbox' }));
    loadLayout(room);
  } else {
    panel.appendChild(html('p', { class: 'muted roomhint' }, '点地图上的房间，看它的布局和门位。'));
  }

  if (p.log.length) {
    panel.appendChild(html('h4', {}, '记录'));
    const ol = html('ol', { class: 'log' });
    const per = scoreOf(state.floor, p).per;
    p.log.forEach((e, idx) => {
      const [head, result, hit] = logText(e, f);
      const li = html('li', {}, head);
      li.appendChild(html('span', { class: hit ? 'hit' : '' }, result));
      const pts = per[idx];
      li.appendChild(html('span', { class: pts > 0 ? 'pts up' : 'pts' }, pts > 0 ? `+${pts}` : String(pts)));
      ol.appendChild(li);
    });
    panel.appendChild(ol);
    ol.scrollTop = ol.scrollHeight;
  }
}

// the card after a floor: Isaac's thumbs up, the points, and what is next
function winCard(f, p) {
  const complete = isComplete(f, p) && !p.revealed;
  const sc = scoreOf(state.floor, p);
  const last = state.floor >= state.data.floors.length - 1;
  const all = ranked() && state.data.floors.every((_, i) => floorDone(i));
  const win = html('div', { class: complete ? 'win' : 'win gaveup' });
  const head = html('div', { class: 'win-head' });
  if (complete && ui()) {
    const isaac = isaacSprite(ui(), 2, 0);
    head.appendChild(isaac);
    animateIsaac(isaac, ui(), 2);
  }
  if (sc.clean && ui() && ui().perfect) head.appendChild(perfectIcon(ui(), 2));   // 满分考卷: no miss on the floor
  head.appendChild(html('h3', {}, complete ? (sc.clean ? '一次不空，全找到了！' : '全找到了！') : '这一层结束了'));
  win.appendChild(head);
  const ai = aiFor(state.floor).expected;
  const used = `你用了 ${p.bombs} 颗炸弹` + (p.keys ? `、${p.keys} 次红钥匙` : '');
  const aiText = ai ? `AI 平均要消耗 ${ai.bombs.toFixed(1)} 颗炸弹${rep() ? `和 ${ai.keys.toFixed(1)} 次红钥匙` : ''}。` : '';
  const perfect = sc.clean ? `（含一次不空 +${sc.bonus}）` : '';
  const took = ranked() && typeof p.ms === 'number' ? `，用时 ${clockText(p.ms)}` : '';
  win.appendChild(html('p', {}, `${ranked() ? `本层 ${sc.total} 分${perfect}${took}。` : ''}${used}；${aiText}`));
  if (all) {
    const time = runTime();
    win.appendChild(html('p', { class: 'total' }, `挑战完成！${state.data.floors.length} 层一共 ${runPoints()} 分${time !== null ? `，用时 ${clockText(time)}` : ''}。`));
  }
  const acts = html('div', { class: 'actions' });
  const watch = html('button', { type: 'button' }, '看 AI 怎么找');
  watch.addEventListener('click', () => {
    state.watch = null;
    showAi(0);
    // in one column the controls come right under the map: bring the map to the top
    if (ONE_COLUMN.matches) $('map').closest('.frame').scrollIntoView({ block: 'start', behavior: 'smooth' });
  });
  acts.appendChild(watch);
  if (!last) {
    const next = html('button', { type: 'button', class: 'primary' }, '下一层');
    next.addEventListener('click', () => goFloor(state.floor + 1));
    acts.appendChild(next);
  }
  if (all || (last && !ranked())) {
    const board = html('button', { type: 'button', class: last && ranked() ? 'primary' : '' }, ranked() ? '看排行榜' : '回首页');
    board.addEventListener('click', () => goHome());
    acts.appendChild(board);
  }
  if (ranked() && (all || last)) {
    const again = html('button', { type: 'button' }, '再来一局随机挑战');
    again.addEventListener('click', () => start('random'));
    acts.appendChild(again);
  }
  win.appendChild(acts);
  return win;
}

// ---------------------------------------------------------------------------- map mode
function renderMapPanel(panel) {
  const f = floor();
  const counts = new Map();
  for (const r of f.rooms) counts.set(r.type_name, (counts.get(r.type_name) || 0) + 1);
  const tags = html('div');
  for (const [name, n] of counts) tags.appendChild(html('span', { class: 'tag' }, n > 1 ? `${name} ×${n}` : name));
  panel.appendChild(html('h3', {}, f.name));
  panel.appendChild(kv([
    ['诅咒', f.curses.length ? f.curses.join('、') : '无'],
    ['头目', f.bosses.map((b) => b.name).join('、')],
    ['房间', tags],
    ['楼层种子', String(f.stage_seed), 'num'],
  ]));
  const box = html('div', { id: 'roominfo' });
  panel.appendChild(box);
  if (state.selected === null) {
    box.appendChild(html('p', { class: 'muted' }, '点地图上的房间，看它的布局。'));
    return;
  }
  const r = f.rooms.find((q) => q.index === state.selected);
  if (!r) return;
  const slots = [];
  for (let s = 0; s < 8; s++) if (r.layout_doors >> s & 1) slots.push(['左', '上', '右', '下', '左2', '上2', '右2', '下2'][s]);
  box.append(html('h4', {}, `${r.type_name}：${r.name}`), kv([
    ['位置', where(r.cells[0], f.start)], ['形状', r.shape_name],
    ['布局', `${r.variant}（难度 ${r.difficulty}）`],
    ['门', r.door_names.length ? r.door_names.join(' ') : '无'], ['布局门位', slots.join(' ') || '无'],
  ]), html('div', { class: 'layoutbox', id: 'layoutbox' }));
  loadLayout(r);
}

// ---------------------------------------------------------------------------- the replays
// After a floor, the AI's way through it (ai.js): the moves the hint policy makes on this floor, with
// the probabilities it saw, shown on the map one step at a time. A player on a daily board the same
// way (state.watch): their own moves, each with the chance it had of finding a room.
function aiFor(i) {
  if (!state.aiCache.has(i)) {
    const f = state.data.floors[i];
    state.aiCache.set(i, { replay: aiReplay(f, rep()), expected: aiExpected(f, rep()) });
  }
  return state.aiCache.get(i);
}
function watching() { return !!state.watch && !state.watch.paused; }
// the replay shown for floor i: the player's (null when there is no record of that floor) or the AI's
function replayOf(i) {
  if (!watching()) return aiFor(i).replay;
  const w = state.watch;
  if (!w.cache.has(i)) {
    const rec = w.floors.get(i);
    w.cache.set(i, rec ? playerReplay(state.data.floors[i], rec.moves, rep(), rec.gaveUp) : null);
  }
  return w.cache.get(i);
}
function aiShown() { return !!state.ai && state.ai.floor === state.floor && !!state.data; }
function replayShown() { return aiShown() && !!replayOf(state.floor); }
function aiAt() {
  const replay = replayOf(state.floor);
  const k = Math.min(state.ai.step, replay.steps.length);
  const step = replay.steps[k] || null;
  return { k, step, steps: replay.steps, replay, obs: step ? step.before : replay.final, inf: step ? step.inf : infer(replay.rows, replay.final) };
}
function showAi(step) {
  const r = replayOf(state.floor);
  const n = r ? r.steps.length : 0;
  state.ai = { floor: state.floor, step: Math.max(0, Math.min(n, step)) };
  state.selected = null;
  render();
}
function stopAi() {
  clearInterval(state.aiTimer);
  state.aiTimer = null;
  state.ai = null;
}
function playAi() {
  if (state.aiTimer) { clearInterval(state.aiTimer); state.aiTimer = null; render(); return; }
  const n = replayOf(state.floor).steps.length;
  if (state.ai.step >= n) state.ai.step = 0;
  state.aiTimer = setInterval(() => {
    if (!aiShown() || state.ai.step >= n) { clearInterval(state.aiTimer); state.aiTimer = null; if (aiShown()) render(); return; }
    showAi(state.ai.step + 1);
  }, 1100);
  render();
}
function goFloor(i) {
  const keep = !!state.watch && aiShown();          // a player's replay goes on to the next floor
  stopAi();
  state.floor = i;
  state.selected = null;
  state.pendingAct = null;
  state.tool = floorTool();
  if (keep) state.ai = { floor: i, step: 0 };
  writeUrl();
  render();
}
function endReplay() {
  state.watch = null;
  stopAi();
  render();
}
// a player's replay: open their daily (this browser's own run of it, finished) and ask for their moves
async function watchPlayer(r, day) {
  const game = state.game;
  window.scrollTo(0, 0);                         // the board is far down the home page
  await start('daily', { day });
  if (!state.data || state.day !== day) return;
  try {
    const rows = await state.scoreboard.replay({ game, day, player: r.client_id, client: state.profile.id, token: state.profile.token });
    state.watch = { id: r.client_id, name: r.player, cache: new Map(), paused: false,
      floors: new Map(rows.map((x) => [x.floor, { moves: parseMoves(x.moves), gaveUp: !!x.gave_up }])) };
    showAi(0);
    if (ONE_COLUMN.matches) $('frame').scrollIntoView({ block: 'start' });   // the map, the controls under it
  } catch (err) {
    flash(err.notFinished ? '你这一天的记录还没传到服务器，联网后再试一次。' : err.message);
    state.scoreboard.flush();
  }
}
function aiStepText(st, f, player = false) {
  const at = where(st.cell, f.start);
  const found = st.found.map((k) => KIND_NAME[k]).join('、');
  if (player) {
    const chance = `命中的概率 ${pct(st.p)}`;
    if (st.tool === 'bomb') return [`炸 ${at}`, chance, found ? `找到${found}` : '空的', !!found];
    if (st.added[0][1] !== 'red') return [`红钥匙 ${at}`, chance, `门后是${found}`, true];
    return [`红钥匙 ${at}`, chance, found ? `开出红房间，通到${found}` : '开出红房间，没有通到隐藏房', !!found];
  }
  if (st.tool === 'bomb') return [`炸 ${at}`, `这里有隐藏房的概率 ${pct(st.p)}`, found ? `找到${found}` : '空的', !!found];
  if (st.added[0][1] !== 'red') return [`红钥匙 ${at}`, `通到究极隐藏房的概率 ${pct(st.p)}`, `门后是${found}`, true];
  return [`红钥匙 ${at}`, `通到究极隐藏房的概率 ${pct(st.p)}`, found ? `开出红房间，通到${found}` : '开出红房间，没有通到隐藏房', !!found];
}
function renderAiBoard(f, base) {
  const at = aiAt();
  const labels = new Map();
  at.steps.slice(0, at.k).forEach((st, i) => labels.set(st.cell, String(i + 1)));
  state.geo = drawMap($('map'), f, {
    ...base, view: 'find', obs: at.obs, revealed: !at.step && !!at.replay.gaveUp, selected: null, heat: at.inf && at.inf.ok ? at.inf : null,
    suggest: at.step ? { bomb: at.step.tool === 'bomb' ? [at.step.cell, at.step.p] : null, key: at.step.tool === 'key' ? [at.step.cell, at.step.p] : null } : null,
    stepLabels: labels, actionable: () => false, onCell: () => {}, onRoom: null,
  });
}
function renderAiPanel(panel) {
  const f = floor(), p = play();
  const w = watching() ? state.watch : null;
  const { expected } = aiFor(state.floor);
  const replay = replayOf(state.floor);
  const used = (q, gaveUp) => `${q.bombs} 颗炸弹${rep() ? `、${q.keys} 次红钥匙` : ''}${gaveUp ? '（放弃了）' : ''}`;
  const exp = expected ? `${expected.bombs.toFixed(1)} 颗炸弹${rep() ? `、${expected.keys.toFixed(1)} 次红钥匙` : ''}` : '—';
  const mine = ['你', used(p, p.revealed && !isComplete(f, p))];
  panel.appendChild(html('h3', {}, w ? `${w.name} 怎么找` : 'AI 怎么找'));
  if (!replay) {                                   // a player with no record of this floor
    panel.appendChild(html('p', { class: 'advice quiet' }, `没有 ${w.name} 这一层的操作记录（可能是在回看功能上线前玩的）。`));
    panel.appendChild(kv([mine, ['AI 平均要', exp]]));
    panel.appendChild(replayExits());
    return;
  }
  const at = aiAt();
  const n = replay.steps.length;
  const kinds = hiddenRooms(f).map((h) => KIND_NAME[h.kind]).join('、');
  const about = html('p', { class: 'advice quiet' }, w
    ? `${w.name} 在这一层的每一步。地图上的数字是那一步之前按规则推算的概率（挑战里看不到），框出来的是 TA 这一步的选择；"命中的概率"是这一步能找到隐藏房的机会。`
    : `AI 只看你也能看到的：地图、房间布局和门位，以及这一层藏着${kinds}。`
      + `它先用炸弹，每次炸最可能有隐藏房的墙${rep() ? '；找完后用红钥匙，每次开最可能通到究极隐藏房的门位' : ''}。地图上的数字是它推算的概率，框出来的是它下一步的选择。`);
  panel.appendChild(kv(w
    ? [[w.name, `${used(replay, replay.gaveUp)} · ${floorScore(f, replay.play, state.floor, state.rules).total} 分`], mine, ['AI 平均要', exp]]
    : [['AI 平均要', exp], ['这一层 AI', used(replay, false)], mine]));
  const bar = html('div', { class: 'actions aibar' });
  const btn = (label, fn, disabled, title) => {
    const b = html('button', { type: 'button', disabled, title: title || false }, label);
    b.addEventListener('click', fn);
    bar.appendChild(b);
  };
  btn('⏮', () => showAi(0), at.k === 0, '回到开头');
  btn('◀ 上一步', () => showAi(at.k - 1), at.k === 0, '←');
  btn(at.k >= n ? '看完了' : `下一步 ▶`, () => showAi(at.k + 1), at.k >= n, '→');
  btn(state.aiTimer ? '暂停' : '自动播放', playAi, n === 0);
  panel.appendChild(bar);
  const end = replay.gaveUp ? `共 ${n} 步，然后放弃了这一层。` : `共 ${n} 步，全部找到了。`;
  panel.appendChild(html('p', { class: 'aistep' }, at.k >= n ? end : `第 ${at.k + 1} / ${n} 步`));
  panel.appendChild(about);
  const ol = html('ol', { class: 'log ailog' });
  replay.steps.forEach((st, i) => {
    const [head, prob, result, hit] = aiStepText(st, f, !!w);
    const li = html('li', { class: i === at.k ? 'now' : i < at.k ? 'done' : '' });
    li.append(html('b', {}, `${i + 1}. ${head}`), html('span', { class: 'prob' }, prob), html('span', { class: hit ? 'hit' : '' }, ` → ${result}`));
    li.addEventListener('click', () => showAi(i));
    ol.appendChild(li);
  });
  panel.appendChild(ol);
  panel.appendChild(replayExits());
  const cur = ol.querySelector('.now');   // the list scrolls to it; the page stays where it is
  if (cur) ol.scrollTop = Math.max(0, cur.offsetTop - (ol.clientHeight - cur.offsetHeight) / 2);
}
// under a replay: between the player's and the AI's, and back to the floor's own result
function replayExits() {
  const back = html('div', { class: 'actions' });
  if (state.watch) {
    const w = state.watch;
    const other = html('button', { type: 'button' }, w.paused ? `看 ${w.name} 怎么找` : '看 AI 怎么找');
    other.addEventListener('click', () => { w.paused = !w.paused; showAi(0); });
    back.appendChild(other);
  }
  const b = html('button', { type: 'button', class: 'primary' }, '回到我的结果');
  b.addEventListener('click', endReplay);
  back.appendChild(b);
  return back;
}

// ---------------------------------------------------------------------------- room layouts
function layoutKey(r) {
  return `${state.data.game}.${r.file}.${r.type}.${r.variant}`;
}
// layouts already fetched in state.layouts, requests in flight in state.layoutJobs
function fetchLayout(r) {
  const key = layoutKey(r);
  if (state.layouts.has(key)) return Promise.resolve(state.layouts.get(key));
  if (!state.layoutJobs.has(key)) {
    const job = state.backend.call('layout', { game: state.data.game, stage: r.file, type: r.type, variant: r.variant })
      .then((lay) => { state.layouts.set(key, lay); return lay; })
      .finally(() => state.layoutJobs.delete(key));
    state.layoutJobs.set(key, job);
  }
  return state.layoutJobs.get(key);
}
// every room the player can look at on this floor: their layouts in one request, then their art
function prefetchFloor() {
  const f = floor(), p = play(), game = state.data.game;
  const rooms = f.rooms.filter((r) => state.view === 'map' || !r.hidden || p.revealed || p.obs.has(r.cells[0]));
  const missing = rooms.filter((r) => !state.layouts.has(layoutKey(r)) && !state.layoutJobs.has(layoutKey(r)));
  const art = () => loadArt().then((a) => {
    if (!a || state.data?.game !== game) return;
    for (const r of rooms) {
      const lay = state.layouts.get(layoutKey(r));
      if (lay) prepare(a, lay, r, f);
    }
  });
  if (!missing.length) { art(); return; }
  const keys = [...new Set(missing.map((r) => `${r.file}.${r.type}.${r.variant}`))];
  const job = state.backend.call('layouts', { game, keys: keys.join(',') })
    .then((out) => { for (const [k, lay] of Object.entries(out)) if (lay) state.layouts.set(`${game}.${k}`, lay); })
    .catch(() => {});
  for (const r of missing) {
    const key = layoutKey(r);
    state.layoutJobs.set(key, job.then(() => state.layouts.get(key) || Promise.reject(new Error('没有这个布局')))
      .finally(() => state.layoutJobs.delete(key)));
  }
  job.then(art);
}
// what the art needs besides the layout: the room (backdrop, seeds) and its floor (stage)
function roomWhere(r) {
  const f = floor();
  return { room: r, floor: f, borrowed: state.data.game === 'abplus',
    title: `${r.type_name} · ${where(r.cells[0], f.start)}${r.name && r.type !== 1 ? ' · ' + r.name : ''}` };
}
async function loadLayout(r) {
  const box = $('layoutbox');
  if (box && !state.layouts.has(layoutKey(r))) box.replaceChildren(roomSkeleton(r.shape));
  try {
    const lay = await fetchLayout(r);
    const now = $('layoutbox');
    if (now && state.selected === r.index) await drawLayout(now, lay, doorState(r), roomWhere(r));
  } catch (err) {
    const now = $('layoutbox');
    if (now) now.textContent = '布局加载失败：' + err.message;
  }
}
// the room as the hover preview: the game art once its files are loaded (asked for here), else a
// placeholder of the room's size; the diagram only if the art cannot be had at all
function roomPreview(r) {
  const again = () => {
    if (state.hover && state.hover.target.room === r) { state.tipSig = null; showTip(state.hover.target, state.hover.e); }
  };
  const lay = state.layouts.get(layoutKey(r));
  if (!lay) {
    fetchLayout(r).then(again).catch(() => {});
    return roomSkeleton(r.shape);
  }
  if (state.art === null) return layoutSvg(lay, doorState(r));
  if (state.art === undefined) {
    loadArt().then(again);
    return roomSkeleton(r.shape);
  }
  const f = floor();
  const prep = prepare(state.art, lay, r, f);
  if (prep.ready) return layoutArt(state.art, lay, r, f, doorState(r));
  prep.job.then(again);
  return roomSkeleton(r.shape);
}
function selectRoom(index) {
  if (!state.data) return;
  state.selected = index;
  renderBoard();
  renderPanel();
}
// a room tapped on the map. In one column its layout (in the panel) would be out of sight below the
// map, so it opens in the dialog, or failing that the panel is scrolled to it.
function pickRoom(r) {
  dismissCoach();
  selectRoom(r.index);
  if (!ONE_COLUMN.matches) return;
  fetchLayout(r).then(async (lay) => {
    if (state.selected !== r.index) return;
    if (!(await openRoom(lay, doorState(r), roomWhere(r)))) $('layoutbox')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }).catch(() => {});
}

function renderPanel() {
  const panel = $('panel');
  panel.replaceChildren();
  if (aiShown()) renderAiPanel(panel);
  else if (state.view === 'map') renderMapPanel(panel);
  else renderFindPanel(panel);
}

function renderLegend() {
  const f = floor();
  const p = play();
  const shown = new Set();
  for (const r of f.rooms) {
    const visible = !r.hidden || state.view === 'map' || p.revealed || p.obs.get(r.cells[0]) !== undefined;
    if (visible && TYPE_NAME[r.type]) shown.add(r.type === 19 ? 18 : r.type);
  }
  const items = [];
  const entry = (node, name) => {
    const s = html('span');
    s.append(node, name);
    items.push(s);
  };
  const swatch = (col) => {
    const i = html('i');
    i.style.background = css(col);
    return i;
  };
  if (useIcons()) {
    entry(tileIcon(state.sheet, 0, 2, 'RoomCurrent'), '起始房间');
    entry(tileIcon(state.sheet, 0, 2, 'RoomVisited'), '普通房间');
    if (state.view === 'map') entry(tileIcon(state.sheet, 0, 2, 'RoomUnvisited'), '要找到才显示');
    for (const t of LEGEND_TYPES) if (shown.has(t)) entry(tileIcon(state.sheet, t, 2), TYPE_NAME[t]);
  } else {
    entry(swatch('--t-start'), '起始房间');
    entry(swatch('--room'), '普通房间');
    for (const t of LEGEND_TYPES) if (shown.has(t)) entry(swatch(TYPE_COLOR[t]), TYPE_NAME[t]);
  }
  if (state.view === 'find' && state.hints.heat && !p.revealed && !isComplete(f, p)) {
    for (const [key, , col] of HEAT) {
      if (key === 'ultra' && !rep()) continue;
      entry(swatch(col), `${KIND_NAME[key]}概率`);
    }
  }
  $('legend').replaceChildren(...items);
}

// ---------------------------------------------------------------------------- tooltip
function positionTip(e) {
  const tip = $('tip');
  const frame = $('frame').getBoundingClientRect();
  let x = e.clientX - frame.left + 16, y = e.clientY - frame.top + 16;
  if (x + tip.offsetWidth > frame.width - 6) x = e.clientX - frame.left - tip.offsetWidth - 12;
  if (y + tip.offsetHeight > frame.height - 6) y = e.clientY - frame.top - tip.offsetHeight - 12;
  tip.style.left = `${Math.max(6, x)}px`;
  tip.style.top = `${Math.max(6, y)}px`;
}
function showTip(target, e) {
  if (!state.data) return;
  const tip = $('tip');
  state.hover = { target, e };
  const p = play();
  const sig = [target.room ? `r${target.room.index}` : `c${target.cell}`, state.floor, state.view, state.tool,
    state.hints.heat, p.log.length, p.revealed, state.ai ? state.ai.step : '-'].join('|');
  if (sig === state.tipSig && !tip.hidden) { positionTip(e); return; }
  state.tipSig = sig;
  const f = floor();
  const lines = [];
  let preview = null;
  if (target.room) {
    const r = target.room;
    if (state.view === 'find' && r.hidden && !p.obs.has(r.cells[0]) && !p.revealed) { hideTip(); return; }
    lines.push(['b', `${r.type_name} · ${where(r.cells[0], f.start)}`], ['', r.name && r.type !== 1 ? r.name : '']);
    preview = roomPreview(r);
  } else {
    const c = target.cell;
    lines.push(['b', where(c, f.start)]);
    if (replayShown()) {
      const at = aiAt();
      const probs = ['secret', 'super'].concat(rep() ? ['ultra'] : []).map((k) => [k, at.inf && at.inf.ok ? at.inf[k].get(c) || 0 : 0])
        .filter(([, q]) => q >= 0.005);
      if (probs.length) lines.push(['p', probs.map(([k, q]) => `${KIND_NAME[k]} ${pct(q)}`).join(' · ')]);
      const i = at.steps.findIndex((st) => st.cell === c);
      if (i >= 0) lines.push(['', `AI 第 ${i + 1} 步${at.steps[i].tool === 'bomb' ? '炸了这里' : '在这里用了红钥匙'}`]);
      if (lines.length === 1) { hideTip(); return; }
    } else if (state.view === 'find' && !p.revealed && !isComplete(f, p)) {
      if (hintsOn() && state.hints.heat) {
        const inf = infer(f.joint, p.obs);
        const probs = ['secret', 'super'].concat(rep() ? ['ultra'] : [])
          .map((k) => [k, inf[k].get(c) || 0]).filter(([, q]) => q >= 0.005);
        if (probs.length) lines.push(['p', probs.map(([k, q]) => `${KIND_NAME[k]} ${pct(q)}`).join(' · ')]);
        for (const [k, q] of probs) {
          if (q > 0.001 && q < 0.999) {
            const h = infer(f.joint, p.obs, [c, k]);
            const others = ['secret', 'super', 'ultra'].filter((o) => o !== k && !foundKinds(p).has(o))
              .map((o) => [o, best(h[o])]).filter(([, t]) => t);
            if (others.length) {
              lines.push(['', `假如是${KIND_NAME[k]}：` + others.map(([o, t]) => `${KIND_NAME[o]}多半在${where(t[0], f.start)}（${pct(t[1])}）`).join('，')]);
            }
          }
        }
      }
      const ok = state.tool === 'bomb' ? canBomb(f, p, c) : canKey(f, p, c);
      lines.push(['', ok ? (state.tool === 'bomb' ? '点击：炸开这面墙' : '点击：用红钥匙开门') : (state.tool === 'bomb' ? '挨不着房间，炸不到' : '这里没有门位')]);
    } else if (lines.length === 1) {
      hideTip();
      return;
    }
  }
  tip.replaceChildren(...lines.filter(([, t]) => t).map(([cls, t]) => {
    const d = html('div');
    d.appendChild(cls === 'b' ? html('b', {}, t) : cls === 'p' ? html('span', { class: 'p' }, t) : document.createTextNode(t));
    return d;
  }));
  if (preview) tip.appendChild(preview);
  tip.hidden = false;
  positionTip(e);
}
function hideTip() {
  $('tip').hidden = true;
  state.hover = null;
  state.tipSig = null;
}

// ---------------------------------------------------------------------------- wiring
function formatSeed() {
  const input = $('seed');
  const raw = input.value.toUpperCase().replace(/[^0-9A-Z]/g, '');
  if (/^\d+$/.test(raw)) { input.value = raw; return; }
  input.value = raw.length > 4 ? `${raw.slice(0, 4)} ${raw.slice(4, 8)}` : raw;
}
function renderMute() {
  $('mute').setAttribute('aria-pressed', String(isMuted()));
  $('mute').classList.toggle('muted', isMuted());
  $('mute').title = isMuted() ? '声音已关，点一下打开' : '声音已开，点一下关掉';
}
function switchGame(game) {
  if (state.game === game) return;
  state.game = game;
  if (state.tool === 'key' && game !== 'repplus') state.tool = 'bomb';
  warmup(game);
  if (state.screen === 'home') {
    state.board = null;
    writeUrl();
    renderHome();
    loadBoard();
  } else {
    render();
    if ($('seed').value.trim()) generate({ keepFloor: true });
  }
}

function init() {
  state.profile = loadProfile();
  state.scoreboard = createScoreboard();
  try { state.text = localStorage.getItem(STORE_TEXT) === '1'; } catch { /* storage unavailable */ }
  try { state.floorsOpen = localStorage.getItem(FLOORS_OPEN) === '1'; } catch { /* storage unavailable */ }
  state.confirm = COARSE.matches;
  try { const c = localStorage.getItem(CONFIRM); if (c !== null) state.confirm = c === '1'; } catch { /* storage unavailable */ }
  const route = readUrl();
  state.backend = createBackend(onProgress, state.game);

  $('brand').addEventListener('click', (e) => { e.preventDefault(); goHome(); });
  $('back').addEventListener('click', () => goHome());
  $('me').addEventListener('click', async () => { await askName(); if (state.screen === 'home') renderBoardList(); });
  $('mute').addEventListener('click', () => { setMuted(!isMuted()); renderMute(); if (!isMuted()) playSound('thumbsup', 0.6); });
  $('pname').addEventListener('change', () => { setName($('pname').value); if (state.board) renderBoardList(); });
  $('pname').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); $('pname').blur(); } });
  for (const b of $('home-game').querySelectorAll('button')) b.addEventListener('click', () => switchGame(b.dataset.game));
  for (const b of $('play-game').querySelectorAll('button')) b.addEventListener('click', () => switchGame(b.dataset.game));
  for (const b of document.querySelectorAll('[data-start]')) {
    b.addEventListener('click', () => {
      const name = cleanName($('pname').value);
      if (name && name !== state.profile.name) setName(name);
      start(b.dataset.start);
    });
  }
  for (const t of $('board').parentElement.querySelectorAll('.tabs button')) {
    t.addEventListener('click', () => { state.scope = t.dataset.scope; state.board = null; renderHome(); loadBoard(); });
  }
  $('dlgcancel').addEventListener('click', () => answerName(false));
  $('namedlg').addEventListener('close', () => answerName(false));
  $('nameform').addEventListener('submit', (e) => {
    e.preventDefault();
    const name = cleanName($('dlgname').value);
    if (!name) { $('dlgerror').textContent = '名字不能为空。'; return; }
    setName(name);
    answerName(true);
  });
  $('seedform').addEventListener('submit', (e) => { e.preventDefault(); generate({ push: true }); });
  $('seed').addEventListener('input', formatSeed);
  $('dice').addEventListener('click', randomRun);
  $('welcome-random').addEventListener('click', randomRun);
  for (const b of document.querySelectorAll('.board-head .tabs button')) {
    b.addEventListener('click', () => {
      if (b.dataset.view === 'map' && !mapOpen()) return;
      stopAi();
      state.view = b.dataset.view;
      state.selected = null;
      render();
    });
  }
  for (const k of PARAMS) $(k).addEventListener('change', () => { if (state.data) generate({ keepFloor: true }); });
  $('t-text').addEventListener('change', () => {
    state.text = $('t-text').checked;
    try { localStorage.setItem(STORE_TEXT, state.text ? '1' : '0'); } catch { /* storage unavailable */ }
    render();
  });
  $('t-confirm').addEventListener('change', () => {
    state.confirm = $('t-confirm').checked;
    state.pendingAct = null;
    try { localStorage.setItem(CONFIRM, state.confirm ? '1' : '0'); } catch { /* storage unavailable */ }
    render();
  });
  // the clock stops while the page is in the background, and its time is saved now and then
  document.addEventListener('visibilitychange', syncClock);
  window.addEventListener('pagehide', stopClock);
  window.addEventListener('pageshow', syncClock);
  setInterval(() => { if (clock.i !== null) { stopClock(); syncClock(); } }, 15000);
  document.addEventListener('keydown', (e) => {
    if (state.screen !== 'play' || !state.data || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    if (document.querySelector('dialog[open]')) return;
    const key = e.key.toLowerCase();
    if (e.key === 'Escape' && state.pendingAct) { state.pendingAct = null; renderBoard(); return; }
    if (aiShown() && (e.key === 'ArrowRight' || e.key === 'ArrowLeft')) { showAi(state.ai.step + (e.key === 'ArrowRight' ? 1 : -1)); return; }
    if (e.key === 'Escape' && aiShown()) { endReplay(); return; }
    if (e.key === 'ArrowRight' && state.floor < state.data.floors.length - 1) goFloor(state.floor + 1);
    else if (e.key === 'ArrowLeft' && state.floor > 0) goFloor(state.floor - 1);
    else if (state.view === 'find' && key === 'b') { state.tool = 'bomb'; render(); }
    else if (state.view === 'find' && key === 'k' && rep()) { state.tool = 'key'; render(); }
    else if (state.view === 'find' && key === 'h' && !ranked()) { state.hints.heat = !state.hints.heat; render(); }
    else if (state.view === 'find' && !ranked() && (key === 'z' || key === 'u')) { if (undo(play())) render(); }
    else if (key === 'm' && (state.view === 'map' || mapOpen())) { state.view = state.view === 'find' ? 'map' : 'find'; state.selected = null; render(); }
  });
  let resize = null;
  window.addEventListener('resize', () => {
    clearTimeout(resize);
    resize = setTimeout(() => { if (state.screen === 'play' && state.data) renderBoard(); }, 120);
  });
  document.addEventListener('click', (e) => {
    const o = $('options');
    if (o.open && !o.contains(e.target)) o.open = false;
  });
  window.addEventListener('popstate', () => {
    const r = readUrl();
    if (r.screen === 'home') goHome(false);
    else start(r.mode, { seed: r.seed, day: r.day, floorIndex: r.floorIndex, push: false });
  });

  renderMute();
  renderMe();
  loadArt().then((art) => {
    state.art = art || null;
    if (!art) return;
    loadSounds(art.ui);
    drawBrand();
    if (state.screen === 'play' && state.data) render();
    if (state.screen === 'home' && state.board) renderBoardList();       // the trophies
  });
  sendMoves();
  state.scoreboard.flush();
  if (route.screen === 'play') {
    start(route.mode, { seed: route.seed, day: route.day, floorIndex: route.floorIndex, push: false });
  } else {
    showScreen('home');
    renderHome();
    loadBoard();
  }
  warmup(state.game);
}

init();
