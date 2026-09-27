// 找隐藏房: the page. Generates a run (backend.js), then lets the player find each floor's hidden
// rooms on the game's minimap (play.js), with the linked posterior as hints (infer.js).
import { createBackend } from './backend.js';
import {
  KIND_NAME, KIND_TYPE, best, expectedBombs, infer, nextBomb, nextRedKey, pct, where,
} from './infer.js';
import {
  HEAT, boundsOf, css, drawLayout, drawMap, drawThumb, iconOnly, layoutArt, layoutSvg, playEffect, tileIcon,
} from './minimap.js';
import { loadArt, prepare } from './roomart.js';
import {
  canBomb, canKey, foundKinds, hiddenRooms, isComplete, keyTargets, knownCells, newPlay, undo, useBomb, useKey,
} from './play.js';

const $ = (id) => document.getElementById(id);
const GAMES = { abplus: '胎衣†', repplus: '忏悔+' };
const PARAMS = ['mode', 'route', 'last', 'coins', 'keys', 'hearts', 'max_hearts', 'soul'];
const DEFAULTS = { mode: 'normal', route: 'sheol', last: '11', coins: '0', keys: '0', hearts: '6', max_hearts: '6', soul: '0' };
const STORE_TEXT = 'isaac-mapgen-text-labels';
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
  backend: null, game: 'abplus', data: null, floor: 0, view: 'find', plays: new Map(), tool: 'bomb',
  hints: { heat: false, next: false }, text: false, sheets: new Map(), sheet: null, selected: null,
  layouts: new Map(), layoutJobs: new Map(), expected: new Map(), progress: new Map(), pending: 0, geo: null, k: 6,
  hover: null, art: undefined,
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
function play(i = state.floor) {
  if (!state.plays.has(i)) state.plays.set(i, newPlay());
  return state.plays.get(i);
}
function useIcons() { return !state.text && !!state.sheet; }

// ---------------------------------------------------------------------------- backend
function onProgress(m) {
  if (m.state === 'error') {
    setStatus('运行环境加载失败：' + m.text, true);
    return;
  }
  const prev = state.progress.get(m.step) || {};
  state.progress.set(m.step, { state: m.state, text: m.text || prev.text });
  renderLoading();
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

// ---------------------------------------------------------------------------- generate
function readParams() {
  const out = { game: state.game, seed: $('seed').value.trim() };
  for (const k of PARAMS) out[k] = $(k).value;
  return out;
}
async function generate({ keepFloor = false, floorIndex = null } = {}) {
  const params = readParams();
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
    state.data = data;
    state.sheet = sheet;
    state.plays = new Map();
    state.expected = new Map();
    state.selected = null;
    const i = floorIndex !== null ? floorIndex : keepFloor ? state.floor : 0;
    state.floor = Math.max(0, Math.min(data.floors.length - 1, i));
    $('seed').value = data.seed.text;
    writeUrl();
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
  try {
    const r = await call('random', {});
    $('seed').value = r.text;
    generate();
  } catch (err) {
    setStatus('生成失败：' + err.message, true);
  }
}

// ---------------------------------------------------------------------------- URL
function writeUrl() {
  const q = new URLSearchParams({ game: state.game, seed: $('seed').value.trim(), view: state.view, f: String(state.floor + 1) });
  for (const k of PARAMS) if ($(k).value !== DEFAULTS[k]) q.set(k, $(k).value);
  history.replaceState(null, '', '?' + q.toString());
}
function readUrl() {
  const q = new URLSearchParams(location.search);
  if (GAMES[q.get('game')]) state.game = q.get('game');
  if (q.get('seed')) $('seed').value = q.get('seed');
  if (q.get('view') === 'map') state.view = 'map';
  for (const k of PARAMS) if (q.get(k) !== null) $(k).value = q.get(k);
  const f = parseInt(q.get('f') || '1', 10);
  return Number.isFinite(f) ? f - 1 : 0;
}

// ---------------------------------------------------------------------------- render
function render() {
  const has = !!state.data;
  $('app').classList.toggle('ready', has);
  $('welcome').hidden = has || state.pending > 0;
  $('floors').hidden = !has;
  $('boardhead').hidden = !has;
  $('boardfoot').hidden = !has;
  $('panel').hidden = !has;
  $('map').style.display = has ? '' : 'none';
  for (const b of document.querySelectorAll('.segmented button')) b.setAttribute('aria-checked', String(b.dataset.game === state.game));
  if (!has) return;
  $('t-text').checked = state.text;
  $('t-text').disabled = !state.sheet;
  renderFloors();
  renderBoard();
  renderPanel();
  renderLegend();
}

function floorStatus(i) {
  const f = state.data.floors[i];
  if (state.view === 'map') {
    return { text: [`${f.rooms.length} 间`].concat(f.curses).join(' · '), done: false };
  }
  const p = state.plays.get(i);
  const total = hiddenRooms(f).length;
  const found = p ? foundKinds(p).size : 0;
  if (p && p.revealed) return { text: '已揭晓', done: false };
  return { text: found >= total ? '全找到了' : `${found}/${total}`, done: found >= total };
}

function renderFloors() {
  const nav = $('floors');
  nav.replaceChildren(...state.data.floors.map((f, i) => {
    const btn = html('button', { type: 'button', class: 'floor', 'aria-current': String(i === state.floor), title: f.name });
    const mini = html('span', { class: 'mini' });
    const p = state.plays.get(i);
    mini.appendChild(drawThumb(f, state.sheet, useIcons() ? 'icons' : 'text', state.view, p ? p.obs : null, p && p.revealed));
    const st = floorStatus(i);
    btn.append(mini, html('span', { class: 'fname' }, f.name), html('span', { class: st.done ? 'fmeta done' : 'fmeta' }, st.text));
    btn.addEventListener('click', () => { state.floor = i; state.selected = null; writeUrl(); render(); });
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

function renderBoard() {
  const f = floor();
  const title = $('floortitle');
  title.replaceChildren(document.createTextNode(f.name), html('small', {}, f.name_en));
  for (const b of document.querySelectorAll('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.view === state.view));
  const b = viewBounds(f);
  const k = scaleFor(b);
  state.k = k;
  const base = { k, style: useIcons() ? 'icons' : 'text', sheet: state.sheet, bounds: b, onHover: showTip, onLeave: hideTip };
  hideTip();
  if (state.view === 'map') {
    state.geo = drawMap($('map'), f, { ...base, view: 'map', selected: state.selected, onRoom: (r) => selectRoom(r.index) });
    return;
  }
  const p = play();
  const done = isComplete(f, p) || p.revealed;
  const inf = infer(f.joint, p.obs);
  const suggest = state.hints.next && !done ? {
    bomb: state.tool === 'bomb' ? nextBomb(inf, p.obs, (c) => canBomb(f, p, c)) : null,
    key: state.tool === 'key' ? nextRedKey(inf, p.obs, keyTargets(f, p)) : null,
  } : null;
  state.geo = drawMap($('map'), f, {
    ...base, view: 'find', obs: p.obs, revealed: p.revealed, selected: state.selected,
    heat: state.hints.heat && !done && inf.ok ? inf : null, suggest,
    actionable: done ? () => false : (c) => (state.tool === 'bomb' ? canBomb(f, p, c) : canKey(f, p, c)),
    onCell: act, onRoom: (r) => selectRoom(r.index),
  });
}

// What a room's door slots show. In the map view every door is known. In the find view a door is
// seen only into a room the player knows (explored, found, or a red room); a doorway into a hidden
// room would give it away, so the rest are drawn as layout slots without a door.
function doorState(r) {
  if (state.view === 'map') return (bit) => ((r.doors & bit) ? 'open' : 'slot');
  const known = knownCells(floor(), play());
  return (bit) => {
    const target = r.slot_targets ? r.slot_targets[31 - Math.clz32(bit)] : -1;
    return target >= 0 && known.has(target) ? 'open' : 'slot';
  };
}

// ---------------------------------------------------------------------------- find mode
function act(cell) {
  const f = floor();
  const p = play();
  if (p.revealed || isComplete(f, p)) return;
  let entry;
  if (state.tool === 'bomb') {
    if (!canBomb(f, p, cell)) { flash('这里挨不着任何房间，炸不到。'); return; }
    entry = useBomb(f, p, cell);
  } else {
    if (!canKey(f, p, cell)) { flash('红钥匙只能用在房间的门位外面。'); return; }
    entry = useKey(f, p, cell);
  }
  render();
  playEffect($('map'), state.geo, entry.cell, entry.found.length > 0 || entry.tool === 'key', state.k);
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
  if (!state.hints.next) {
    return [state.tool === 'key' ? '选一个房间的门位外面，用红钥匙开门。' : '点挨着房间的空格子，用炸弹炸开那面墙。', true];
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

function renderFindPanel(panel) {
  const f = floor();
  const p = play();
  const hidden = hiddenRooms(f);
  const found = foundKinds(p);
  const inf = infer(f.joint, p.obs);
  const complete = isComplete(f, p);
  panel.appendChild(html('h3', {}, `这一层藏着 ${hidden.length} 个房间`));

  const ul = html('ul', { class: 'targets' });
  for (const h of hidden) {
    const got = found.has(h.kind);
    const li = html('li', { class: got ? 'found' : '' });
    li.appendChild(state.sheet ? tileIcon(state.sheet, KIND_TYPE[h.kind], 2, got ? 'RoomVisited' : 'RoomUnvisited')
      : html('i', { style: `display:block;width:18px;height:16px;border-radius:3px;background:${css(TYPE_COLOR[KIND_TYPE[h.kind]])}` }));
    li.appendChild(html('span', {}, KIND_NAME[h.kind]));
    const st = got ? `找到了 · ${where(h.cell, f.start)}` : p.revealed ? `在${where(h.cell, f.start)}` : '还没找到';
    li.appendChild(html('span', { class: 'state' }, st));
    ul.appendChild(li);
  }
  panel.appendChild(ul);

  const tools = html('div', { class: 'tools', role: 'group', 'aria-label': '工具' });
  const tool = (id, label, icon, count, extra = '') => {
    const b = html('button', { type: 'button', 'aria-pressed': String(state.tool === id), title: `${label}（${id === 'bomb' ? 'B' : 'K'}）` });
    if (state.sheet) {
      const s = iconOnly(state.sheet, icon, 3);
      if (extra) s.setAttribute('class', extra);
      b.appendChild(s);
    } else {
      b.appendChild(html('span', {}, id === 'bomb' ? '●' : '⚷'));
    }
    b.append(html('span', {}, label), html('b', { class: 'count', title: '已用' }, String(count)));
    b.addEventListener('click', () => { state.tool = id; render(); });
    return b;
  };
  tools.appendChild(tool('bomb', '炸弹', 'IconBomb', p.bombs));
  if (rep()) tools.appendChild(tool('key', '红钥匙', 'IconKey', p.keys, 'key-red'));
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
  panel.appendChild(sw);

  const actions = html('div', { class: 'actions' });
  const button = (label, fn, disabled = false, title = '') => {
    const b = html('button', { type: 'button', disabled, title: title || false }, label);
    b.addEventListener('click', fn);
    actions.appendChild(b);
  };
  button('撤销', () => { undo(p); render(); }, !p.log.length || p.revealed, '撤销上一步（Z）');
  button('重来', () => { state.plays.set(state.floor, newPlay()); render(); }, !p.log.length && !p.revealed);
  button('揭晓答案', () => { p.revealed = true; render(); }, complete || p.revealed);
  panel.appendChild(actions);

  if (complete && !p.revealed) {
    if (!state.expected.has(state.floor)) state.expected.set(state.floor, expectedBombs(f.joint));
    const exp = state.expected.get(state.floor);
    const win = html('div', { class: 'win' });
    win.appendChild(html('h3', {}, '全找到了！'));
    const used = `用了 ${p.bombs} 颗炸弹` + (p.keys ? `、${p.keys} 把红钥匙` : '');
    win.appendChild(html('p', {}, exp !== null ? `${used}。照着提示炸，平均要 ${exp.toFixed(1)} 颗炸弹。` : `${used}。`));
    const acts = html('div', { class: 'actions' });
    if (state.floor < state.data.floors.length - 1) {
      const next = html('button', { type: 'button', class: 'primary' }, '下一层');
      next.addEventListener('click', () => { state.floor += 1; writeUrl(); render(); });
      acts.appendChild(next);
    }
    const again = html('button', { type: 'button' }, '换个种子');
    again.addEventListener('click', randomRun);
    acts.appendChild(again);
    win.appendChild(acts);
    panel.appendChild(win);
  }

  const room = state.selected !== null ? f.rooms.find((q) => q.index === state.selected) : null;
  if (room && (!room.hidden || p.revealed || p.obs.has(room.cells[0]))) {
    panel.appendChild(html('h4', {}, `${room.type_name} · ${where(room.cells[0], f.start)}`));
    panel.appendChild(html('div', { class: 'layoutbox', id: 'layoutbox' }, '加载布局…'));
    loadLayout(room);
  } else {
    panel.appendChild(html('p', { class: 'muted roomhint' }, '点地图上的房间，看它的布局和门位。'));
  }

  if (p.log.length) {
    panel.appendChild(html('h4', {}, '记录'));
    const ol = html('ol', { class: 'log' });
    for (const e of p.log) {
      const [head, result, hit] = logText(e, f);
      const li = html('li', {}, head);
      li.appendChild(html('span', { class: hit ? 'hit' : '' }, result));
      ol.appendChild(li);
    }
    panel.appendChild(ol);
    ol.scrollTop = ol.scrollHeight;
  }
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
  ]), html('div', { class: 'layoutbox', id: 'layoutbox' }, '加载布局…'));
  loadLayout(r);
}
function layoutKey(r) {
  return `${state.data.game}.${r.file}.${r.type}.${r.variant}`;
}
// layouts already fetched in state.layouts, requests in flight in state.layoutJobs
function fetchLayout(r) {
  const key = layoutKey(r);
  if (state.layouts.has(key)) return Promise.resolve(state.layouts.get(key));
  if (!state.layoutJobs.has(key)) {
    const job = call('layout', { game: state.data.game, stage: r.file, type: r.type, variant: r.variant })
      .then((lay) => { state.layouts.set(key, lay); return lay; })
      .finally(() => state.layoutJobs.delete(key));
    state.layoutJobs.set(key, job);
  }
  return state.layoutJobs.get(key);
}
async function loadLayout(r) {
  try {
    const lay = await fetchLayout(r);
    const box = $('layoutbox');
    if (box && state.selected === r.index) await drawLayout(box, lay, doorState(r), roomWhere(r));
  } catch (err) {
    const box = $('layoutbox');
    if (box) box.textContent = '布局加载失败：' + err.message;
  }
}
// what the art needs besides the layout: the room (backdrop, seeds) and its floor (stage)
function roomWhere(r) {
  const f = floor();
  return { room: r, floor: f, borrowed: state.data.game === 'abplus',
    title: `${r.type_name} · ${where(r.cells[0], f.start)}${r.name && r.type !== 1 ? ' · ' + r.name : ''}` };
}
// the room as the hover preview: the game art once its files are loaded (asked for here), else the diagram
function roomPreview(r, lay) {
  const again = () => {
    if (state.hover && state.hover.target.room === r) showTip(state.hover.target, state.hover.e);
  };
  if (state.art === undefined) {
    state.art = null;
    loadArt().then((art) => { state.art = art; if (art) again(); });
  }
  if (state.art) {
    const where = roomWhere(r);
    const prep = prepare(state.art, lay, r, where.floor);
    if (prep.ready) return layoutArt(state.art, lay, r, where.floor, doorState(r));
    prep.job.then(again);
  }
  return layoutSvg(lay, doorState(r));
}
function selectRoom(index) {
  state.selected = index;
  renderBoard();
  renderPanel();
}

function renderPanel() {
  const panel = $('panel');
  panel.replaceChildren();
  if (state.view === 'map') renderMapPanel(panel);
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
function showTip(target, e) {
  const tip = $('tip');
  const f = floor();
  const lines = [];
  let preview = null;
  state.hover = { target, e };
  if (target.room) {
    const r = target.room;
    if (state.view === 'find' && r.hidden && !play().obs.has(r.cells[0]) && !play().revealed) return hideTip();
    lines.push(['b', `${r.type_name} · ${where(r.cells[0], f.start)}`], ['', r.name && r.type !== 1 ? r.name : '']);
    const lay = state.layouts.get(layoutKey(r));
    if (lay) {
      preview = roomPreview(r, lay);
    } else {
      lines.push(['', '布局加载中…']);
      fetchLayout(r).then(() => {
        if (state.hover && state.hover.target.room === r) showTip(state.hover.target, state.hover.e);
      }).catch(() => {});
    }
  } else {
    const c = target.cell;
    const p = play();
    lines.push(['b', where(c, f.start)]);
    if (state.view === 'find' && !p.revealed && !isComplete(f, p)) {
      if (state.hints.heat) {
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
      return hideTip();
    }
  }
  tip.replaceChildren(...lines.filter(([, t]) => t).map(([cls, t]) => {
    const d = html('div');
    d.appendChild(cls === 'b' ? html('b', {}, t) : cls === 'p' ? html('span', { class: 'p' }, t) : document.createTextNode(t));
    return d;
  }));
  if (preview) tip.appendChild(preview);
  tip.hidden = false;
  const frame = $('frame').getBoundingClientRect();
  let x = e.clientX - frame.left + 16, y = e.clientY - frame.top + 16;
  if (x + tip.offsetWidth > frame.width - 6) x = e.clientX - frame.left - tip.offsetWidth - 12;
  if (y + tip.offsetHeight > frame.height - 6) y = e.clientY - frame.top - tip.offsetHeight - 12;
  tip.style.left = `${Math.max(6, x)}px`;
  tip.style.top = `${Math.max(6, y)}px`;
}
function hideTip() {
  $('tip').hidden = true;
  state.hover = null;
}

// ---------------------------------------------------------------------------- wiring
function formatSeed() {
  const input = $('seed');
  const raw = input.value.toUpperCase().replace(/[^0-9A-Z]/g, '');
  if (/^\d+$/.test(raw)) { input.value = raw; return; }
  input.value = raw.length > 4 ? `${raw.slice(0, 4)} ${raw.slice(4, 8)}` : raw;
}

function init() {
  state.backend = createBackend(onProgress);
  try { state.text = localStorage.getItem(STORE_TEXT) === '1'; } catch { /* storage unavailable */ }
  const floorIndex = readUrl();
  $('seedform').addEventListener('submit', (e) => { e.preventDefault(); generate(); });
  $('seed').addEventListener('input', formatSeed);
  $('dice').addEventListener('click', randomRun);
  $('welcome-random').addEventListener('click', randomRun);
  for (const b of document.querySelectorAll('.segmented button')) {
    b.addEventListener('click', () => {
      if (state.game === b.dataset.game) return;
      state.game = b.dataset.game;
      if (state.tool === 'key' && state.game !== 'repplus') state.tool = 'bomb';
      render();
      if ($('seed').value.trim()) generate({ keepFloor: true });
    });
  }
  for (const b of document.querySelectorAll('.tabs button')) {
    b.addEventListener('click', () => { state.view = b.dataset.view; state.selected = null; writeUrl(); render(); });
  }
  for (const k of PARAMS) $(k).addEventListener('change', () => { if (state.data) generate({ keepFloor: true }); });
  $('t-text').addEventListener('change', () => {
    state.text = $('t-text').checked;
    try { localStorage.setItem(STORE_TEXT, state.text ? '1' : '0'); } catch { /* storage unavailable */ }
    render();
  });
  document.addEventListener('keydown', (e) => {
    if (!state.data || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    const key = e.key.toLowerCase();
    if (e.key === 'ArrowRight' && state.floor < state.data.floors.length - 1) { state.floor += 1; state.selected = null; writeUrl(); render(); }
    else if (e.key === 'ArrowLeft' && state.floor > 0) { state.floor -= 1; state.selected = null; writeUrl(); render(); }
    else if (state.view === 'find' && key === 'b') { state.tool = 'bomb'; render(); }
    else if (state.view === 'find' && key === 'k' && rep()) { state.tool = 'key'; render(); }
    else if (state.view === 'find' && key === 'h') { state.hints.heat = !state.hints.heat; render(); }
    else if (state.view === 'find' && (key === 'z' || key === 'u')) { if (undo(play())) render(); }
    else if (key === 'm') { state.view = state.view === 'find' ? 'map' : 'find'; state.selected = null; writeUrl(); render(); }
  });
  let resize = null;
  window.addEventListener('resize', () => {
    clearTimeout(resize);
    resize = setTimeout(() => { if (state.data) renderBoard(); }, 120);
  });
  document.addEventListener('click', (e) => {
    const o = $('options');
    if (o.open && !o.contains(e.target)) o.open = false;
  });
  render();
  if ($('seed').value.trim()) generate({ floorIndex });
}

init();
