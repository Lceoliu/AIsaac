'use strict';

const $ = (id) => document.getElementById(id);
const SVGNS = 'http://www.w3.org/2000/svg';
const GRID = 13;
const TRAVEL = [[-1, 0], [0, -1], [1, 0], [0, 1]];

const TYPE_COLOR = {
  1: '--room', 2: '--t-shop', 3: '--t-other', 4: '--t-treasure', 5: '--t-boss', 6: '--t-miniboss',
  7: '--t-secret', 8: '--t-supersecret', 9: '--t-arcade', 10: '--t-curse', 11: '--t-challenge',
  12: '--t-library', 13: '--t-sacrifice', 18: '--t-bedroom', 19: '--t-bedroom', 20: '--t-vault',
  21: '--t-dice', 24: '--t-planetarium', 29: '--t-ultrasecret',
};
const LEGEND = [
  ['--t-start', '起始房间'], ['--room', '普通'], ['--t-boss', '头目房'], ['--t-treasure', '宝箱房'],
  ['--t-shop', '商店'], ['--t-secret', '隐藏房'], ['--t-supersecret', '超级隐藏房'], ['--t-curse', '诅咒房'],
  ['--t-miniboss', '小头目房'], ['--t-challenge', '挑战房'], ['--t-library', '图书馆'], ['--t-sacrifice', '献祭房'],
  ['--t-arcade', '赌博房'], ['--t-vault', '宝库'], ['--t-dice', '骰子房'], ['--t-bedroom', '卧室'],
];
const LEGEND_REP = [['--t-planetarium', '星象房'], ['--t-ultrasecret', '究极隐藏房']];
// hidden-room heat layers: [JSON key, map tag, colour, legend]
const HEAT = [
  ['secret_posterior', '隐', '--heat', '隐藏房概率'],
  ['super_secret_posterior', '超', '--heat-ss', '超级隐藏房概率'],
  ['ultra_secret_posterior', '极', '--heat-us', '究极隐藏房概率'],
];
const GAME_TEXT = {
  abplus: { sub: '输入种子，离线生成《以撒的结合：胎衣†》v1.06 每一层的地图', title: '胎衣†' },
  repplus: { sub: '输入种子，离线生成《以撒的结合：忏悔+》v1.9.7.17 每一层的地图（静态移植，未经游戏验证）', title: '忏悔+' },
};
const KIND = {
  rock: ['#8a8175', '石头类'], poop: ['#7a5230', '大便'], tnt: ['#b0412e', '炸药桶'], block: ['#9aa3ad', '方块'],
  pit: ['#050404', '沟壑'], spikes: ['#7d2020', '地刺'], web: ['#cfcfcf', '蛛网'], plate: ['#3d6fb0', '按钮'],
  door: ['#3d6fb0', '活板门/暗门'], deco: ['#5a5246', '装饰'], grid: ['#777777', '网格物体'],
  pickup: ['#f2c33b', '拾取物'], enemy: ['#e2553f', '敌人'], object: ['#3fb0a8', '机器/火堆等'],
};

const state = { data: null, floor: 0, room: null, layoutCache: new Map() };

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function html(tag, attrs = {}, text = null) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  return e;
}
function roomColor(room) {
  if (room.start) return css('--t-start');
  return css(TYPE_COLOR[room.type] || '--t-other');
}
function hex(n) { return '0x' + (n >>> 0).toString(16).toUpperCase().padStart(8, '0'); }

// ---------------------------------------------------------------------------- requests
function readParams() {
  return {
    game: $('game').value, seed: $('seed').value.trim(), mode: $('mode').value, route: $('route').value,
    last: $('last').value, coins: $('coins').value, keys: $('keys').value, hearts: $('hearts').value,
    max_hearts: $('max_hearts').value, soul: $('soul').value,
  };
}
function writeParams(p) {
  for (const k of ['game', 'seed', 'mode', 'route', 'last', 'coins', 'keys', 'hearts', 'max_hearts', 'soul']) {
    if (p.get(k) !== null && $(k)) $(k).value = p.get(k);
  }
  if (!GAME_TEXT[$('game').value]) $('game').value = 'abplus';
}
function applyGame() {
  const game = $('game').value;
  $('subtitle').textContent = GAME_TEXT[game].sub;
  document.title = `以撒楼层生成器 · ${GAME_TEXT[game].title}`;
  for (const f of document.querySelectorAll('footer[data-game]')) f.hidden = f.dataset.game !== game;
}
function setStatus(text, error = false) {
  const s = $('status');
  s.textContent = text;
  s.classList.toggle('error', error);
}

async function generate(keepFloor = false) {
  const params = readParams();
  if (!params.seed) { setStatus('请输入种子，或点"随机"。', true); return; }
  $('go').disabled = true;
  setStatus('生成中…');
  try {
    const qs = new URLSearchParams(params);
    const res = await fetch('/api/run?' + qs.toString());
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    history.replaceState(null, '', '?' + qs.toString());
    state.data = data;
    $('seed').value = data.seed.text;
    if (!keepFloor || state.floor >= data.floors.length) state.floor = 0;
    state.room = null;
    setStatus('');
    render();
  } catch (err) {
    setStatus('生成失败：' + err.message, true);
  } finally {
    $('go').disabled = false;
  }
}

// ---------------------------------------------------------------------------- map drawing
function bounds(floor) {
  let x0 = GRID, y0 = GRID, x1 = -1, y1 = -1;
  for (const r of floor.rooms) {
    for (const c of r.cells) {
      const x = c % GRID, y = Math.floor(c / GRID);
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
  }
  x0 = Math.max(0, x0 - 1); y0 = Math.max(0, y0 - 1);
  x1 = Math.min(GRID - 1, x1 + 1); y1 = Math.min(GRID - 1, y1 + 1);
  return { x0, y0, cols: x1 - x0 + 1, rows: y1 - y0 + 1 };
}

function drawFloor(svg, floor, opts) {
  const { S, gap, labels, player, depth, heat, selected, interactive } = opts;
  svg.replaceChildren();
  const b = bounds(floor);
  svg.setAttribute('viewBox', `0 0 ${b.cols * S} ${b.rows * S}`);
  const pos = (c) => [(c % GRID - b.x0) * S, (Math.floor(c / GRID) - b.y0) * S];
  const hidden = (r) => r.hidden && player;
  const visibleRooms = floor.rooms.filter((r) => !hidden(r) && (opts.showHidden || !r.hidden));
  const byCell = new Map();
  for (const r of visibleRooms) for (const c of r.cells) byCell.set(c, r);

  // empty grid dots
  const g0 = el('g', {}, svg);
  for (let yy = 0; yy < b.rows; yy++) {
    for (let xx = 0; xx < b.cols; xx++) {
      el('rect', { x: xx * S + gap / 2, y: yy * S + gap / 2, width: S - gap, height: S - gap, rx: S * 0.08,
        fill: css('--cell-empty') }, g0);
    }
  }

  // heat map of hidden-room probabilities (player view)
  if (heat) {
    const gh = el('g', {}, svg);
    const layers = HEAT.filter(([key]) => floor[key] && Object.keys(floor[key]).length);
    const cells = new Set(layers.flatMap(([key]) => Object.keys(floor[key])));
    for (const key of cells) {
      const c = Number(key);
      if (byCell.has(c)) continue;
      const [x, y] = pos(c);
      const lines = layers.map(([k, tag, col]) => [tag, floor[k][key] || 0, col]).filter(([, p]) => p >= 0.01);
      if (!lines.length) continue;
      const main = lines.reduce((a, b) => (b[1] > a[1] ? b : a));
      el('rect', { x: x + gap / 2, y: y + gap / 2, width: S - gap, height: S - gap, rx: S * 0.08,
        fill: css(main[2]), 'fill-opacity': 0.12 + 0.6 * Math.min(1, main[1] / 0.5) }, gh);
      lines.forEach(([tag, p], i) => {
        const t = el('text', { x: x + S / 2, y: y + S / 2 + (i - (lines.length - 1) / 2) * S * 0.26 + S * 0.08,
          'text-anchor': 'middle', 'font-size': S * 0.21, fill: css('--label'), 'font-weight': 600 }, gh);
        t.textContent = `${tag}${Math.round(p * 100)}%`;
      });
    }
  }

  // selection halo
  const gRooms = el('g', {}, svg);
  const drawCells = (r, parent, fill, grow, attrs = {}) => {
    const set = new Set(r.cells);
    for (const c of r.cells) {
      const [x, y] = pos(c);
      el('rect', { x: x + gap / 2 - grow, y: y + gap / 2 - grow, width: S - gap + 2 * grow, height: S - gap + 2 * grow,
        rx: S * 0.08, fill, class: 'cell', ...attrs }, parent);
      const right = c + 1, down = c + GRID;
      if (c % GRID < GRID - 1 && set.has(right)) {
        el('rect', { x: x + S - gap / 2 - grow, y: y + gap / 2 - grow, width: gap + 2 * grow, height: S - gap + 2 * grow, fill }, parent);
      }
      if (set.has(down)) {
        el('rect', { x: x + gap / 2 - grow, y: y + S - gap / 2 - grow, width: S - gap + 2 * grow, height: gap + 2 * grow, fill }, parent);
      }
      if (c % GRID < GRID - 1 && set.has(right) && set.has(down) && set.has(down + 1)) {
        el('rect', { x: x + S - gap / 2 - grow, y: y + S - gap / 2 - grow, width: gap + 2 * grow, height: gap + 2 * grow, fill }, parent);
      }
    }
  };
  if (selected !== null && selected !== undefined) {
    const r = floor.rooms.find((q) => q.index === selected);
    if (r && visibleRooms.includes(r)) drawCells(r, gRooms, css('--sel'), Math.max(1.5, S * 0.05));
  }

  // doors (under rooms so they read as bridges)
  const gDoors = el('g', {}, svg);
  for (const d of floor.doors) {
    const ra = byCell.get(d.a), rb = byCell.get(d.b);
    if (!ra || !rb) continue;
    const [xa, ya] = pos(d.a), [xb, yb] = pos(d.b);
    const w = S * 0.26;
    const col = css('--room-edge');
    if (ya === yb) {
      const x = Math.min(xa, xb) + S - gap / 2 - 2;
      el('rect', { x, y: ya + S / 2 - w / 2, width: gap + 4, height: w, fill: col }, gDoors);
    } else {
      const y = Math.min(ya, yb) + S - gap / 2 - 2;
      el('rect', { x: xa + S / 2 - w / 2, y, width: w, height: gap + 4, fill: col }, gDoors);
    }
  }

  // rooms
  for (const r of visibleRooms) {
    const g = el('g', { class: interactive ? 'room' : '', 'data-index': r.index }, gRooms);
    const attrs = r.hidden ? { stroke: css('--label'), 'stroke-dasharray': `${S * 0.08} ${S * 0.06}`, 'stroke-width': Math.max(1, S * 0.03) } : {};
    drawCells(r, g, roomColor(r), 0, attrs);
    if (labels) {
      let cx = 0, cy = 0;
      for (const c of r.cells) { const [x, y] = pos(c); cx += x + S / 2; cy += y + S / 2; }
      cx /= r.cells.length; cy /= r.cells.length;
      const text = depth ? String(r.depth) : r.label;
      if (text) {
        const t = el('text', { x: cx, y: cy + S * 0.13, 'text-anchor': 'middle', 'font-size': S * 0.36,
          'font-weight': 700, fill: css('--label') }, g);
        t.textContent = text;
      }
    }
    if (interactive) {
      const title = el('title', {}, g);
      title.textContent = `${r.type_name}：${r.name}（布局 ${r.variant}）`;
      g.addEventListener('click', () => selectRoom(r.index));
    }
  }
}

// ---------------------------------------------------------------------------- page sections
function render() {
  const d = state.data;
  $('app').hidden = false;
  const modeText = d.mode === 'debug' ? '调试开局' : '正常开局';
  const parts = [
    ['', `${d.game_name} ${d.version}`], ['种子 ', d.seed.text], ['数字 ', String(d.seed.value)], ['', modeText],
    ['', `${d.floors.length} 层，${d.floors.reduce((a, f) => a + f.rooms.length, 0)} 个房间，${d.ms} ms`],
  ];
  const line = parts.map(([k, v]) => {
    const s = html('span', {}, k);
    const b = html('b', {}, v);
    s.appendChild(b);
    return s;
  });
  if (!d.validated) {
    line.splice(1, 0, html('span', { class: 'badge', title: '静态移植，只做过一致性检查，还没和游戏本体核对' }, '未经游戏验证'));
  }
  $('seedline').replaceChildren(...line);

  const nav = $('floors');
  nav.replaceChildren();
  d.floors.forEach((f, i) => {
    const btn = html('button', { class: 'floorbtn', type: 'button', 'aria-current': String(i === state.floor) });
    const svg = document.createElementNS(SVGNS, 'svg');
    drawFloor(svg, f, { S: 12, gap: 2, labels: false, player: false, showHidden: true, heat: false, interactive: false });
    btn.appendChild(svg);
    btn.appendChild(html('span', { class: 'fname' }, f.name));
    const meta = [`${f.rooms.length} 间`].concat(f.curses.length ? f.curses : []).join(' · ');
    btn.appendChild(html('span', { class: 'fmeta' }, meta));
    btn.addEventListener('click', () => { state.floor = i; state.room = null; render(); });
    nav.appendChild(btn);
  });
  renderFloor();
}

function renderFloor() {
  const f = state.data.floors[state.floor];
  const player = $('t-player').checked;
  $('t-hidden').disabled = player;
  $('t-hidden').parentElement.style.opacity = player ? 0.5 : 1;
  $('floortitle').textContent = `${f.name}  ·  ${f.name_en}`;
  drawFloor($('map'), f, {
    S: 56, gap: 6, labels: true, player, showHidden: $('t-hidden').checked || player, depth: $('t-depth').checked,
    heat: player, selected: state.room, interactive: true,
  });
  renderLegend(player);
  renderFloorInfo(f);
  if (state.room === null) {
    $('roominfo').replaceChildren(html('p', { class: 'muted' }, '点击地图上的房间查看详情和布局。'));
  }
}

function renderLegend(player) {
  const rep = state.data.game === 'repplus';
  const items = LEGEND.concat(rep ? LEGEND_REP : []);
  if (player) {
    for (const [, , col, name] of HEAT) {
      if (rep || col !== '--heat-us') items.push([col, name]);
    }
  }
  $('legend').replaceChildren(...items.map(([v, name]) => {
    const s = html('span');
    const i = html('i');
    i.style.background = css(v);
    s.append(i, name);
    return s;
  }));
}

function kvList(rows) {
  const dl = html('dl', { class: 'kv' });
  for (const [k, v, cls] of rows) {
    dl.appendChild(html('dt', {}, k));
    const dd = html('dd', cls ? { class: cls } : {});
    if (v instanceof Node) dd.appendChild(v); else dd.textContent = v;
    dl.appendChild(dd);
  }
  return dl;
}

function renderFloorInfo(f) {
  const counts = new Map();
  for (const r of f.rooms) counts.set(r.type_name, (counts.get(r.type_name) || 0) + 1);
  const tags = html('div');
  for (const [name, n] of counts) tags.appendChild(html('span', { class: 'tag' }, n > 1 ? `${name} ×${n}` : name));
  const curses = f.curses.length ? f.curses.join('、') : '无';
  const bosses = f.bosses.map((b) => `${b.name}（布局 ${b.variant}）`).join('；');
  const card = $('floorinfo');
  card.replaceChildren(
    html('h3', {}, `第 ${f.stage} 层 · 类型 ${f.stage_type}`),
    kvList([
      ['楼层种子', `${f.stage_seed}（${hex(f.stage_seed)}）`, 'mono'],
      ['诅咒', curses], ['头目', bosses], ['房间数', String(f.rooms.length)],
      ['生成尝试', `${f.attempts} 次`], ['房间构成', tags],
    ]),
  );
}

async function selectRoom(index) {
  state.room = index;
  renderFloor();
  const f = state.data.floors[state.floor];
  const r = f.rooms.find((q) => q.index === index);
  const box = $('roominfo');
  const x = r.cells[0] % GRID, y = Math.floor(r.cells[0] / GRID);
  const layoutSlots = [];
  for (let s = 0; s < 8; s++) if (r.layout_doors >> s & 1) layoutSlots.push(['左', '上', '右', '下', '左2', '上2', '右2', '下2'][s]);
  box.replaceChildren(
    html('h3', {}, `${r.type_name}：${r.name}`),
    kvList([
      ['布局', `类型 ${r.type} · 变体 ${r.variant} · 子类型 ${r.subtype}`],
      ['形状', r.shape_name], ['位置', `(${r.x}, ${r.y})，列表第 ${r.index} 个`],
      ['难度 / 权重', `${r.difficulty} / ${r.weight}`],
      ['实际门', r.door_names.length ? r.door_names.join(' ') : '无'],
      ['布局门位', layoutSlots.join(' ') || '无'],
      ['生成深度', r.depth >= 0 ? String(r.depth) : '—'],
      ['装饰种子', `${r.seeds.decoration}`, 'mono'], ['刷怪种子', `${r.seeds.spawn}`, 'mono'],
      ['奖励种子', `${r.seeds.award}`, 'mono'],
    ]),
    html('div', { class: 'layoutbox', id: 'layoutbox' }, '加载布局…'),
  );
  try {
    const game = state.data.game;
    const key = `${game}.${r.file}.${r.type}.${r.variant}`;
    let lay = state.layoutCache.get(key);
    if (!lay) {
      const res = await fetch(`/api/layout?game=${game}&stage=${r.file}&type=${r.type}&variant=${r.variant}`);
      lay = await res.json();
      if (!res.ok) throw new Error(lay.error || res.statusText);
      state.layoutCache.set(key, lay);
    }
    if (state.room === index) drawLayout(lay, r);
  } catch (err) {
    $('layoutbox').textContent = '布局加载失败：' + err.message;
  }
}

function drawLayout(lay, room) {
  const box = $('layoutbox');
  box.replaceChildren();
  const T = 16, W = lay.width, H = lay.height;
  const svg = el('svg', { viewBox: `0 0 ${(W + 2) * T} ${(H + 2) * T}`, role: 'img', 'aria-label': '房间布局' });
  el('rect', { x: T, y: T, width: W * T, height: H * T, fill: css('--tile-floor') }, svg);
  for (const [mx, my, mw, mh] of lay.missing) {
    el('rect', { x: (mx + 1) * T, y: (my + 1) * T, width: mw * T, height: mh * T, fill: css('--tile-wall') }, svg);
  }
  for (const [dx, dy, bit] of lay.door_list) {
    const on = (room.doors & bit) !== 0;
    el('rect', { x: (dx + 1) * T + 2, y: (dy + 1) * T + 2, width: T - 4, height: T - 4, rx: 2,
      fill: on ? '#4fbf6a' : '#6b6356' }, svg).appendChild(Object.assign(document.createElementNS(SVGNS, 'title'),
      { textContent: on ? '门（通向相邻房间）' : '布局门位（这边没有房间）' }));
  }
  const tally = new Map();
  for (const sp of lay.spawns) {
    const total = sp.entries.reduce((a, e) => a + e.weight, 0) || 1;
    const shown = [...sp.entries].sort((a, b) => b.weight - a.weight).find((e) => e.kind !== 'none') || sp.entries[0];
    const [col] = KIND[shown.kind] || ['#999'];
    const x = (sp.x + 1) * T, y = (sp.y + 1) * T;
    let node;
    if (['pickup', 'enemy', 'object'].includes(shown.kind)) {
      node = el('circle', { cx: x + T / 2, cy: y + T / 2, r: T * 0.36, fill: col, stroke: '#000', 'stroke-opacity': 0.4 }, svg);
    } else if (shown.kind === 'none') {
      node = el('circle', { cx: x + T / 2, cy: y + T / 2, r: T * 0.12, fill: '#999' }, svg);
    } else {
      node = el('rect', { x: x + 1, y: y + 1, width: T - 2, height: T - 2, rx: 2, fill: col,
        stroke: shown.kind === 'pit' ? '#6b6356' : 'none' }, svg);
    }
    const title = document.createElementNS(SVGNS, 'title');
    title.textContent = sp.entries.map((e) => `${e.name}（${e.type}.${e.variant}.${e.subtype}）${sp.entries.length > 1 ? ' ' + Math.round(100 * e.weight / total) + '%' : ''}`).join('\n');
    node.appendChild(title);
    if (sp.entries.length > 1) {
      el('circle', { cx: x + T - 3, cy: y + 3, r: 2.6, fill: '#fff' }, svg);
    }
    const key = sp.entries.length > 1 ? sp.entries.map((e) => e.name).join(' / ') + '（随机）' : shown.name;
    tally.set(key, (tally.get(key) || 0) + 1);
  }
  box.appendChild(svg);
  box.appendChild(html('div', { class: 'caption' },
    `布局文件：${lay.file} · ${lay.width}×${lay.height} 格。白点表示该刷怪点有多个候选（悬停查看概率）。具体刷哪个由房间种子决定，这一步还没翻译。`));
  const ul = html('ul', { class: 'spawnlist' });
  for (const [name, n] of [...tally].sort((a, b) => b[1] - a[1])) ul.appendChild(html('li', {}, `${name} ×${n}`));
  box.appendChild(ul);
}

// ---------------------------------------------------------------------------- wiring
$('controls').addEventListener('submit', (e) => { e.preventDefault(); generate(); });
$('random').addEventListener('click', async () => {
  const res = await fetch('/api/random');
  const d = await res.json();
  $('seed').value = d.text;
  generate();
});
for (const id of ['t-hidden', 't-player', 't-depth']) {
  $(id).addEventListener('change', () => { if (state.data) renderFloor(); });
}
$('game').addEventListener('change', () => {
  applyGame();
  if ($('seed').value.trim()) generate(true);
});
document.addEventListener('keydown', (e) => {
  if (!state.data || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowRight' && state.floor < state.data.floors.length - 1) { state.floor++; state.room = null; render(); }
  if (e.key === 'ArrowLeft' && state.floor > 0) { state.floor--; state.room = null; render(); }
});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.data) render(); });

const initial = new URLSearchParams(location.search);
writeParams(initial);
applyGame();
if (initial.get('seed')) generate(true);
