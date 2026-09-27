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
// the game's minimap icon (gfx/ui/minimap_icons.anm2, AB+: minimap1.anm2) for each room type
const ICONS = {
  2: 'IconShop', 4: 'IconTreasureRoom', 5: 'IconBoss', 6: 'IconMiniboss', 7: 'IconSecretRoom',
  8: 'IconSuperSecretRoom', 9: 'IconArcade', 10: 'IconCurseRoom', 11: 'IconAmbushRoom', 12: 'IconLibrary',
  13: 'IconSacrificeRoom', 14: 'IconDevilRoom', 15: 'IconAngelRoom', 18: 'IconIsaacsRoom', 19: 'IconBarrenRoom',
  20: 'IconChestRoom', 21: 'IconDiceRoom', 24: 'IconPlanetarium', 29: 'IconUltraSecretRoom',
};
// legend: [colour (text style), icon type (icon style), name]; type 0/-1/-2 = start / normal / hidden tile
const LEGEND = [
  ['--t-start', 0, '起始房间'], ['--room', -1, '普通'], ['--t-boss', 5, '头目房'], ['--t-treasure', 4, '宝箱房'],
  ['--t-shop', 2, '商店'], ['--t-secret', 7, '隐藏房'], ['--t-supersecret', 8, '超级隐藏房'], ['--t-curse', 10, '诅咒房'],
  ['--t-miniboss', 6, '小头目房'], ['--t-challenge', 11, '挑战房'], ['--t-library', 12, '图书馆'],
  ['--t-sacrifice', 13, '献祭房'], ['--t-arcade', 9, '赌博房'], ['--t-vault', 20, '宝库'], ['--t-dice', 21, '骰子房'],
  ['--t-bedroom', 18, '卧室'],
];
const LEGEND_REP = [['--t-planetarium', 24, '星象房'], ['--t-ultrasecret', 29, '究极隐藏房']];
// hidden-room heat layers: [key in infer(), map tag, colour, legend]
const HEAT = [
  ['secret', '隐', '--heat', '隐藏房概率'],
  ['super', '超', '--heat-ss', '超级隐藏房概率'],
  ['ultra', '究', '--heat-us', '究极隐藏房概率'],
];
// what the player saw in a cell -> label, and the room type drawn for it
const OBS = {
  empty: ['炸过/打开过，是空的', null],
  red: ['开了红房间，没有门通到隐藏房', null],
  secret: ['隐藏房在这里', 7],
  super: ['超级隐藏房在这里', 8],
  ultra: ['究极隐藏房在这里', 29],
};
const HIDDEN_NAME = { secret: '隐藏房', super: '超级隐藏房', ultra: '究极隐藏房' };
const ROOM_LABEL = { 7: '隐', 8: '超', 29: '究' };
const GAME_TEXT = {
  abplus: { sub: '输入种子，离线生成《以撒的结合：胎衣†》v1.06 每一层的地图', title: '胎衣†' },
  repplus: { sub: '输入种子，离线生成《以撒的结合：忏悔+》v1.9.7.17 每一层的地图', title: '忏悔+' },
};
const KIND = {
  rock: ['#8a8175', '石头类'], poop: ['#7a5230', '大便'], tnt: ['#b0412e', '炸药桶'], block: ['#9aa3ad', '方块'],
  pit: ['#050404', '沟壑'], spikes: ['#7d2020', '地刺'], web: ['#cfcfcf', '蛛网'], plate: ['#3d6fb0', '按钮'],
  door: ['#3d6fb0', '活板门/暗门'], deco: ['#5a5246', '装饰'], grid: ['#777777', '网格物体'],
  pickup: ['#f2c33b', '拾取物'], enemy: ['#e2553f', '敌人'], object: ['#3fb0a8', '机器/火堆等'],
};
const TEXT_STYLE_KEY = 'isaac-mapgen-text-labels';
const ICON_SCALE = 6;           // minimap pixels -> map units: a 9x8 cell becomes 54x48

const state = {
  data: null, floor: 0, room: null, layoutCache: new Map(), obs: new Map(),
  sprites: new Map(),           // game -> Promise of {tiles, icons} (null if unavailable)
  sheet: null,                  // sprites of the game on screen
};

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
function iconName(type, subtype) {
  return type === 11 && subtype === 1 ? 'IconBossAmbushRoom' : ICONS[type];
}
// one frame of a sprite sheet, drawn at scale k with crisp pixels
function sprite(parent, sheet, frame, x, y, k, attrs = {}) {
  const [fx, fy, fw, fh] = frame;
  const s = el('svg', { x, y, width: fw * k, height: fh * k, viewBox: `${fx} ${fy} ${fw} ${fh}`, ...attrs }, parent);
  el('image', { href: sheet.url, width: sheet.w, height: sheet.h, class: 'pixel' }, s);
  return s;
}
function textStyle() {
  try { return localStorage.getItem(TEXT_STYLE_KEY) === '1'; } catch { return false; }
}
function useIcons() {
  return !$('t-text').checked && !!state.sheet;
}

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
function loadSprites(game) {
  if (!state.sprites.has(game)) {
    state.sprites.set(game, fetch(`/api/sprites?game=${game}`)
      .then((res) => (res.ok ? res.json() : null)).catch(() => null));
  }
  return state.sprites.get(game);
}

async function generate(keepFloor = false) {
  const params = readParams();
  if (!params.seed) { setStatus('请输入种子，或点"随机"。', true); return; }
  $('go').disabled = true;
  setStatus('生成中…');
  try {
    const qs = new URLSearchParams(params);
    const [res, sheet] = await Promise.all([fetch('/api/run?' + qs.toString()), loadSprites(params.game)]);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    history.replaceState(null, '', '?' + qs.toString());
    state.data = data;
    state.sheet = sheet;
    state.obs = new Map();
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

// ---------------------------------------------------------------------------- linked inference
// f.joint rows are [secret cell, super secret cell, ultra secret cell, p] (-1: no such room), the
// joint posterior given the visible map. The player's observations only remove rows, so the
// update is exact: keep the consistent rows and renormalise.
function neighbours(c) {
  const x = c % GRID, y = Math.floor(c / GRID);
  return TRAVEL.map(([dx, dy]) => [x + dx, y + dy])
    .filter(([a, b]) => a >= 0 && a < GRID && b >= 0 && b < GRID).map(([a, b]) => a + b * GRID);
}
function floorObs(i = state.floor) {
  if (!state.obs.has(i)) state.obs.set(i, new Map());
  return state.obs.get(i);
}
function consistent(row, cell, kind) {
  const [c, h, u] = row;
  if (kind === 'secret') return c === cell;
  if (kind === 'super') return h === cell;
  if (kind === 'ultra') return u === cell;
  if (c === cell || h === cell || u === cell) return false;
  if (kind !== 'red') return true;
  // Level::MakeRedRoomDoor (RVA 0x34D010) gives the new red room (a 1x1 layout with all four doors) a
  // door towards every adjacent room, so no door into a hidden room means none is next to it
  const nb = neighbours(cell);
  return ![c, h, u].some((x) => x >= 0 && nb.includes(x));
}
function infer(f, obs, extra = null) {
  const checks = [...obs];
  if (extra) checks.push(extra);
  let total = 0;
  const rows = f.joint.filter((row) => checks.every(([cell, kind]) => consistent(row, cell, kind)));
  for (const row of rows) total += row[3];
  const out = { ok: total > 0, secret: new Map(), super: new Map(), ultra: new Map() };
  if (!out.ok) return out;
  const keys = ['secret', 'super', 'ultra'];
  for (const row of rows) {
    for (let k = 0; k < 3; k++) {
      if (row[k] >= 0) out[keys[k]].set(row[k], (out[keys[k]].get(row[k]) || 0) + row[3] / total);
    }
  }
  return out;
}
function best(map) {
  let top = null;
  for (const [c, p] of map) if (!top || p > top[1] + 1e-12 || (Math.abs(p - top[1]) <= 1e-12 && c < top[0])) top = [c, p];
  return top;
}
function pct(p) {
  if (p >= 0.995) return '100%';
  if (p > 0 && p < 0.01) return '<1%';
  return `${Math.round(p * 100)}%`;
}
function where(c, f) {
  const dx = c % GRID - f.start % GRID, dy = Math.floor(c / GRID) - Math.floor(f.start / GRID);
  return '起点' + (dx ? (dx > 0 ? '右' : '左') + Math.abs(dx) : '') + (dy ? (dy > 0 ? '下' : '上') + Math.abs(dy) : '');
}
function foundCell(obs, kind) {
  for (const [c, k] of obs) if (k === kind) return c;
  return null;
}
function nextBomb(inf, obs) {
  const score = new Map();
  for (const k of ['secret', 'super']) {
    for (const [c, p] of inf[k]) if (!obs.has(c)) score.set(c, (score.get(c) || 0) + p);
  }
  return best(score);
}
function nextRedKey(f, inf, obs) {
  if (!inf.ultra.size || foundCell(obs, 'ultra') !== null) return null;
  const score = new Map();
  for (const e of f.door_targets || []) {
    if (obs.has(e)) continue;
    const p = neighbours(e).reduce((a, n) => a + (inf.ultra.get(n) || 0), 0);
    if (p > 0) score.set(e, p);
  }
  return best(score);
}
// "if the hidden room were here" read-outs for one cell
function hypotheses(f, obs, inf, c) {
  const rep = state.data.game === 'repplus';
  const p = { secret: inf.secret.get(c) || 0, super: inf.super.get(c) || 0, ultra: inf.ultra.get(c) || 0 };
  const lines = [`${where(c, f)}：隐 ${pct(p.secret)} · 超 ${pct(p.super)}` + (rep ? ` · 究 ${pct(p.ultra)}` : '')];
  const say = (kind, head, targets) => {
    const h = infer(f, obs, [c, kind]);
    if (!h.ok) return;
    const parts = [];
    for (const k of targets) {
      if (foundCell(obs, k) !== null) continue;
      const b = best(h[k]);
      if (b) parts.push(`${HIDDEN_NAME[k]}最可能在${where(b[0], f)}（${pct(b[1])}）`);
    }
    if (parts.length) lines.push(`${head}：${parts.join('，')}`);
  };
  const others = (k) => ['secret', 'super'].concat(rep ? ['ultra'] : []).filter((q) => q !== k);
  for (const k of ['secret', 'super', 'ultra']) {
    if (p[k] > 0.001 && p[k] < 0.999) say(k, `若${HIDDEN_NAME[k]}在这里`, others(k));
  }
  if (p.secret + p.super > 0.001 && p.secret + p.super < 0.999) say('empty', '若炸开是空的', ['secret', 'super'].concat(rep ? ['ultra'] : []));
  return lines;
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

// Two styles. Icon style (opts.sprites) draws the game's own minimap: 9x8-pixel room tiles for each
// shape (visited; the start room as the current room; hidden rooms as unvisited) with the room icons
// on top, scaled by an integer so the pixels stay crisp. Text style draws coloured rooms with labels.
function drawFloor(svg, floor, opts) {
  const { labels, player, depth, heat, selected, interactive } = opts;
  const sheet = opts.sprites || null;
  const k = ICON_SCALE;
  const Sx = sheet ? 9 * k : opts.S, Sy = sheet ? 8 * k : opts.S;
  const S = Math.min(Sx, Sy);
  const gap = sheet ? 0 : opts.gap;
  const inset = sheet ? k / 2 : gap / 2;
  svg.replaceChildren();
  const b = bounds(floor);
  svg.setAttribute('viewBox', `0 0 ${b.cols * Sx} ${b.rows * Sy}`);
  const pos = (c) => [(c % GRID - b.x0) * Sx, (Math.floor(c / GRID) - b.y0) * Sy];
  const centre = (r) => {
    let cx = 0, cy = 0;
    for (const c of r.cells) { const [x, y] = pos(c); cx += x + Sx / 2; cy += y + Sy / 2; }
    return [cx / r.cells.length, cy / r.cells.length];
  };
  const cellRect = (c, parent, attrs) => {
    const [x, y] = pos(c);
    return el('rect', { x: x + inset, y: y + inset, width: Sx - 2 * inset, height: Sy - 2 * inset, rx: S * 0.08, ...attrs }, parent);
  };
  const hidden = (r) => r.hidden && player;
  const visibleRooms = floor.rooms.filter((r) => !hidden(r) && (opts.showHidden || !r.hidden));
  const byCell = new Map();
  for (const r of visibleRooms) for (const c of r.cells) byCell.set(c, r);

  // empty grid
  const g0 = el('g', {}, svg);
  for (let yy = 0; yy < b.rows; yy++) {
    for (let xx = 0; xx < b.cols; xx++) cellRect((yy + b.y0) * GRID + xx + b.x0, g0, { fill: css('--cell-empty') });
  }

  // heat map of hidden-room probabilities (player view), after the player's observations
  const obs = opts.obs || new Map();
  if (heat && opts.post) {
    const gh = el('g', {}, svg);
    const post = opts.post;
    const cells = new Set(HEAT.flatMap(([key]) => [...post[key].keys()]));
    for (const c of cells) {
      if (byCell.has(c) || obs.has(c)) continue;
      const [x, y] = pos(c);
      const lines = HEAT.map(([key, tag, col]) => [tag, post[key].get(c) || 0, col]).filter(([, p]) => p >= 0.01);
      if (!lines.length) continue;
      const main = lines.reduce((a, q) => (q[1] > a[1] ? q : a));
      cellRect(c, gh, { fill: css(main[2]), 'fill-opacity': 0.12 + 0.6 * Math.min(1, main[1] / 0.5) });
      lines.forEach(([tag, p], i) => {
        const t = el('text', { x: x + Sx / 2, y: y + Sy / 2 + (i - (lines.length - 1) / 2) * S * 0.26 + S * 0.08,
          'text-anchor': 'middle', 'font-size': S * 0.22, fill: css('--label'), 'font-weight': 600 }, gh);
        t.textContent = `${tag}${Math.round(p * 100)}%`;
      });
    }
  }

  const gRooms = el('g', {}, svg);
  const selRoom = selected !== null && selected !== undefined ? floor.rooms.find((q) => q.index === selected) : null;
  if (sheet) {
    for (const r of visibleRooms) {
      const g = el('g', { class: interactive ? 'room' : '', 'data-index': r.index }, gRooms);
      const tiles = sheet.tiles.frames[r.start ? 'RoomCurrent' : r.hidden ? 'RoomUnvisited' : 'RoomVisited'];
      const [x, y] = pos(r.y * GRID + r.x);
      sprite(g, sheet.tiles, tiles[r.shape - 1], x, y, k, { class: 'cell' });
      const [cx, cy] = centre(r);
      const icon = sheet.icons.frames[iconName(r.type, r.subtype)];
      if (icon) sprite(g, sheet.icons, icon, cx - 6.5 * k, cy - 6 * k, k);   // glyph sits on the 9x8 tile
      if (labels && depth) {
        const t = el('text', { x: cx, y: cy + S * 0.13, 'text-anchor': 'middle', 'font-size': S * 0.36,
          'font-weight': 700, fill: css('--label'), class: 'outlined' }, g);
        t.textContent = String(r.depth);
      }
      if (interactive) {
        el('title', {}, g).textContent = `${r.type_name}：${r.name}（布局 ${r.variant}）`;
        g.addEventListener('click', () => selectRoom(r.index));
      }
    }
    // selection: outline the room's cells
    if (selRoom && visibleRooms.includes(selRoom)) {
      const set = new Set(selRoom.cells);
      const w = Math.max(2, k * 0.6);
      for (const c of selRoom.cells) {
        const [x, y] = pos(c);
        const edges = [[c - 1, x, y, x, y + Sy], [c - GRID, x, y, x + Sx, y], [c + 1, x + Sx, y, x + Sx, y + Sy],
          [c + GRID, x, y + Sy, x + Sx, y + Sy]];
        for (const [n, x1, y1, x2, y2] of edges) {
          const outside = !set.has(n) || (n === c - 1 && c % GRID === 0) || (n === c + 1 && c % GRID === GRID - 1);
          if (outside) el('line', { x1, y1, x2, y2, stroke: css('--sel'), 'stroke-width': w, 'stroke-linecap': 'square' }, gRooms);
        }
      }
    }
  } else {
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
    // selection halo
    if (selRoom && visibleRooms.includes(selRoom)) drawCells(selRoom, gRooms, css('--sel'), Math.max(1.5, S * 0.05));
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
    for (const r of visibleRooms) {
      const g = el('g', { class: interactive ? 'room' : '', 'data-index': r.index }, gRooms);
      const attrs = r.hidden ? { stroke: css('--label'), 'stroke-dasharray': `${S * 0.08} ${S * 0.06}`, 'stroke-width': Math.max(1, S * 0.03) } : {};
      drawCells(r, g, roomColor(r), 0, attrs);
      if (labels) {
        const [cx, cy] = centre(r);
        const text = depth ? String(r.depth) : r.label;
        if (text) {
          const t = el('text', { x: cx, y: cy + S * 0.13, 'text-anchor': 'middle', 'font-size': S * 0.36,
            'font-weight': 700, fill: css('--label') }, g);
          t.textContent = text;
        }
      }
      if (interactive) {
        el('title', {}, g).textContent = `${r.type_name}：${r.name}（布局 ${r.variant}）`;
        g.addEventListener('click', () => selectRoom(r.index));
      }
    }
  }

  if (!heat || !opts.post) return;
  // the player's marks: empty cells, red rooms, hidden rooms found
  const gMarks = el('g', {}, svg);
  for (const [c, kind] of obs) {
    const [x, y] = pos(c);
    const type = OBS[kind][1];
    if (sheet && type) {
      sprite(gMarks, sheet.tiles, sheet.tiles.frames.RoomVisited[0], x, y, k);
      sprite(gMarks, sheet.icons, sheet.icons.frames[ICONS[type]], x - 2 * k, y - 2 * k, k);
      continue;
    }
    const fill = type ? css(TYPE_COLOR[type]) : kind === 'red' ? '#7a1f2b' : css('--cell-empty');
    cellRect(c, gMarks, { fill, stroke: css('--label'), 'stroke-width': Math.max(1, S * 0.03) });
    const t = el('text', { x: x + Sx / 2, y: y + Sy / 2 + S * 0.13, 'text-anchor': 'middle', 'font-size': S * 0.36,
      'font-weight': 700, fill: css('--label') }, gMarks);
    t.textContent = type ? ROOM_LABEL[type] : kind === 'red' ? '红' : '×';
  }
  // suggestions: the wall to bomb next, the cell to open with the Red Key
  const ring = (c, col, tag, corner) => {
    const [x, y] = pos(c);
    el('rect', { x: x + inset - 1, y: y + inset - 1, width: Sx - 2 * inset + 2, height: Sy - 2 * inset + 2, rx: S * 0.1,
      fill: 'none', stroke: col, 'stroke-width': Math.max(2, S * 0.06) }, gMarks);
    const tx = corner ? x + Sx - inset - S * 0.13 : x + inset + S * 0.13;
    el('circle', { cx: tx, cy: y + inset + S * 0.13, r: S * 0.15, fill: col }, gMarks);
    const t = el('text', { x: tx, y: y + inset + S * 0.19, 'text-anchor': 'middle', 'font-size': S * 0.17,
      'font-weight': 700, fill: '#1b1814' }, gMarks);
    t.textContent = tag;
  };
  if (opts.suggest?.bomb) ring(opts.suggest.bomb[0], css('--suggest'), '炸', false);
  if (opts.suggest?.red) ring(opts.suggest.red[0], css('--t-ultrasecret'), '钥', true);
  // every empty cell can be clicked to record what was found there
  if (!interactive) return;
  const gHit = el('g', {}, svg);
  for (let yy = 0; yy < b.rows; yy++) {
    for (let xx = 0; xx < b.cols; xx++) {
      const c = (yy + b.y0) * GRID + xx + b.x0;
      if (byCell.has(c)) continue;
      const hit = el('rect', { x: xx * Sx, y: yy * Sy, width: Sx, height: Sy, fill: 'transparent', class: 'hitcell',
        'data-cell': c }, gHit);
      el('title', {}, hit).textContent = opts.tip ? opts.tip(c) : '';
      hit.addEventListener('click', (e) => { e.stopPropagation(); openCellMenu(c, e); });
    }
  }
}

// ---------------------------------------------------------------------------- page sections
function render() {
  const d = state.data;
  $('app').hidden = false;
  const parts = [
    ['', `${d.game_name} ${d.version}`], ['种子 ', d.seed.text], ['', d.mode === 'debug' ? '调试开局' : '正常开局'],
    ['', `${d.floors.length} 层，${d.floors.reduce((a, f) => a + f.rooms.length, 0)} 个房间`],
  ];
  $('seedline').replaceChildren(...parts.map(([k, v]) => {
    const s = html('span', {}, k);
    s.appendChild(html('b', {}, v));
    return s;
  }));

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
  $('t-text').disabled = !state.sheet;
  $('floortitle').textContent = `${f.name}  ·  ${f.name_en}`;
  closeCellMenu();
  const icons = useIcons();
  const opts = {
    S: 56, gap: 6, labels: true, player, showHidden: $('t-hidden').checked || player, depth: $('t-depth').checked,
    heat: player, selected: state.room, interactive: true, sprites: icons ? state.sheet : null,
  };
  if (player) {
    const obs = floorObs();
    const inf = infer(f, obs);
    Object.assign(opts, {
      post: inf, obs,
      suggest: { bomb: nextBomb(inf, obs), red: state.data.game === 'repplus' ? nextRedKey(f, inf, obs) : null },
      tip: (c) => {
        const lines = hypotheses(f, obs, inf, c);
        return lines.concat(obs.has(c) ? [`已标记：${OBS[obs.get(c)][0]}`] : [], ['点击记下这里的结果']).join('\n');
      },
    });
    renderLinked(f, inf, obs);
  }
  $('linkinfo').hidden = !player;
  drawFloor($('map'), f, opts);
  renderLegend(player, icons);
  renderFloorInfo(f);
  if (state.room === null) {
    $('roominfo').replaceChildren(html('p', { class: 'muted' }, '点击地图上的房间查看详情和布局。'));
  }
}

function legendIcon(type) {
  const sheet = state.sheet, k = 2;
  const svg = el('svg', { width: 9 * k, height: 8 * k, viewBox: `0 0 ${9 * k} ${8 * k}`, 'aria-hidden': 'true' });
  const tile = type === 0 ? 'RoomCurrent' : type === -2 ? 'RoomUnvisited' : 'RoomVisited';
  sprite(svg, sheet.tiles, sheet.tiles.frames[tile][0], 0, 0, k);
  const icon = type > 0 ? sheet.icons.frames[ICONS[type]] : null;
  if (icon) sprite(svg, sheet.icons, icon, -2 * k, -2 * k, k);
  return svg;
}

function renderLegend(player, icons) {
  const rep = state.data.game === 'repplus';
  const items = LEGEND.concat(rep ? LEGEND_REP : []);
  if (icons && $('t-hidden').checked && !player) items.splice(2, 0, [null, -2, '要找到才显示的房间']);
  const nodes = items.map(([col, type, name]) => {
    const s = html('span');
    if (icons) {
      s.append(legendIcon(type), name);
    } else if (col) {
      const i = html('i');
      i.style.background = css(col);
      s.append(i, name);
    }
    return s;
  }).filter((s) => s.childNodes.length);
  if (player) {
    for (const [, , col, name] of HEAT) {
      if (!rep && col === '--heat-us') continue;
      const s = html('span');
      const i = html('i');
      i.style.background = css(col);
      s.append(i, name);
      nodes.push(s);
    }
  }
  $('legend').replaceChildren(...nodes);
}

function renderLinked(f, inf, obs) {
  const rep = state.data.game === 'repplus';
  const rows = [];
  for (const k of ['secret', 'super'].concat(rep ? ['ultra'] : [])) {
    const fc = foundCell(obs, k);
    const b = best(inf[k]);
    rows.push([HIDDEN_NAME[k], fc !== null ? `已找到：${where(fc, f)}` : b ? `最可能在${where(b[0], f)}（${pct(b[1])}）` : '这层没有']);
  }
  const bomb = nextBomb(inf, obs);
  rows.push(['下一炸', bomb ? `${where(bomb[0], f)}，炸出隐藏房或超级隐藏房的概率 ${pct(bomb[1])}` : '都找到了']);
  if (rep) {
    const key = nextRedKey(f, inf, obs);
    if (key) rows.push(['红钥匙', `在${where(key[0], f)}开红房间，通到究极隐藏房的概率 ${pct(key[1])}`]);
  }
  const children = [html('h3', {}, '隐藏房联动推断'), kvList(rows)];
  if (obs.size) {
    const ul = html('ul', { class: 'obslist' });
    for (const [c, kind] of obs) {
      const li = html('li', {}, `${where(c, f)}：${OBS[kind][0]}`);
      const del = html('button', { type: 'button', class: 'mini', title: '删除这个标记' }, '×');
      del.addEventListener('click', () => { obs.delete(c); renderFloor(); });
      li.appendChild(del);
      ul.appendChild(li);
    }
    const clear = html('button', { type: 'button', class: 'mini wide' }, '清除本层全部标记');
    clear.addEventListener('click', () => { obs.clear(); renderFloor(); });
    children.push(html('h4', {}, `已标记 ${obs.size} 格`), ul, clear);
  }
  children.push(html('p', { class: 'hint' }, rep
    ? '点地图上的空格子，记下炸墙或开红房间的结果，三种隐藏房的概率会一起更新。究极隐藏房离隐藏房、超级隐藏房都至少 3 格，所以一边的概率升高，另一边附近的候选就会下降。鼠标停在概率格上，可以看"假如在这里"时其余隐藏房最可能在哪。'
    : '点地图上的空格子，记下炸墙的结果，隐藏房和超级隐藏房的概率会一起更新。鼠标停在概率格上，可以看"假如在这里"时另一个隐藏房最可能在哪。'));
  $('linkinfo').replaceChildren(...children);
}

function closeCellMenu() {
  $('cellmenu').hidden = true;
}

function openCellMenu(c, evt) {
  const f = state.data.floors[state.floor];
  const rep = state.data.game === 'repplus';
  const obs = floorObs();
  const inf = infer(f, obs);
  const lines = hypotheses(f, obs, inf, c);
  const menu = $('cellmenu');
  const msg = html('p', { class: 'menumsg' });
  const children = [html('h4', {}, lines[0])];
  for (const l of lines.slice(1)) children.push(html('p', { class: 'hyp' }, l));
  const btns = html('div', { class: 'menubtns' });
  for (const kind of ['empty'].concat(rep ? ['red'] : [], ['secret', 'super'], rep ? ['ultra'] : [])) {
    const bt = html('button', { type: 'button', 'aria-pressed': String(obs.get(c) === kind) }, OBS[kind][0]);
    bt.addEventListener('click', () => {
      const prev = obs.get(c);
      if (prev === kind) { closeCellMenu(); return; }
      const moved = OBS[kind][1] ? foundCell(obs, kind) : null;   // a room found once: move the mark
      if (moved !== null) obs.delete(moved);
      obs.set(c, kind);
      if (!infer(f, obs).ok) {
        obs.delete(c);
        if (prev) obs.set(c, prev);
        if (moved !== null) obs.set(moved, kind);
        msg.textContent = '没有记下：按生成规则，加上已有的标记，这里不可能是这样。也可能是模型没考虑的情况，比如碎卡片带来的第二个隐藏房。';
        return;
      }
      renderFloor();
    });
    btns.appendChild(bt);
  }
  if (obs.has(c)) {
    const del = html('button', { type: 'button' }, '清除这个标记');
    del.addEventListener('click', () => { obs.delete(c); renderFloor(); });
    btns.appendChild(del);
  }
  const close = html('button', { type: 'button', class: 'ghost' }, '关闭');
  close.addEventListener('click', closeCellMenu);
  btns.appendChild(close);
  children.push(btns, msg);
  menu.replaceChildren(...children);
  menu.hidden = false;
  const wrap = $('mapwrap').getBoundingClientRect();
  const mw = menu.offsetWidth, mh = menu.offsetHeight;
  let left = evt.clientX - wrap.left + 12, top = evt.clientY - wrap.top + 12;
  if (left + mw > wrap.width - 4) left = Math.max(4, evt.clientX - wrap.left - mw - 12);
  if (top + mh > wrap.height - 4) top = Math.max(4, wrap.height - mh - 4);
  menu.style.left = `${left}px`;
  menu.style.top = `${top}px`;
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
  $('floorinfo').replaceChildren(
    html('h3', {}, `第 ${f.stage} 层 · 类型 ${f.stage_type}`),
    kvList([
      ['楼层种子', `${f.stage_seed}（${hex(f.stage_seed)}）`, 'mono'],
      ['诅咒', curses], ['头目', bosses], ['房间数', String(f.rooms.length)], ['房间构成', tags],
    ]),
  );
}

async function selectRoom(index) {
  state.room = index;
  renderFloor();
  const f = state.data.floors[state.floor];
  const r = f.rooms.find((q) => q.index === index);
  const box = $('roominfo');
  const layoutSlots = [];
  for (let s = 0; s < 8; s++) if (r.layout_doors >> s & 1) layoutSlots.push(['左', '上', '右', '下', '左2', '上2', '右2', '下2'][s]);
  box.replaceChildren(
    html('h3', {}, `${r.type_name}：${r.name}`),
    kvList([
      ['布局', `类型 ${r.type} · 变体 ${r.variant} · 子类型 ${r.subtype}`],
      ['形状', r.shape_name], ['位置', where(r.cells[0], f)],
      ['难度 / 权重', `${r.difficulty} / ${r.weight}`],
      ['实际门', r.door_names.length ? r.door_names.join(' ') : '无'],
      ['布局门位', layoutSlots.join(' ') || '无'],
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
    title.textContent = sp.entries.map((e) => `${e.name}${sp.entries.length > 1 ? ' ' + Math.round(100 * e.weight / total) + '%' : ''}`).join('\n');
    node.appendChild(title);
    if (sp.entries.length > 1) {
      el('circle', { cx: x + T - 3, cy: y + 3, r: 2.6, fill: '#fff' }, svg);
    }
    const key = sp.entries.length > 1 ? sp.entries.map((e) => e.name).join(' / ') + '（随机）' : shown.name;
    tally.set(key, (tally.get(key) || 0) + 1);
  }
  box.appendChild(svg);
  box.appendChild(html('div', { class: 'caption' },
    `${lay.width}×${lay.height} 格。白点表示该处有多个候选，悬停查看概率。`));
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
$('t-text').addEventListener('change', () => {
  try { localStorage.setItem(TEXT_STYLE_KEY, $('t-text').checked ? '1' : '0'); } catch { /* private mode */ }
  if (state.data) renderFloor();
});
$('game').addEventListener('change', () => {
  applyGame();
  if ($('seed').value.trim()) generate(true);
});
document.addEventListener('click', (e) => {
  const menu = $('cellmenu');
  if (!menu.hidden && !menu.contains(e.target)) closeCellMenu();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeCellMenu();
  if (!state.data || ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowRight' && state.floor < state.data.floors.length - 1) { state.floor++; state.room = null; render(); }
  if (e.key === 'ArrowLeft' && state.floor > 0) { state.floor--; state.room = null; render(); }
});
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.data) render(); });

const initial = new URLSearchParams(location.search);
writeParams(initial);
$('t-text').checked = textStyle();
applyGame();
if (initial.get('seed')) generate(true);
