// Drawing the floor. Icon style is the game's own minimap: 9x8-pixel room tiles for each shape
// (gfx/ui/minimap1; the start room as the current room, rooms not found yet as unvisited) with the
// room icons on top, scaled by an integer k so the pixels stay crisp. Text style draws coloured rooms
// with labels. Both use the same geometry: a cell is 9k x 8k.
import { GRID, TYPE_KIND, neighbours } from './infer.js';
import { ORIGIN, TILE, compose, doorPlace, loadArt, prepare, shapeSize, shownEntry, unshown, walkable } from './roomart.js';

const SVGNS = 'http://www.w3.org/2000/svg';
export const ICONS = {
  2: 'IconShop', 4: 'IconTreasureRoom', 5: 'IconBoss', 6: 'IconMiniboss', 7: 'IconSecretRoom',
  8: 'IconSuperSecretRoom', 9: 'IconArcade', 10: 'IconCurseRoom', 11: 'IconAmbushRoom', 12: 'IconLibrary',
  13: 'IconSacrificeRoom', 14: 'IconDevilRoom', 15: 'IconAngelRoom', 18: 'IconIsaacsRoom', 19: 'IconBarrenRoom',
  20: 'IconChestRoom', 21: 'IconDiceRoom', 24: 'IconPlanetarium', 29: 'IconUltraSecretRoom',
};
const TYPE_COLOR = {
  1: '--room', 2: '--t-shop', 3: '--t-other', 4: '--t-treasure', 5: '--t-boss', 6: '--t-miniboss',
  7: '--t-secret', 8: '--t-supersecret', 9: '--t-arcade', 10: '--t-curse', 11: '--t-challenge',
  12: '--t-library', 13: '--t-sacrifice', 18: '--t-bedroom', 19: '--t-bedroom', 20: '--t-vault',
  21: '--t-dice', 24: '--t-planetarium', 29: '--t-ultrasecret',
};
const LABEL = { 7: '隐', 8: '超', 29: '究' };
const NARROW = { 2: 'h', 7: 'h', 3: 'v', 5: 'v' };      // the narrow room shapes: a corridor across or down
export const HEAT = [['secret', '隐', '--heat'], ['super', '超', '--heat-ss'], ['ultra', '究', '--heat-us']];

export function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
export function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
export function iconName(type, subtype) {
  return type === 11 && subtype === 1 ? 'IconBossAmbushRoom' : ICONS[type];
}
// one frame of a sprite sheet at scale k
export function sprite(parent, sheet, frame, x, y, k, attrs = {}) {
  const [fx, fy, fw, fh] = frame;
  const s = el('svg', { x, y, width: fw * k, height: fh * k, viewBox: `${fx} ${fy} ${fw} ${fh}`, ...attrs }, parent);
  el('image', { href: sheet.url, width: sheet.w, height: sheet.h, class: 'pixel' }, s);
  return s;
}
// a 9x8 tile (with an optional icon), e.g. for legends and the panel
export function tileIcon(sheet, type, k = 2, tile = 'RoomVisited') {
  const svg = el('svg', { width: 9 * k, height: 8 * k, viewBox: `0 0 ${9 * k} ${8 * k}`, 'aria-hidden': 'true' });
  sprite(svg, sheet.tiles, sheet.tiles.frames[tile][0], 0, 0, k);
  const icon = type > 0 ? sheet.icons.frames[iconName(type, 0)] : null;
  if (icon) sprite(svg, sheet.icons, icon, -2 * k, -2 * k, k);
  return svg;
}
export function iconOnly(sheet, name, k = 2) {
  const svg = el('svg', { width: 9 * k, height: 8 * k, viewBox: `0 0 ${9 * k} ${8 * k}`, 'aria-hidden': 'true' });
  const frame = sheet.icons.frames[name];
  if (frame) sprite(svg, sheet.icons, frame, -2 * k, -2 * k, k);
  return svg;
}

// cells [x0, y0, cols, rows] around the given cells, with a margin
export function boundsOf(cells, margin = 1) {
  let x0 = GRID, y0 = GRID, x1 = -1, y1 = -1;
  for (const c of cells) {
    const x = c % GRID, y = Math.floor(c / GRID);
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  x0 = Math.max(0, x0 - margin); y0 = Math.max(0, y0 - margin);
  x1 = Math.min(GRID - 1, x1 + margin); y1 = Math.min(GRID - 1, y1 + margin);
  return { x0, y0, cols: x1 - x0 + 1, rows: y1 - y0 + 1 };
}

// Which rooms to draw and how: map view shows everything (rooms to be found as unvisited tiles);
// find view shows the explored rooms, the hidden rooms found, red rooms, and after "reveal" the rest.
function roomsToDraw(floor, v) {
  const out = [];
  for (const r of floor.rooms) {
    if (!r.hidden || v.view === 'map') {
      out.push({ room: r, tile: r.start ? 'RoomCurrent' : r.hidden ? 'RoomUnvisited' : 'RoomVisited', dashed: r.hidden });
    } else if (v.obs && v.obs.get(r.cells[0]) === TYPE_KIND[r.type]) {
      out.push({ room: r, tile: 'RoomVisited', found: true });
    } else if (v.revealed) {
      out.push({ room: r, tile: 'RoomUnvisited', dashed: true, missed: true });
    }
  }
  return out;
}

export function drawMap(svg, floor, v) {
  const k = v.k;
  const Sx = 9 * k, Sy = 8 * k;
  const icons = v.style === 'icons' && v.sheet;
  const sheet = v.sheet;
  svg.replaceChildren();
  const b = v.bounds;
  svg.setAttribute('viewBox', `0 0 ${b.cols * Sx} ${b.rows * Sy}`);
  svg.setAttribute('width', b.cols * Sx);
  svg.setAttribute('height', b.rows * Sy);
  const pos = (c) => [(c % GRID - b.x0) * Sx, (Math.floor(c / GRID) - b.y0) * Sy];
  const inside = (c) => {
    const x = c % GRID - b.x0, y = Math.floor(c / GRID) - b.y0;
    return x >= 0 && y >= 0 && x < b.cols && y < b.rows;
  };
  const gap = icons ? 0 : Math.max(1, Math.round(k * 0.8));
  const cellRect = (c, parent, attrs, inset = gap / 2) => {
    const [x, y] = pos(c);
    return el('rect', { x: x + inset, y: y + inset, width: Sx - 2 * inset, height: Sy - 2 * inset, rx: k, ...attrs }, parent);
  };
  const centre = (r) => {
    let cx = 0, cy = 0;
    for (const c of r.cells) { const [x, y] = pos(c); cx += x + Sx / 2; cy += y + Sy / 2; }
    return [cx / r.cells.length, cy / r.cells.length];
  };
  const drawn = roomsToDraw(floor, v);
  const occupied = new Map();
  for (const d of drawn) for (const c of d.room.cells) occupied.set(c, d.room);
  const obs = v.obs || new Map();

  // the floor grid
  const gGrid = el('g', {}, svg);
  if (v.grid !== false) {
    for (let yy = 0; yy < b.rows; yy++) {
      for (let xx = 0; xx < b.cols; xx++) {
        cellRect((yy + b.y0) * GRID + xx + b.x0, gGrid, { fill: css('--cell-empty') }, Math.max(1, k * 0.6));
      }
    }
  }

  // probabilities of the hidden rooms (find view)
  if (v.heat) {
    const gh = el('g', {}, svg);
    const fs = Math.max(8, Math.round(k * 1.75));
    const cells = new Set(HEAT.flatMap(([key]) => [...v.heat[key].keys()]));
    for (const c of cells) {
      if (occupied.has(c) || obs.has(c) || !inside(c)) continue;
      const lines = HEAT.map(([key, tag, col]) => [tag, v.heat[key].get(c) || 0, col]).filter(([, p]) => p >= 0.01);
      if (!lines.length) continue;
      const main = lines.reduce((a, q) => (q[1] > a[1] ? q : a));
      cellRect(c, gh, { fill: css(main[2]), 'fill-opacity': 0.1 + 0.55 * Math.min(1, main[1] / 0.5) }, Math.max(1, k * 0.6));
      const [x, y] = pos(c);
      lines.forEach(([tag, p], i) => {
        const t = el('text', { x: x + Sx / 2, y: y + Sy / 2 + (i - (lines.length - 1) / 2) * fs * 1.15 + fs * 0.36,
          'text-anchor': 'middle', 'font-size': fs, fill: css('--label'), 'font-weight': 700 }, gh);
        el('tspan', {}, t).textContent = tag;
        el('tspan', { class: 'num' }, t).textContent = `${Math.round(p * 100)}%`;
      });
    }
  }

  const gRooms = el('g', {}, svg);
  // doors as bridges (text style)
  if (!icons) {
    const gDoors = el('g', {}, gRooms);
    for (const d of floor.doors) {
      if (!occupied.has(d.a) || !occupied.has(d.b)) continue;
      const [xa, ya] = pos(d.a), [xb, yb] = pos(d.b);
      if (ya === yb) {
        const w = Sy * 0.28;
        el('rect', { x: Math.min(xa, xb) + Sx - gap, y: ya + Sy / 2 - w / 2, width: gap * 2, height: w, fill: css('--room-edge') }, gDoors);
      } else {
        const w = Sx * 0.28;
        el('rect', { x: xa + Sx / 2 - w / 2, y: Math.min(ya, yb) + Sy - gap, width: w, height: gap * 2, fill: css('--room-edge') }, gDoors);
      }
    }
  }
  for (const d of drawn) {
    const r = d.room;
    const g = el('g', { class: v.onRoom ? 'room' : '', 'data-index': r.index }, gRooms);
    if (d.found) g.setAttribute('data-found', r.cells[0]);
    const narrow = NARROW[r.shape];
    if (icons) {
      // the game's own tile for the shape (narrow rooms are thin), then the icon: half size on a narrow
      // room so the thin tile still shows around it. Neither takes the pointer: the icon's frame is
      // bigger than the room, so the room's own cells below answer instead.
      const [x, y] = pos(r.y * GRID + r.x);
      sprite(g, sheet.tiles, sheet.tiles.frames[d.tile][r.shape - 1], x, y, k, { class: 'cell', 'pointer-events': 'none' });
      const icon = sheet.icons.frames[iconName(r.type, r.subtype)];
      const [cx, cy] = centre(r);
      const ki = narrow ? Math.max(1, Math.round(k / 2)) : k;
      if (icon) sprite(g, sheet.icons, icon, cx - 6.5 * ki, cy - 6 * ki, ki, { 'pointer-events': 'none', ...(d.missed ? { opacity: 0.75 } : {}) });
    } else if (narrow) {
      // a narrow room as a corridor through its cells
      const fill = r.start ? css('--t-start') : css(TYPE_COLOR[r.type] || '--t-other');
      const xs = r.cells.map((c) => pos(c)[0]), ys = r.cells.map((c) => pos(c)[1]);
      const x0 = Math.min(...xs), y0 = Math.min(...ys), x1 = Math.max(...xs) + Sx, y1 = Math.max(...ys) + Sy;
      const t = narrow === 'h' ? Sy * 0.42 : Sx * 0.42;
      el('rect', narrow === 'h'
        ? { x: x0 + gap / 2, y: (y0 + y1) / 2 - t / 2, width: x1 - x0 - gap, height: t, rx: k * 0.8, fill, class: 'cell' }
        : { x: (x0 + x1) / 2 - t / 2, y: y0 + gap / 2, width: t, height: y1 - y0 - gap, rx: k * 0.8, fill, class: 'cell' }, g);
    } else {
      const fill = r.start ? css('--t-start') : css(TYPE_COLOR[r.type] || '--t-other');
      const set = new Set(r.cells);
      for (const c of r.cells) {
        cellRect(c, g, { fill, class: 'cell', ...(d.missed ? { 'fill-opacity': 0.55 } : {}) });
        const [x, y] = pos(c);
        if (c % GRID < GRID - 1 && set.has(c + 1)) el('rect', { x: x + Sx - gap / 2 - 1, y: y + gap / 2, width: gap + 2, height: Sy - gap, fill }, g);
        if (set.has(c + GRID)) el('rect', { x: x + gap / 2, y: y + Sy - gap / 2 - 1, width: Sx - gap, height: gap + 2, fill }, g);
      }
      const label = r.start ? '起' : r.label;
      if (label && v.labels !== false) {
        const [cx, cy] = centre(r);
        el('text', { x: cx, y: cy + k * 1.2, 'text-anchor': 'middle', 'font-size': k * 3.3, 'font-weight': 700,
          fill: css('--label') }, g).textContent = label;
      }
    }
    if (d.dashed) {
      for (const c of r.cells) {
        cellRect(c, g, { fill: 'none', stroke: d.missed ? css('--coin') : css('--label'), 'stroke-width': Math.max(1, k / 2),
          'stroke-dasharray': `${k * 1.5} ${k}`, 'stroke-opacity': d.missed ? 0.9 : 0.35 }, k * 0.5);
      }
    }
    if (v.depth && r.depth >= 0) {
      const [cx, cy] = centre(r);
      el('text', { x: cx, y: cy + k * 1.2, 'text-anchor': 'middle', 'font-size': k * 3, 'font-weight': 700, fill: css('--label'),
        class: 'outlined num', 'stroke-width': k * 0.7 }, g).textContent = r.depth;
    }
    for (const c of r.cells) {                  // the room answers the pointer exactly over its cells
      const [x, y] = pos(c);
      el('rect', { x, y, width: Sx, height: Sy, fill: 'transparent', class: 'roomhit' }, g);
    }
    if (v.onRoom) g.addEventListener('click', () => v.onRoom(r));
    if (v.onHover) {
      g.addEventListener('pointermove', (e) => v.onHover({ room: r }, e));
      g.addEventListener('pointerleave', () => v.onLeave && v.onLeave());
    }
  }

  // what the player learned (find view): empty walls and red rooms
  const gMarks = el('g', {}, svg);
  for (const [c, kind] of obs) {
    if (!inside(c)) continue;
    const [x, y] = pos(c);
    if (kind === 'empty') {
      const m = k * 2.2;
      const attrs = { stroke: css('--bone-3'), 'stroke-width': Math.max(1, k * 0.8), 'stroke-linecap': 'square' };
      el('line', { x1: x + Sx / 2 - m, y1: y + Sy / 2 - m, x2: x + Sx / 2 + m, y2: y + Sy / 2 + m, ...attrs }, gMarks);
      el('line', { x1: x + Sx / 2 + m, y1: y + Sy / 2 - m, x2: x + Sx / 2 - m, y2: y + Sy / 2 + m, ...attrs }, gMarks);
    } else if (kind === 'red') {
      const g = el('g', { 'data-red': c }, gMarks);
      if (icons) sprite(g, sheet.tiles, sheet.tiles.frames.RoomVisited[0], x, y, k);
      else cellRect(c, g, { fill: css('--room') });
      cellRect(c, g, { fill: css('--blood'), 'fill-opacity': 0.45 }, icons ? k : gap / 2);
      if (!icons) el('text', { x: x + Sx / 2, y: y + Sy / 2 + k * 1.2, 'text-anchor': 'middle', 'font-size': k * 3.3,
        'font-weight': 700, fill: css('--label') }, g).textContent = '红';
    } else if (!icons && !occupied.has(c)) {
      cellRect(c, gMarks, { fill: css(TYPE_COLOR[{ secret: 7, super: 8, ultra: 29 }[kind]]) });
      el('text', { x: x + Sx / 2, y: y + Sy / 2 + k * 1.2, 'text-anchor': 'middle', 'font-size': k * 3.3, 'font-weight': 700,
        fill: css('--label') }, gMarks).textContent = LABEL[{ secret: 7, super: 8, ultra: 29 }[kind]];
    }
  }

  // the order of the AI's moves (its replay)
  if (v.stepLabels) {
    const fs = Math.max(9, Math.round(k * 2));
    for (const [c, label] of v.stepLabels) {
      if (!inside(c)) continue;
      const [x, y] = pos(c);
      el('text', { x: x + k * 0.8, y: y + fs * 0.95, 'font-size': fs, fill: css('--coin'), class: 'outlined num step',
        'stroke-width': Math.max(2, k * 0.6) }, svg).textContent = label;
    }
  }

  // selection outline (map view)
  if (v.selected !== null && v.selected !== undefined) {
    const r = floor.rooms.find((q) => q.index === v.selected);
    if (r && occupied.get(r.cells[0]) === r) {
      const set = new Set(r.cells);
      const w = Math.max(2, k * 0.6);
      for (const c of r.cells) {
        const [x, y] = pos(c);
        for (const [n, x1, y1, x2, y2] of [[c - 1, x, y, x, y + Sy], [c - GRID, x, y, x + Sx, y],
          [c + 1, x + Sx, y, x + Sx, y + Sy], [c + GRID, x, y + Sy, x + Sx, y + Sy]]) {
          const edge = !set.has(n) || (n === c - 1 && c % GRID === 0) || (n === c + 1 && c % GRID === GRID - 1);
          if (edge) el('line', { x1, y1, x2, y2, stroke: css('--sel'), 'stroke-width': w, 'stroke-linecap': 'square' }, svg);
        }
      }
    }
  }

  // suggestions: the wall to bomb, the door to open with the Red Key
  const ring = (c, col, iconName, fallback) => {
    if (!inside(c)) return;
    const [x, y] = pos(c);
    el('rect', { x: x + k * 0.5, y: y + k * 0.5, width: Sx - k, height: Sy - k, rx: k, fill: 'none', stroke: col,
      'stroke-width': Math.max(2, k * 0.55), class: 'suggest' }, svg);
    if (iconName === 'IconKey' && v.redKey) {             // the Red Key's own sprite
      const w = v.redKey.w * Math.max(1, k / 4), h = v.redKey.h * Math.max(1, k / 4);
      el('image', { href: v.redKey.url, x: x - w / 2, y: y - h / 2, width: w, height: h, class: 'pixel suggest-icon' }, svg);
    } else if (icons && sheet.icons.frames[iconName]) {
      const k2 = Math.max(1, Math.round(k / 2));          // a half-size glyph on the ring's corner
      sprite(svg, sheet.icons, sheet.icons.frames[iconName], x - k - 2 * k2, y - k - 2 * k2, k2, { class: 'suggest-icon' });
    } else {
      el('circle', { cx: x + k * 1.5, cy: y + k * 1.5, r: k * 1.6, fill: col }, svg);
      el('text', { x: x + k * 1.5, y: y + k * 2.3, 'text-anchor': 'middle', 'font-size': k * 2, 'font-weight': 700,
        fill: '#140f0c' }, svg).textContent = fallback;
    }
  };
  if (v.suggest && v.suggest.bomb) ring(v.suggest.bomb[0], css('--coin'), 'IconBomb', '炸');
  if (v.suggest && v.suggest.key) ring(v.suggest.key[0], css('--heat-us'), 'IconKey', '钥');

  // every free cell in view answers the pointer (find view)
  if (v.onCell) {
    const gHit = el('g', {}, svg);
    for (let yy = 0; yy < b.rows; yy++) {
      for (let xx = 0; xx < b.cols; xx++) {
        const c = (yy + b.y0) * GRID + xx + b.x0;
        if (occupied.has(c) || (obs.has(c) && obs.get(c) !== 'empty')) continue;
        const can = v.actionable ? v.actionable(c) : false;
        const hit = el('rect', { x: xx * Sx, y: yy * Sy, width: Sx, height: Sy, fill: 'transparent',
          class: can ? 'hit can' : 'hit', 'data-cell': c }, gHit);
        hit.addEventListener('click', (e) => v.onCell(c, e));
        if (v.onHover) {
          hit.addEventListener('pointermove', (e) => v.onHover({ cell: c }, e));
          hit.addEventListener('pointerleave', () => v.onLeave && v.onLeave());
        }
      }
    }
  }
  return { pos, Sx, Sy };
}

// A bomb going off at a cell, and the room it opened popping in. Skipped with reduced motion.
export function playEffect(svg, geo, cell, found, k) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const [x, y] = geo.pos(cell);
  const cx = x + geo.Sx / 2, cy = y + geo.Sy / 2;
  const g = el('g', { 'pointer-events': 'none' }, svg);
  const flash = el('circle', { cx, cy, r: k * 5, fill: '#fff6d8' }, g);
  flash.style.transformOrigin = `${cx}px ${cy}px`;
  flash.animate([{ opacity: 0.95, transform: 'scale(0.3)' }, { opacity: 0, transform: 'scale(1.4)' }],
    { duration: 260, easing: 'ease-out', fill: 'forwards' });
  const colours = ['#fff2b0', '#ffb23f', '#e8562c', '#6b5a4a', '#2b221c'];
  for (let i = 0; i < 14; i++) {
    const a = (i / 14) * Math.PI * 2 + (i % 3) * 0.2;
    const dist = k * (5 + (i % 4) * 2.2);
    const size = k * (i % 3 === 0 ? 1.4 : 1);
    const p = el('rect', { x: cx - size / 2, y: cy - size / 2, width: size, height: size, fill: colours[i % colours.length] }, g);
    p.animate([{ transform: 'translate(0,0)', opacity: 1 },
      { transform: `translate(${Math.cos(a) * dist}px, ${Math.sin(a) * dist}px)`, opacity: 0 }],
    { duration: 420 + (i % 5) * 40, easing: 'cubic-bezier(.2,.7,.3,1)', fill: 'forwards' });
  }
  setTimeout(() => g.remove(), 700);
  if (found) {
    const room = svg.querySelector(`[data-found="${cell}"], [data-red="${cell}"]`);
    if (room) {
      room.style.transformOrigin = `${cx}px ${cy}px`;
      room.animate([{ transform: 'scale(0.4)', opacity: 0 }, { transform: 'scale(1.12)', opacity: 1, offset: 0.7 },
        { transform: 'scale(1)', opacity: 1 }], { duration: 380, delay: 120, easing: 'ease-out', fill: 'backwards' });
    }
  }
}

// a floor as a small minimap at the game's own pixel size (k = 1): everything in the map view, what
// the player has explored and found in the find view
export function drawThumb(floor, sheet, style, view, obs, revealed) {
  const svg = el('svg', { 'aria-hidden': 'true' });
  const shown = floor.rooms.filter((r) => view === 'map' || !r.hidden || revealed
    || (obs && obs.get(r.cells[0]) === TYPE_KIND[r.type]));
  drawMap(svg, floor, { k: 1, style, sheet, view, bounds: boundsOf(shown.flatMap((r) => r.cells), 0), grid: false,
    labels: false, obs, revealed });
  return svg;
}

const KIND = {
  rock: ['#8a8175', '石头类'], poop: ['#7a5230', '大便'], tnt: ['#b0412e', '炸药桶'], block: ['#9aa3ad', '方块'],
  pit: ['#050404', '沟壑'], spikes: ['#7d2020', '地刺'], web: ['#cfcfcf', '蛛网'], plate: ['#3d6fb0', '按钮'],
  door: ['#3d6fb0', '活板门/暗门'], deco: ['#5a5246', '装饰'], grid: ['#777777', '网格物体'],
  pickup: ['#f2c33b', '拾取物'], enemy: ['#e2553f', '敌人'], object: ['#3fb0a8', '机器/火堆等'],
};

// the text for a spawn point: each candidate with its chance
function spawnTitle(sp) {
  const total = sp.entries.reduce((a, e) => a + e.weight, 0) || 1;
  return sp.entries.map((e) => `${e.name}${sp.entries.length > 1 ? ' ' + Math.round(100 * e.weight / total) + '%' : ''}`).join('\n');
}
function countSpawn(tally, sp) {
  if (!tally) return;
  const key = sp.entries.length > 1 ? sp.entries.map((e) => e.name).join(' / ') + '（随机）' : shownEntry(sp).name;
  tally.set(key, (tally.get(key) || 0) + 1);
}
// what a door slot shows (doorState(bit)): 'open', a door into a room the player knows; 'key', a slot the
// Red Key can open (only while the Red Key is the tool); 'wall', nothing, as in the game
const DOOR_TITLE = { open: '门：通向你知道的房间', key: '红钥匙可以在这里开门' };
const RED_DOOR = 'M -12 13 V -6 A 12 12 0 0 1 12 -6 V 13';     // an arched doorway, open side into the room

// A room layout as a plain diagram (tiles 16 units), for when the game art cannot be loaded: walls,
// doors, obstacles and spawns.
export function layoutSvg(lay, doorState, tally = null) {
  const T = 16, W = lay.width, H = lay.height;
  const svg = el('svg', { viewBox: `0 0 ${(W + 2) * T} ${(H + 2) * T}`, role: 'img', 'aria-label': '房间布局' });
  el('rect', { x: 0, y: 0, width: (W + 2) * T, height: (H + 2) * T, fill: css('--tile-wall') }, svg);
  el('rect', { x: T, y: T, width: W * T, height: H * T, fill: css('--tile-floor') }, svg);
  for (const [mx, my, mw, mh] of lay.missing) {
    el('rect', { x: (mx + 1) * T, y: (my + 1) * T, width: mw * T, height: mh * T, fill: css('--tile-wall') }, svg);
  }
  for (const [dx, dy, bit] of lay.door_list) {
    const st = doorState(bit);
    if (st === 'wall') continue;
    const d = el('rect', { x: (dx + 1) * T + 1, y: (dy + 1) * T + 1, width: T - 2, height: T - 2, rx: 2,
      fill: st === 'open' ? '#4fbf6a' : 'rgba(201, 53, 44, .3)', stroke: st === 'open' ? 'none' : '#ff4b3e',
      'stroke-width': 1.5, 'stroke-dasharray': st === 'open' ? 'none' : '3 2' }, svg);
    el('title', {}, d).textContent = DOOR_TITLE[st];
  }
  for (const sp of lay.spawns) {
    spawnMarker(svg, sp, (sp.x + 1) * T, (sp.y + 1) * T, T);
    countSpawn(tally, sp);
  }
  return svg;
}

// a coloured marker for a spawn point (the diagram, or what the art has no sprite for)
function spawnMarker(svg, sp, x, y, T) {
  const shown = shownEntry(sp);
  const [col] = KIND[shown.kind] || ['#999'];
  let node;
  if (['pickup', 'enemy', 'object'].includes(shown.kind)) {
    node = el('circle', { cx: x + T / 2, cy: y + T / 2, r: T * 0.36, fill: col, stroke: '#000', 'stroke-opacity': 0.4 }, svg);
  } else if (shown.kind === 'none') {
    node = el('circle', { cx: x + T / 2, cy: y + T / 2, r: T * 0.12, fill: '#999' }, svg);
  } else {
    node = el('rect', { x: x + 1, y: y + 1, width: T - 2, height: T - 2, rx: 2, fill: col,
      stroke: shown.kind === 'pit' ? '#6b6356' : 'none' }, svg);
  }
  el('title', {}, node).textContent = spawnTitle(sp);
  if (sp.entries.length > 1) el('circle', { cx: x + T - 3, cy: y + 3, r: 2.6, fill: '#fff' }, svg);
}

// The layout in the game's art (roomart.js): the composed room, and on top of it the doors the Red
// Key could open (red dashed doorways), markers for spawns the art has no sprite for, and a tooltip on
// every spawn point.
export function layoutArt(art, lay, room, floor, doorState, tally = null) {
  const canvas = compose(art, lay, room, floor, doorState);
  const wrap = document.createElement('div');
  wrap.className = 'roomart';
  wrap.appendChild(canvas);
  const T = TILE, O = ORIGIN;
  const svg = el('svg', { viewBox: `0 0 ${canvas.width} ${canvas.height}`, role: 'img', 'aria-label': '房间布局（原版贴图）' }, wrap);
  const walk = walkable(lay.shape, lay.width, lay.height);
  for (const [dx, dy, bit] of lay.door_list) {
    const st = doorState(bit);
    if (st === 'wall') continue;
    const p = doorPlace([dx, dy], walk);
    const g = el('g', { transform: `translate(${O + (p.x + 0.5) * T} ${O + (p.y + 0.5) * T}) rotate(${Math.round(p.angle * 180 / Math.PI)})` }, svg);
    if (st === 'key') el('path', { d: RED_DOOR, class: 'reddoor' }, g);
    else el('rect', { x: -13, y: -13, width: 26, height: 26, fill: 'transparent' }, g);
    el('title', {}, g).textContent = DOOR_TITLE[st];
  }
  const missing = new Set(unshown(art, lay, room, floor));
  for (const sp of lay.spawns) {
    const x = O + sp.x * T, y = O + sp.y * T;
    if (missing.has(sp)) spawnMarker(svg, sp, x, y, T);
    else {
      const hit = el('rect', { x, y, width: T, height: T, fill: 'transparent' }, svg);
      el('title', {}, hit).textContent = spawnTitle(sp);
      if (sp.entries.length > 1) el('circle', { cx: x + T - 4, cy: y + 4, r: 3.2, fill: '#fff', stroke: '#000', 'stroke-opacity': 0.5 }, svg);
    }
    countSpawn(tally, sp);
  }
  return wrap;
}

// the room art in a dialog, scaled by a whole number so the pixels stay square
function openLarge(build, title) {
  let dlg = document.getElementById('roomdlg');
  if (!dlg) {
    dlg = document.createElement('dialog');
    dlg.id = 'roomdlg';
    dlg.className = 'roomdlg';
    dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
    document.body.appendChild(dlg);
  }
  const view = build();
  const canvas = view.querySelector('canvas');
  const room = Math.min(window.innerWidth * 0.92, 1400) - 24;
  view.style.width = `${canvas.width * Math.max(1, Math.floor(room / canvas.width))}px`;
  const head = document.createElement('div');
  head.className = 'dlghead';
  const h = document.createElement('b');
  h.textContent = title;
  const close = document.createElement('button');
  close.type = 'button';
  close.textContent = '关闭';
  close.addEventListener('click', () => dlg.close());
  head.append(h, close);
  dlg.replaceChildren(head, view);
  dlg.showModal();
  close.focus();
}

// the layout with its caption and a count of what spawns in it: in the game's art when it can be
// loaded (room and floor given), else as the diagram
// a placeholder of the room's size while its layout and art load (instead of a different drawing)
export function roomSkeleton(shape) {
  const [w, h] = shapeSize(shape);
  const box = document.createElement('div');
  box.className = 'roomart skeleton';
  box.style.aspectRatio = `${w + 4} / ${h + 4}`;
  box.setAttribute('aria-label', '加载中');
  return box;
}

// the layout with its caption and a count of what spawns in it: in the game's art (room and floor
// given, art.json loaded), else the diagram
export async function drawLayout(box, lay, doorState, where = null) {
  const tally = new Map();
  let view = null;
  const { room, floor, borrowed } = where || {};
  const art = room && floor ? await loadArt() : null;
  if (art) {
    const prep = prepare(art, lay, room, floor);
    if (!prep.ready) {
      if (!box.querySelector('.skeleton')) box.replaceChildren(roomSkeleton(lay.shape));
      await prep.job;
    }
    if (!box.isConnected) return;
    try {
      view = layoutArt(art, lay, room, floor, doorState, tally);
    } catch (err) {
      console.warn('room art failed', err);
      tally.clear();
    }
  }
  const isArt = !!view;
  if (!view) view = layoutSvg(lay, doorState, tally);
  if (isArt) {
    view.tabIndex = 0;
    view.setAttribute('role', 'button');
    view.title = '点击放大';
    const open = () => openLarge(() => layoutArt(art, lay, room, floor, doorState), where.title || '房间布局');
    view.addEventListener('click', open);
    view.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  }
  const keys = lay.door_list.some((d) => doorState(d[2]) === 'key');
  const cap = document.createElement('div');
  cap.className = 'caption';
  cap.textContent = isArt
    ? `${lay.width}×${lay.height} 格，原版贴图${borrowed ? '（借用忏悔+ 的美术）' : ''}，点击放大。画出的门通向你知道的房间${keys ? '；红色虚线门是红钥匙能打开的门位' : ''}。白点：多个候选，悬停查看。地板和石头的样式按规则随机，细节可能与游戏不同。`
    : `${lay.width}×${lay.height} 格。绿色是门${keys ? '，红色虚线是红钥匙能打开的门位' : ''}。白点表示该处有多个候选，悬停查看概率。`;
  const ul = document.createElement('ul');
  ul.className = 'spawnlist';
  for (const [name, n] of [...tally].sort((a, q) => q[1] - a[1])) {
    const li = document.createElement('li');
    li.textContent = `${name} ×${n}`;
    ul.appendChild(li);
  }
  box.replaceChildren(view, cap, ul);
}

export { neighbours };
