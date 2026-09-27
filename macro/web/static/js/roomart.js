// A room layout in the game's own art (the Repentance+ files, also for AB+; see web/roomart.py).
// The backdrop is composed from the stage's sheet the way Backdrop::pre_render_floor and
// pre_render_walls do it (AB+ 0x1150A0 / 0x10A2C0): 7x4-tile floor pieces from a shuffled list,
// mirrored towards the right and bottom; walls from 52-px corners and 26-px segments whose sheet
// variant is picked at random. Rocks pick their shape like GridEntity_Rock::InitSubclass (decoration
// seed + grid index, shift 35), pits their tile like GridEntity_Pit::PostInit (neighbours).
// The random picks use the room's decoration seed but are not checked against the game, so the
// details can differ from what the game shows; the layout itself is exact.

const T = 26;                 // pixels per grid tile
const O = 2 * T;              // the interior's top-left corner: 2 tiles of wall around it
const BLOCK_W = 234, BLOCK_H = 156;

let artJob = null;
const images = new Map();

export function loadArt() {
  if (!artJob) {
    artJob = fetch('art/art.json').then((r) => (r.ok ? r.json() : null)).catch(() => null);
  }
  return artJob;
}

function image(url) {
  if (!images.has(url)) {
    const img = new Image();
    const job = new Promise((resolve) => {
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
    });
    img.src = 'art/' + url;
    images.set(url, { img, job });
  }
  return images.get(url);
}
const loaded = (url) => { const e = images.get(url); return !!(e && e.img.complete && e.img.naturalWidth); };

// RNG::Next with the shift triple (5, 9, 7) (s_Shifts[35]); Random(n) advances first
function rng(seed) {
  let s = seed >>> 0;
  return {
    int(n) {
      s ^= s >>> 5; s ^= s << 9; s ^= s >>> 7; s >>>= 0;
      return n ? s % n : 0;
    },
  };
}

// walkable interior tiles of each room shape (narrow rooms keep the 1x1 or 2x1 grid)
export function walkable(shape, W, H) {
  return (x, y) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return false;
    switch (shape) {
      case 2: case 7: return y >= 2 && y <= 4;
      case 3: case 5: return x >= 4 && x <= 8;
      case 9: return x >= 13 || y >= 7;
      case 10: return x < 13 || y >= 7;
      case 11: return x >= 13 || y < 7;
      case 12: return x < 13 || y < 7;
      default: return true;
    }
  };
}

function blit(ctx, img, sx, sy, sw, sh, dx, dy, dw, dh, fx = false, fy = false) {
  if (!fx && !fy) { ctx.drawImage(img, sx, sy, sw, sh, dx, dy, dw, dh); return; }
  ctx.save();
  ctx.translate(dx + (fx ? dw : 0), dy + (fy ? dh : 0));
  ctx.scale(fx ? -1 : 1, fy ? -1 : 1);
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, dw, dh);
  ctx.restore();
}

// ------------------------------------------------------------------------------------------ floor
function drawFloor(ctx, bd, bdId, sheet, lfloor, W, H, shape, R) {
  const piece = (img, sx, sy, sw, sh, tx, ty, tw, th, fx, fy) => blit(ctx, img, sx, sy, sw, sh, O + tx * T, O + ty * T, tw * T, th * T, fx, fy);
  if (shape >= 9 && lfloor) {       // three 13x7 chunks of the L floor sheet, mirrored per shape
    const chunk = [0, 1, 2].map((i) => [i * 338, R.int(2) * 182]);
    const place = {
      11: [[0, 0], [13, 0], [13, 7], false, false], 9: [[0, 7], [13, 7], [13, 0], false, true],
      10: [[13, 7], [0, 7], [0, 0], true, true], 12: [[13, 0], [0, 0], [0, 7], true, false],
    }[shape];
    for (let i = 0; i < 3; i++) piece(lfloor, chunk[i][0], chunk[i][1], 338, 182, place[i][0], place[i][1], 13, 7, place[3], place[4]);
    return;
  }
  const gw = W + 2, gh = H + 2;
  if (bdId === 15 && [4, 6, 8].includes(shape)) {      // Cathedral: 2x2 tiles and four corners
    const quads = [[286, 364], [338, 364], [338, 416], [286, 416]];
    for (let y = 0; y < H * T - 0.001; y += 52) {
      for (let x = 0; x < W * T - 0.001; x += 52) {
        const [sx, sy] = quads[R.int(4)];
        blit(ctx, sheet, sx, sy, 52, 52, O + x, O + y, 52, 52);
      }
    }
    for (const [tx, ty, fx, fy] of [[0, 0, false, false], [gw - 3, 0, true, false], [0, gh - 3, false, true], [gw - 3, gh - 3, true, true]]) {
      piece(sheet, 390, 364, 26, 26, tx, ty, 1, 1, fx, fy);
    }
    return;
  }
  // the general case: a shuffled list of the set's floor variants (each a 7x4 tile piece)
  const fv = bd.floorvariants;
  const set = R.int(bd.floors);
  let list = [];
  for (let v = set * fv; v < set * fv + fv; v++) list.push(v);
  while (list.length < 32) list = list.concat(list);
  for (let i = list.length - 1; i > 0; i--) {
    const j = R.int(i + 1);
    [list[i], list[j]] = [list[j], list[i]];
  }
  const random = bdId === 16 || bdId === 18;   // Dark Room, Mega Satan: pieces picked at random
  let k = -1;
  const next = () => (k = random ? R.int(4) : k + 1);
  const draw = (v, tx, ty, fx, fy) => {
    const sx = (v % 2) * BLOCK_W + 52, sy = Math.floor(v / 2) * BLOCK_H + 52;
    piece(sheet, sx, sy, 182, 104, tx, ty, 7, 4, fx, fy);
  };
  const right = gw - 9, bottom = gh - (random ? 5 : 6);
  const skip = (x, y) => (shape === 9 && y <= 4 && x <= 7) || (shape === 10 && y <= 4 && x >= 8)
    || (shape === 11 && y >= 5 && x <= 7) || (shape === 12 && y >= 5 && x >= 8);
  if (bottom > 0) {
    for (let y = 0; ;) {
      if (right > 0) {
        for (let x = 0; ;) {
          if (!skip(x, y)) draw(list[next()], x, y, x >= 1, y >= 1);
          if (right <= x + 6) break;
          x += 6;
          if (x === 6 && right !== 7) x = 7;
        }
      }
      y += 3;
      if (bottom <= y) break;
      if (y === 3 && bottom !== 4) y = 4;
    }
    for (let y = 0; y < bottom; y += 3) {                     // the right column
      if (y === 3 && bottom > 4) y = 4;
      if ((y === 3 || y < 5 || shape !== 12) && (y > 4 || shape !== 10)) draw(list[next()], right, y, true, y !== 0);
    }
  }
  if (right > 0) {                                            // the bottom row
    for (let x = 0; x < right; x += 6) {
      if (x === 6 && right > 7) x = 7;
      if ((x < 8 && shape !== 11) || (x >= 8 && shape !== 12)) {
        k = random ? R.int(2) + 4 : k + 1;
        draw(list[k], x, bottom, x !== 0, true);
      }
    }
  }
  if (shape !== 12) draw(list[Math.max(0, k)], right, bottom, true, true);   // the last piece again
}

// ------------------------------------------------------------------------------------------ walls
function drawWalls(ctx, bd, sheet, W, H, walk, R) {
  if (!bd.walls) return;
  const set = R.int(bd.walls), wv = bd.wallvariants;
  const choose = () => (R.int(3) === 0 ? set * wv + R.int(wv) : set * wv);
  const block = (i) => [(i % 2) * BLOCK_W, Math.floor(i / 2) * BLOCK_H];
  const corners = [choose(), choose(), choose(), choose()];      // top-left, top-right, bottom-left, bottom-right
  const cols = [], rows = [];
  for (let c = 0; c < W; c++) {
    let t = -1, b = -1;
    for (let y = 0; y < H; y++) if (walk(c, y)) { if (t < 0) t = y; b = y; }
    if (t >= 0) cols.push([c, t, b, choose(), choose()]);
  }
  for (let r = 0; r < H; r++) {
    let l = -1, rt = -1;
    for (let x = 0; x < W; x++) if (walk(x, r)) { if (l < 0) l = x; rt = x; }
    if (l >= 0) rows.push([r, l, rt, choose(), choose()]);
  }
  const at = (tx, ty) => [O + tx * T, O + ty * T];
  for (let y = 0; y < H; y++) {                                  // outer corners
    for (let x = 0; x < W; x++) {
      if (!walk(x, y)) continue;
      const l = !walk(x - 1, y), r = !walk(x + 1, y), u = !walk(x, y - 1), d = !walk(x, y + 1);
      for (const [on, i, tx, ty, fx, fy] of [[l && u, 0, x - 2, y - 2, false, false], [r && u, 1, x + 1, y - 2, true, false],
        [l && d, 2, x - 2, y + 1, false, true], [r && d, 3, x + 1, y + 1, true, true]]) {
        if (!on) continue;
        const [bx, by] = block(corners[i]);
        blit(ctx, sheet, bx, by, 52, 52, ...at(tx, ty), 52, 52, fx, fy);
      }
    }
  }
  // along a wall, the first 7 tiles copy the sheet's, the last 6 mirror it, the middle alternates
  const along = (i, n, first, last, mid) => (i < first ? [i + 2, false] : i >= n - last ? [2 + (n - 1 - i), true] : [mid + (i % 2), false]);
  for (const [c, t, b, top, bottom] of cols) {
    const [sc, fx] = along(c, W, 7, 6, 7);
    let [bx, by] = block(top);
    blit(ctx, sheet, bx + sc * T, by, T, 52, ...at(c, t - 2), T, 52, fx, false);
    [bx, by] = block(bottom);
    blit(ctx, sheet, bx + sc * T, by, T, 52, ...at(c, b + 1), T, 52, fx, true);
  }
  for (const [r, l, rt, left, right] of rows) {
    const [sr, fy] = along(r, H, 4, 3, 4);
    let [bx, by] = block(left);
    blit(ctx, sheet, bx, by + sr * T, 52, T, ...at(l - 2, r), 52, T, false, fy);
    [bx, by] = block(right);
    blit(ctx, sheet, bx, by + sr * T, 52, T, ...at(rt + 1, r), 52, T, true, fy);
  }
  for (let y = 0; y < H; y++) {                                  // inner corners of L rooms
    for (let x = 0; x < W; x++) {
      if (walk(x, y)) continue;
      const dx = walk(x + 1, y) ? 1 : walk(x - 1, y) ? -1 : 0;
      const dy = walk(x, y + 1) ? 1 : walk(x, y - 1) ? -1 : 0;
      if (!dx || !dy) continue;
      blit(ctx, sheet, 468, set * 52, 52, 52, ...at(dx > 0 ? x - 1 : x, dy > 0 ? y - 1 : y), 52, 52, dx < 0, dy < 0);
    }
  }
}

// -------------------------------------------------------------------------------- what spawns where
const POOP = { 1490: 'red', 1494: 'rainbow', 1495: 'corn', 1496: 'gold', 1497: 'black', 1498: 'white', 1500: 'normal', 1501: 'charming' };
const ROCK = { 1000: 'normal', 1009: 'normal', 1001: 'bombrock', 1002: 'alt', 1003: 'tinted', 1004: 'superspecial', 1008: 'alt2',
  1010: 'spiked', 1011: 'foolsgold', 1900: 'black', 1901: 'pillar' };
const SINGLE = { 1300: 'tnt', 1499: 'giantpoop', 1940: 'web', 4000: 'lock', 4500: 'plate', 6100: 'teleporter', 9000: 'trapdoor', 9100: 'crawlspace' };
const isPit = (e) => e && (e.type === 3000 || e.type === 3009);
const SILENT = new Set([1999, 3001, 3002, 10000]);   // invisible in the game: no marker either

// the entry a spawn point shows: its most likely one that is not empty
export function shownEntry(sp) {
  return [...sp.entries].sort((a, q) => q.weight - a.weight).find((e) => e.kind !== 'none') || sp.entries[0];
}

// sprite key for each spawn (null: nothing the art can show)
function spriteKeys(art, lay, room, bd, stage) {
  const W = lay.width;
  const grid = new Map();
  for (const sp of lay.spawns) grid.set(sp.y * W + sp.x, shownEntry(sp));
  const deco = room.seeds ? room.seeds.decoration : 0;
  const rocks = bd.rocks || 'rocks_basement.png';
  const pits = bd.pit || 'grid_pit.png';
  const out = [];
  for (const sp of lay.spawns) {
    const e = grid.get(sp.y * W + sp.x);
    const g = (sp.y + 1) * (W + 2) + sp.x + 1;                // the game's grid index (walls included)
    const R = rng(((deco + g) | 1) >>> 0);
    const pick = (prefix) => (art.groups[prefix] ? `${prefix}/${R.int(art.groups[prefix])}` : null);
    let key = null, floor = false;
    if (!e || e.type === 0) key = null;
    else if (e.type < 1000) key = `e/${e.type}.${e.variant}.${e.subtype}`;
    else if (ROCK[e.type]) key = pick(`rock/${rocks}/${ROCK[e.type]}`);
    else if (POOP[e.type]) key = pick(`poop/${POOP[e.type]}`);
    else if (SINGLE[e.type]) { key = SINGLE[e.type]; floor = ['plate', 'teleporter', 'trapdoor', 'crawlspace'].includes(key); }
    else if (e.type === 1400 || e.type === 1410) key = `e/33.${e.type === 1410 ? 1 : 0}.0`;
    else if (e.type === 5000 || e.type === 5001) key = `e/1000.${e.type === 5000 ? 6 : 9}.0`;   // devil / angel statue
    else if (e.type === 1930 || e.type === 1931) {
      key = `spikes/${stage === 7 || stage === 8 ? 'Womb' : ''}Spikes0${R.int(4) + 1}`;
      floor = true;
    } else if (isPit(e)) {
      const at = (dx, dy) => sp.x + dx >= 0 && sp.x + dx < W && isPit(grid.get((sp.y + dy) * W + sp.x + dx));
      let m = (at(-1, 0) ? 1 : 0) + (at(0, -1) ? 2 : 0) + (at(1, 0) ? 4 : 0) + (at(0, 1) ? 8 : 0);
      if (m === 0) m = R.int(2) << 4;
      key = `pit/${pits}/${pitFrame(m, at)}`;
      floor = true;
    }
    if (key && art.sprites[key]) out.push({ sp, e, key, floor });
    else out.push({ sp, e, key: null, silent: SILENT.has(e && e.type) });
  }
  return out;
}

// GridEntity_Pit::PostInit (AB+ 0x300C70): the tile from the four neighbours, corners from the diagonals
function pitFrame(m, at) {
  const ul = at(-1, -1), ur = at(1, -1), dl = at(-1, 1), dr = at(1, 1);
  switch (m) {
    case 3: return ul ? 3 : 17;
    case 6: return ur ? 6 : 18;
    case 7: return ul ? (ur ? 7 : 27) : (ur ? 28 : 29);
    case 9: return dl ? 9 : 19;
    case 11: return ul ? 11 : 26;
    case 12: return dr ? 12 : 20;
    case 13: return dl ? (dr ? 13 : 30) : (dr ? 31 : 32);
    case 14: return ur ? 14 : 25;
    case 15:
      if (!ur) return ul ? 22 : 23;
      if (!ul) return 21;
      return dr || dl ? 15 : 24;
    default: return m;
  }
}

// door slots: where the door sits and which way it faces (rotation of the door sprite)
export function doorPlace(d, walk) {
  const [x, y] = d;
  if (walk(x + 1, y)) return { x, y, angle: -Math.PI / 2 };
  if (walk(x - 1, y)) return { x, y, angle: Math.PI / 2 };
  if (walk(x, y + 1)) return { x, y, angle: 0 };
  return { x, y, angle: Math.PI };
}

// ------------------------------------------------------------------------------------------ public
export function backdropOf(art, room, floorBackdrop) {
  for (const id of [room.backdrop, floorBackdrop, 1]) if (art.backdrops[String(id)]) return [art.backdrops[String(id)], id];
  return [null, 0];
}

// the files a room needs; ready when all are loaded
export function prepare(art, lay, room, floor) {
  const [bd] = backdropOf(art, room, floor.backdrop);
  if (!bd) return { ready: false, job: Promise.resolve(false) };
  const keys = spriteKeys(art, lay, room, bd, floor.stage);
  const urls = new Set([bd.sheet]);
  if (lay.shape >= 9 && bd.lfloor) urls.add(bd.lfloor);
  for (const k of keys) if (k.key) urls.add(art.sprites[k.key][0]);
  const door = art.sprites[`door/${bd.door}`] || art.sprites['door/door_01_normaldoor.png'];
  if (door) urls.add(door[0]);
  const list = [...urls];
  return { ready: list.every(loaded), job: Promise.all(list.map((u) => image(u).job)).then(() => true) };
}

// the interior size of a room shape in tiles (for a placeholder of the right size before the layout loads)
export function shapeSize(shape) {
  return [shape >= 6 ? 26 : 13, [4, 5, 8, 9, 10, 11, 12].includes(shape) ? 14 : 7];
}

// composed rooms, by what they depend on (the same room is shown in the panel, the tip and the dialog)
const composed = new Map();
function copyOf(src) {
  const c = document.createElement('canvas');
  c.width = src.width;
  c.height = src.height;
  c.getContext('2d').drawImage(src, 0, 0);
  return c;
}

// the room drawn at the game's scale (26 px a tile) with walls around it; call after prepare() is ready
export function compose(art, lay, room, floor, doorState) {
  const open = lay.door_list.map((d) => (doorState(d[2]) === 'open' ? 1 : 0)).join('');
  const key = [room.backdrop, floor.backdrop, floor.stage, room.seeds && room.seeds.decoration, lay.stage, lay.type, lay.variant, open].join('.');
  if (!composed.has(key)) {
    composed.set(key, composeRoom(art, lay, room, floor, doorState));
    if (composed.size > 80) composed.delete(composed.keys().next().value);
  }
  return copyOf(composed.get(key));
}

function composeRoom(art, lay, room, floor, doorState) {
  const [bd, bdId] = backdropOf(art, room, floor.backdrop);
  const W = lay.width, H = lay.height;
  const canvas = document.createElement('canvas');
  canvas.width = (W + 4) * T;
  canvas.height = (H + 4) * T;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const walk = walkable(lay.shape, W, H);
  const R = rng(room.seeds ? room.seeds.decoration : 1);
  const sheet = image(bd.sheet).img;
  drawFloor(ctx, bd, bdId, sheet, lay.shape >= 9 && bd.lfloor ? image(bd.lfloor).img : null, W, H, lay.shape, R);
  ctx.fillStyle = '#000';
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (!walk(x, y)) ctx.fillRect(O + x * T, O + y * T, T, T);
  drawWalls(ctx, bd, sheet, W, H, walk, R);

  const door = art.sprites[`door/${bd.door}`] || art.sprites['door/door_01_normaldoor.png'];
  if (door) {
    for (const d of lay.door_list) {
      if (doorState(d[2]) !== 'open') continue;
      const p = doorPlace(d, walk);
      ctx.save();
      ctx.translate(O + (p.x + 0.5) * T, O + (p.y + 0.5) * T);
      ctx.rotate(p.angle);
      ctx.drawImage(image(door[0]).img, -door[3], -door[4]);
      ctx.restore();
    }
  }

  const keys = spriteKeys(art, lay, room, bd, floor.stage).filter((k) => k.key);
  keys.sort((a, q) => (q.floor - a.floor) || (a.sp.y - q.sp.y) || (a.sp.x - q.sp.x));
  for (const { sp, key } of keys) {
    const [url, , , ox, oy] = art.sprites[key];
    ctx.drawImage(image(url).img, Math.round(O + (sp.x + 0.5) * T - ox), Math.round(O + (sp.y + 0.5) * T - oy));
  }
  return canvas;
}

// which spawn points have no sprite (drawn as markers on top)
export function unshown(art, lay, room, floor) {
  const [bd] = backdropOf(art, room, floor.backdrop);
  if (!bd) return lay.spawns;
  return spriteKeys(art, lay, room, bd, floor.stage).filter((k) => !k.key && !k.silent && k.e && k.e.kind !== 'none').map((k) => k.sp);
}

export const TILE = T;
export const ORIGIN = O;
