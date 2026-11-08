/* 查看器"真实美术"模式：anm2 渲染器（Canvas 2D）+ 房间（backdrop 拼接、网格、门）+ 实体 + 玩家（皮肤 + 服装）+ HUD。
 * 素材：服务模式从 asset/<键> 读（只读解包目录），自包含 HTML 用内嵌的 D.art.assets。键 = 小写、正斜杠、相对 resources 根。
 * 坐标：实体 x/y 是房间像素坐标（世界单位）；精灵像素 = 世界单位 × 0.65（一格 40 世界单位 = 26 精灵像素）。
 * 估计成分（Lua 读不到的）：backdrop 墙面变体（按装饰种子挑）、v1 文件的精灵文件/翻转/颜色/叠加帧、网格变体帧、服装叠放顺序。
 */
(function (root) {
  'use strict';
  const PX = 0.65;                 // 世界单位 -> 精灵像素
  const DEG = Math.PI / 180;

  function normKey(p) {
    p = String(p || '').replace(/\\/g, '/').toLowerCase().trim();
    const out = [];
    for (const seg of p.split('/')) {
      if (!seg || seg === '.') continue;
      if (seg === '..') out.pop(); else out.push(seg);
    }
    return out.join('/');
  }
  const dirOf = (k) => (k.lastIndexOf('/') >= 0 ? k.slice(0, k.lastIndexOf('/') + 1) : '');

  // ------------------------------------------------------------------ 素材
  class Assets {
    constructor(opts) {
      this.embedded = opts.embedded || null;   // {键: 文本 | data URI}
      this.base = opts.base || 'asset/';
      this.onload = opts.onload || (() => {});
      this.imgs = new Map(); this.anms = new Map(); this.texts = new Map();
      this.missing = new Set(); this.pending = 0; this.loaded = 0;
    }
    _url(key) {
      if (this.embedded) return this.embedded[key] !== undefined ? this.embedded[key] : null;
      return this.base + key.split('/').map(encodeURIComponent).join('/');
    }
    image(key) {
      key = normKey(key);
      let e = this.imgs.get(key);
      if (e) return e.ok ? e.img : null;
      e = { ok: false, img: null };
      this.imgs.set(key, e);
      const url = this._url(key);
      if (!url) { this.missing.add(key); return null; }
      const im = new Image();
      this.pending++;
      im.onload = () => { e.ok = true; e.img = im; this.pending--; this.loaded++; this.onload(); };
      im.onerror = () => { this.missing.add(key); this.pending--; this.onload(); };
      im.src = url;
      return null;
    }
    anm2(key) {
      key = normKey(key);
      let e = this.anms.get(key);
      if (e) return e.anm;
      e = { anm: null };
      this.anms.set(key, e);
      const url = this._url(key);
      if (!url) { this.missing.add(key); return null; }
      if (this.embedded) {
        try { e.anm = parseAnm2(url, key); } catch (err) { this.missing.add(key); }
        return e.anm;
      }
      this.pending++;
      fetch(url).then((r) => (r.ok ? r.text() : Promise.reject(r.status))).then((t) => {
        e.anm = parseAnm2(t, key); this.pending--; this.loaded++; this.onload();
      }).catch(() => { this.missing.add(key); this.pending--; this.onload(); });
      return null;
    }
  }

  // ------------------------------------------------------------------ anm2
  const num = (el, a, d) => { const v = el.getAttribute(a); return v === null || v === '' ? d : +v; };
  const bool = (el, a, d) => { const v = el.getAttribute(a); return v === null ? d : v === 'true'; };
  function parseFrame(el) {
    return {
      x: num(el, 'XPosition', 0), y: num(el, 'YPosition', 0), px: num(el, 'XPivot', 0), py: num(el, 'YPivot', 0),
      cx: num(el, 'XCrop', 0), cy: num(el, 'YCrop', 0), w: num(el, 'Width', 0), h: num(el, 'Height', 0),
      sx: num(el, 'XScale', 100) / 100, sy: num(el, 'YScale', 100) / 100, delay: Math.max(1, num(el, 'Delay', 1)),
      vis: bool(el, 'Visible', true), r: num(el, 'RedTint', 255) / 255, g: num(el, 'GreenTint', 255) / 255,
      b: num(el, 'BlueTint', 255) / 255, a: num(el, 'AlphaTint', 255) / 255, ro: num(el, 'RedOffset', 0) / 255,
      go: num(el, 'GreenOffset', 0) / 255, bo: num(el, 'BlueOffset', 0) / 255, rot: num(el, 'Rotation', 0),
      interp: bool(el, 'Interpolated', false),
    };
  }
  function track(frames) { // 预计算每个关键帧的起始时间
    let t = 0;
    for (const f of frames) { f.t0 = t; t += f.delay; }
    return { frames, total: t };
  }
  function parseAnm2(text, key) {
    const doc = new DOMParser().parseFromString(text, 'text/xml');
    const dir = dirOf(key);
    const sheets = {}, layers = {}, anims = new Map();
    doc.querySelectorAll('Spritesheets > Spritesheet').forEach((s) => { sheets[s.getAttribute('Id')] = normKey(dir + s.getAttribute('Path')); });
    doc.querySelectorAll('Layers > Layer').forEach((l) => { layers[l.getAttribute('Id')] = { name: l.getAttribute('Name'), sheet: l.getAttribute('SpritesheetId') }; });
    const animsEl = doc.querySelector('Animations');
    const def = animsEl ? animsEl.getAttribute('DefaultAnimation') : null;
    doc.querySelectorAll('Animations > Animation').forEach((a) => {
      const ro = a.querySelector('RootAnimation');
      const rootT = track(ro ? Array.from(ro.querySelectorAll('Frame')).map(parseFrame) : []);
      const ls = [];
      a.querySelectorAll('LayerAnimations > LayerAnimation').forEach((la) => {
        const fr = Array.from(la.querySelectorAll('Frame')).map(parseFrame);
        if (!fr.length) return;
        ls.push({ id: la.getAttribute('LayerId'), visible: bool(la, 'Visible', true), tr: track(fr) });
      });
      anims.set(a.getAttribute('Name'), { name: a.getAttribute('Name'), frameNum: num(a, 'FrameNum', 1), loop: bool(a, 'Loop', false), root: rootT, layers: ls });
    });
    const lower = new Map();
    for (const k of anims.keys()) lower.set(k.toLowerCase(), k);
    return { key, sheets, layers, anims, def, lower };
  }
  function getAnim(anm, name) {
    if (!anm) return null;
    if (name && anm.anims.has(name)) return anm.anims.get(name);
    if (name && anm.lower.has(name.toLowerCase())) return anm.anims.get(anm.lower.get(name.toLowerCase()));
    return null;
  }
  // 第 t 帧的（插值后）关键帧
  function frameAt(tr, t) {
    const fs = tr.frames;
    if (!fs.length) return null;
    if (t < 0) t = 0;
    let i = fs.length - 1;
    if (t < tr.total) { let lo = 0, hi = fs.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (fs[m].t0 <= t) lo = m; else hi = m - 1; } i = lo; }
    const f = fs[i];
    if (!f.interp || i + 1 >= fs.length || t >= tr.total) return f;
    const n = fs[i + 1], k = (t - f.t0) / f.delay;
    if (k <= 0) return f;
    const L = (a, b) => a + (b - a) * k;
    return Object.assign({}, f, { x: L(f.x, n.x), y: L(f.y, n.y), sx: L(f.sx, n.sx), sy: L(f.sy, n.sy), rot: L(f.rot, n.rot),
      r: L(f.r, n.r), g: L(f.g, n.g), b: L(f.b, n.b), a: L(f.a, n.a), ro: L(f.ro, n.ro), go: L(f.go, n.go), bo: L(f.bo, n.bo) });
  }

  // ------------------------------------------------------------------ 着色（乘色 + 加色偏移；按 图片/裁剪/颜色 缓存）
  const tintCache = new Map();
  let tintOrder = [];
  function tinted(img, key, cx, cy, w, h, r, g, b, ro, go, bo) {
    const q = (v) => Math.round(v * 100);
    const k = `${key}|${cx},${cy},${w},${h}|${q(r)},${q(g)},${q(b)},${q(ro)},${q(go)},${q(bo)}`;
    let c = tintCache.get(k);
    if (c) return c;
    c = document.createElement('canvas'); c.width = Math.max(1, w); c.height = Math.max(1, h);
    const x = c.getContext('2d');
    x.drawImage(img, cx, cy, w, h, 0, 0, w, h);
    try {
      const d = x.getImageData(0, 0, c.width, c.height), p = d.data;
      const R0 = ro * 255, G0 = go * 255, B0 = bo * 255;
      for (let i = 0; i < p.length; i += 4) {
        if (!p[i + 3]) continue;
        p[i] = Math.max(0, Math.min(255, p[i] * r + R0));
        p[i + 1] = Math.max(0, Math.min(255, p[i + 1] * g + G0));
        p[i + 2] = Math.max(0, Math.min(255, p[i + 2] * b + B0));
      }
      x.putImageData(d, 0, 0);
    } catch (e) { /* 跨域图片无法读像素：退化为不着色 */ }
    tintCache.set(k, c); tintOrder.push(k);
    if (tintOrder.length > 3000) { for (const old of tintOrder.splice(0, 1000)) tintCache.delete(old); }
    return c;
  }

  // 在当前变换（原点 = 精灵原点，单位 = 精灵像素）下画一个动画的一帧。
  // o: {color: [r,g,b,a,ro,go,bo], replace: {sheetId|路径: 新路径}, layers: Set(层名) 只画这些 / skip: Set(层名)}
  // 返回 true = 画了（素材已就绪）
  const WHITE = [1, 1, 1, 1, 0, 0, 0];
  const SKIP_BRIDGE = new Set(['bridge']);
  const DOOR_SKIN = { 2: 'gfx/grid/door_00_shopdoor.png', 13: 'gfx/grid/door_00_sacrificeroomdoor.png', 21: 'gfx/grid/door_00_diceroomdoor.png',
    20: 'gfx/grid/door_02b_chestroomdoor.png' };
  function drawAnim(ctx, A, anm, animName, t, o) {
    const an = getAnim(anm, animName);
    if (!an) return false;
    o = o || {};
    const col = o.color || WHITE;
    const rf = frameAt(an.root, t) || { x: 0, y: 0, sx: 1, sy: 1, rot: 0, r: 1, g: 1, b: 1, a: 1, ro: 0, go: 0, bo: 0, vis: true };
    if (rf.vis === false) return true;
    let drew = false;
    for (const la of an.layers) {
      if (!la.visible) continue;
      const L = anm.layers[la.id];
      if (!L) continue;
      if (o.layers && !o.layers.has(L.name)) continue;
      if (o.skip && o.skip.has(L.name)) continue;
      const f = frameAt(la.tr, t);
      if (!f || !f.vis || f.w <= 0 || f.h <= 0) continue;
      let sheet = anm.sheets[L.sheet];
      if (o.replace) sheet = o.replace[L.sheet] || o.replace[sheet] || sheet;
      const img = A.image(sheet);
      if (!img) continue;
      const r = f.r * rf.r * col[0], g = f.g * rf.g * col[1], b = f.b * rf.b * col[2], a = f.a * rf.a * col[3];
      const ro = f.ro + rf.ro + col[4], go = f.go + rf.go + col[5], bo = f.bo + rf.bo + col[6];
      if (a <= 0.003) continue;
      ctx.save();
      if (rf.x || rf.y) ctx.translate(rf.x, rf.y);
      if (rf.rot) ctx.rotate(rf.rot * DEG);
      if (rf.sx !== 1 || rf.sy !== 1) ctx.scale(rf.sx, rf.sy);
      ctx.translate(f.x, f.y);
      if (f.rot) ctx.rotate(f.rot * DEG);
      if (f.sx !== 1 || f.sy !== 1) ctx.scale(f.sx, f.sy);
      if (a < 1) ctx.globalAlpha *= a;
      const plain = Math.abs(r - 1) < 0.005 && Math.abs(g - 1) < 0.005 && Math.abs(b - 1) < 0.005 && Math.abs(ro) < 0.003 && Math.abs(go) < 0.003 && Math.abs(bo) < 0.003;
      if (plain) ctx.drawImage(img, f.cx, f.cy, f.w, f.h, -f.px, -f.py, f.w, f.h);
      else ctx.drawImage(tinted(img, sheet, f.cx, f.cy, f.w, f.h, r, g, b, ro, go, bo), -f.px, -f.py);
      ctx.restore();
      drew = true;
    }
    return drew;
  }

  // ------------------------------------------------------------------ 渲染器
  class ArtRenderer {
    constructor(opts) {
      this.T = opts.tables || {};                        // art.py tables()
      this.A = new Assets({ embedded: opts.assets, base: opts.base, onload: opts.onload });
      this.run = opts.run;
      this.bgCache = new Map();
      this.stats = { costumes_drawn: new Set(), costumes_skipped: new Set(), entities_no_anm: new Set(), anim_missing: new Set() };
      this.v1 = !this.run.appNames;
      this.hash = (x) => { x = (x ^ 0x9e3779b9) >>> 0; x = Math.imul(x ^ (x >>> 16), 0x85ebca6b) >>> 0; x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35) >>> 0; return (x ^ (x >>> 16)) >>> 0; };
    }
    entRec(t, v, s) { const E = this.T.entities || {}; return E[`${t}.${v}.${s}`] || E[`${t}.${v}`] || E[`${t}`] || null; }
    backdrop(id) { return (this.T.backdrops || {})[id] || (this.T.backdrops || {})['1'] || null; }

    // ---- 房间背景（按进房缓存成离屏画布，单位 = 精灵像素；原点 = 世界 (-40, 40)）
    background(room) {
      const key = room.at + ':' + room.backdrop;
      let c = this.bgCache.get(key);
      if (c && c.ready) return c;
      const bd = this.backdrop(String(room.backdrop));
      const img = bd && bd.gfx ? this.A.image(bd.gfx) : null;
      const gw = room.gw, gh = room.gh;
      const W = (gw + 3) * 26, H = (gh + 3) * 26;
      if (!c) { c = { canvas: document.createElement('canvas'), ready: false, ox: -40, oy: 40 }; this.bgCache.set(key, c); if (this.bgCache.size > 40) this.bgCache.delete(this.bgCache.keys().next().value); }
      if (!img) return c;
      c.canvas.width = W; c.canvas.height = H;
      const x = c.canvas.getContext('2d');
      x.imageSmoothingEnabled = false;
      x.fillStyle = '#000'; x.fillRect(0, 0, W, H);
      const seed = (room.seeds && room.seeds[1]) || 0;  // DecorationSeed：变体只能估计（引擎的挑法 Lua 读不到）
      const nq = Math.max(1, Math.floor(img.width / 234) * Math.floor(Math.min(img.height, 312) / 156));
      // 墙面变体：引擎按装饰种子挑象限变体（Lua 读不到）。按种子随机挑与实录对照常错（带血/虫痕的变体被挑中而游戏是素墙），
      // 默认一律用第 0 号（无装饰）象限；window.__rv.art.randomWalls = true 时改为按种子随机（估计）
      const quad = (k) => { const q = this.randomWalls ? this.hash(seed + k * 7919) % nq : 0; const qc = Math.floor(img.width / 234); return [(q % qc) * 234, Math.floor(q / qc) * 156]; };
      const blit = (sx, sy, sw, sh, dx, dy, fx, fy) => {
        x.save(); x.translate(dx + (fx ? sw : 0), dy + (fy ? sh : 0)); x.scale(fx ? -1 : 1, fy ? -1 : 1);
        x.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh); x.restore();
      };
      const shape = room.shape;
      // 内部（地板）范围：top_left..bottom_right（世界坐标 ×100）-> 精灵像素（相对画布原点）
      const tl = room.top_left.map((v) => v / 100), br = room.bottom_right.map((v) => v / 100);
      const fx0 = (tl[0] - c.ox) * PX, fy0 = (tl[1] - c.oy) * PX, fx1 = (br[0] - c.ox) * PX, fy1 = (br[1] - c.oy) * PX;
      if (shape === 1 && gw === 15 && gh === 9) {
        // 1x1：四个 234x156 象限，镜像拼（与游戏相同的拼法；每象限的变体按种子估计）
        const q = [quad(0), quad(1), quad(2), quad(3)];
        blit(q[0][0], q[0][1], 234, 156, 0, 0, false, false);
        blit(q[1][0], q[1][1], 234, 156, 234, 0, true, false);
        blit(q[2][0], q[2][1], 234, 156, 0, 156, false, true);
        blit(q[3][0], q[3][1], 234, 156, 234, 156, true, true);
      } else {
        // 其它形状：地板 = 象限地板区平铺（交替镜像）；墙 = 象限的墙带切片沿内部边界铺；凸角用象限角块，L 形内角用 104x104 内角块
        const q0 = quad(0);
        const FW = 169, FH = 91, FX = q0[0] + 65, FY = q0[1] + 65;
        x.save(); x.beginPath(); x.rect(fx0, fy0, fx1 - fx0, fy1 - fy0); x.clip();
        for (let yy = fy0, j = 0; yy < fy1; yy += FH, j++) for (let xx = fx0, i = 0; xx < fx1; xx += FW, i++) blit(FX, FY, FW, FH, xx, yy, i % 2 === 1, j % 2 === 1);
        x.restore();
        const voidCell = (cc, rr) => isVoid(shape, cc, rr);
        // L 形缺角：地板铺满了外接矩形，把缺角格涂黑（墙在下面按格补）
        x.fillStyle = '#000';
        for (let rr = 0; rr < gh; rr++) for (let cc = 0; cc < gw; cc++) if (voidCell(cc, rr)) x.fillRect((20 + 40 * cc - c.ox) * PX - 1, (100 + 40 * rr - c.oy) * PX - 1, 28, 28);
        // 逐格画墙：内部格四周若是墙格，就在那一侧画墙带（65 像素宽：1 格墙 + 1.5 格外沿）
        const top = (cx0, cy0) => blit(q0[0] + 104, q0[1], 26, 65, cx0, cy0 - 65, false, false);
        const bottom = (cx0, cy0) => blit(q0[0] + 104, q0[1], 26, 65, cx0, cy0, false, true);
        const left = (cx0, cy0) => blit(q0[0], q0[1] + 104, 65, 26, cx0 - 65, cy0, false, false);
        const right = (cx0, cy0) => blit(q0[0], q0[1] + 104, 65, 26, cx0, cy0, true, false);
        const inside = (cc, rr) => {
          const wx = 20 + 40 * cc, wy = 100 + 40 * rr;
          return cc > 0 && rr > 0 && cc < gw - 1 && rr < gh - 1 && wx >= tl[0] - 1 && wy >= tl[1] - 1 && wx + 40 <= br[0] + 1 && wy + 40 <= br[1] + 1 && !voidCell(cc, rr);
        };
        for (let rr = 0; rr < gh; rr++) for (let cc = 0; cc < gw; cc++) {
          if (!inside(cc, rr)) continue;
          const px = (20 + 40 * cc - c.ox) * PX, py = (100 + 40 * rr - c.oy) * PX;
          if (!inside(cc, rr - 1)) top(px, py);
          if (!inside(cc, rr + 1)) bottom(px, py + 26);
          if (!inside(cc - 1, rr)) left(px, py);
          if (!inside(cc + 1, rr)) right(px + 26, py);
          // 凸角
          if (!inside(cc - 1, rr) && !inside(cc, rr - 1)) blit(q0[0], q0[1], 65, 65, px - 65, py - 65, false, false);
          if (!inside(cc + 1, rr) && !inside(cc, rr - 1)) blit(q0[0], q0[1], 65, 65, px + 26, py - 65, true, false);
          if (!inside(cc - 1, rr) && !inside(cc, rr + 1)) blit(q0[0], q0[1], 65, 65, px - 65, py + 26, false, true);
          if (!inside(cc + 1, rr) && !inside(cc, rr + 1)) blit(q0[0], q0[1], 65, 65, px + 26, py + 26, true, true);
        }
        // L 形内角（贴图右上角的 104x104 内角块，没有就跳过）
        if (shape >= 9 && img.width >= 572) {
          const ic = { 9: [13, 7, false, false], 10: [14, 7, true, false], 11: [13, 8, false, true], 12: [14, 8, true, true] }[shape];
          if (ic) {
            const px = (20 + 40 * ic[0] - c.ox) * PX, py = (100 + 40 * ic[1] - c.oy) * PX;
            blit(468, 0, 104, 104, px + (ic[2] ? -78 : 0), py + (ic[3] ? -78 : 0), ic[2], ic[3]);
          }
        }
      }
      c.ready = true;
      return c;
    }

    // ---- 网格实体（墙/门格以外）
    gridSprite(g, gspr, bd) {
      const [idx, type, variant, state, coll] = g;
      const str = (id) => this.run.str(id);
      if (gspr) {
        const s = gspr.get(idx);
        if (s) return { file: normKey(str(s[1])), anim: str(s[2]), frame: s[3] };
      }
      // v1：按类型估计
      const h = this.hash(idx * 131 + (variant || 0));
      const dead = coll === 0;
      switch (type) {
        case 2: return { file: 'gfx/grid/grid_rock.anm2', anim: dead ? 'rubble' : 'normal', frame: dead ? 0 : h % 3 };
        case 3: return { file: 'gfx/grid/grid_rock.anm2', anim: 'black', frame: 0 };
        case 4: return { file: 'gfx/grid/grid_rock.anm2', anim: dead ? 'rubble' : 'tinted', frame: 0 };
        case 5: return { file: 'gfx/grid/grid_rock.anm2', anim: dead ? 'rubble' : 'bombrock', frame: 0 };
        case 6: return { file: 'gfx/grid/grid_rock.anm2', anim: dead ? 'rubble_alt' : 'alt', frame: dead ? 0 : h % 3 };
        case 22: return { file: 'gfx/grid/grid_rock.anm2', anim: dead ? 'ss_broken' : 'superspecial', frame: 0 };
        case 24: return { file: 'gfx/grid/grid_rock.anm2', anim: 'pillar', frame: 0 };
        case 25: return { file: 'gfx/grid/grid_rock.anm2', anim: dead ? 'rubble_spiked' : 'spiked', frame: 0 };
        case 27: return { file: 'gfx/grid/grid_rock.anm2', anim: dead ? 'rubble' : 'foolsgold', frame: 0 };
        case 7: return { file: 'gfx/grid/grid_pit.anm2', anim: 'pit', frame: 0 };
        case 8: return { file: 'gfx/grid/grid_spikes.anm2', anim: 'Spikes0' + (1 + h % 4), frame: 0 };
        case 9: return { file: 'gfx/grid/grid_spikes.anm2', anim: dead ? 'Unsummon' : 'Spikes0' + (1 + h % 4), frame: dead ? 11 : 0 };
        case 10: return { file: 'gfx/grid/grid_web.anm2', anim: dead ? 'Bombed' : 'Idle', frame: 0 };
        case 11: return { file: 'gfx/grid/grid_locks.anm2', anim: dead ? 'Broken' : 'Idle', frame: 0 };
        case 12: return { file: 'gfx/grid/grid_tnt.anm2', anim: null, frame: dead ? 99 : Math.min(4, Math.floor((state || 0) / 1)) };
        case 14: return { file: 'gfx/grid/grid_poop.anm2', anim: 'State' + Math.min(5, 1 + Math.floor((state || 0) / 250)), frame: 0 };
        case 17: return { file: 'gfx/grid/door_11_trapdoor.anm2', anim: state ? 'Opened' : 'Closed', frame: 0 };
        case 20: return { file: 'gfx/grid/grid_pressureplate.anm2', anim: state ? 'On' : 'Off', frame: 0 };
        case 23: return { file: 'gfx/grid/grid_teleporter.anm2', anim: null, frame: 0 };
        default: return null;
      }
    }
    gridReplace(file, bd) {
      if (!bd) return null;
      if (file === 'gfx/grid/grid_rock.anm2' && bd.rocks) return { 0: bd.rocks };
      if (file === 'gfx/grid/grid_pit.anm2' && bd.pit) return { 0: bd.pit, 1: bd.bridge || bd.rocks };
      if (file === 'gfx/grid/grid_pit_water.anm2') return { 0: bd.waterpit || 'gfx/grid/grid_pit_water.png', 1: bd.bridge || bd.rocks };
      if (file === 'gfx/grid/door_01_normaldoor.anm2' && bd.door) return { 0: bd.door };
      if (file === 'gfx/grid/door_08_holeinwall.anm2' && bd.holeinwall) return { 0: bd.holeinwall };
      if (file === 'gfx/grid/grid_locks.anm2' && bd.door) return { 1: bd.door };
      return null;
    }
    drawGrid(ctx, rs, M) {
      const room = rs.room, bd = this.backdrop(String(room.backdrop));
      const gw = room.gw;
      const list = [];
      for (const g of rs.grid.values()) {
        if (g[1] === 15 || g[1] === 16 || g[1] === 0) continue;
        const sp = this.gridSprite(g, rs.gspr, bd);
        if (!sp || !sp.file) continue;
        list.push([g, sp]);
      }
      // 坑先画（贴地），其余按行
      list.sort((a, b) => (a[0][1] === 7 ? -1 : 0) - (b[0][1] === 7 ? -1 : 0) || a[0][0] - b[0][0]);
      for (const [g, sp] of list) {
        const anm = this.A.anm2(sp.file);
        if (!anm) continue;
        const c = g[0] % gw, r = Math.floor(g[0] / gw);
        const an = sp.anim || anm.def || (anm.anims.keys().next().value);
        M.at(40 + 40 * c, 120 + 40 * r);
        // 坑的 bridge 层只在搭了桥（碰撞 = 0）时画
        const skip = g[1] === 7 && g[4] !== 0 ? SKIP_BRIDGE : null;
        drawAnim(ctx, this.A, anm, an, sp.frame, { replace: this.gridReplace(sp.file, bd), skip });
        ctx.restore();
      }
    }
    // ---- 门
    doorSprite(d, dspr, room) {
      const [slot, target, ttype, open, locked] = d;
      if (dspr) {
        const s = dspr.get(slot);
        if (s) return { file: normKey(this.run.str(s[1])), anim: this.run.str(s[2]), frame: s[3], rot: s[4] === null ? 0 : s[4] / 100, ox: (s[5] || 0) / 100, oy: (s[6] || 0) / 100 };
      }
      const side = slot % 4;
      const rot = [-90, 0, 90, 180][side], off = [[15, 0], [0, 15], [-15, 0], [0, -15]][side];
      const cur = room.type;
      const t = (ttype === 1 || ttype === 0) ? cur : ttype;
      const F = { 4: 'door_02_treasureroomdoor', 5: 'door_10_bossroomdoor', 6: 'door_10_bossroomdoor', 7: 'door_08_holeinwall', 8: 'door_08_holeinwall',
        9: 'door_05_arcaderoomdoor', 10: 'door_04_selfsacrificeroomdoor', 11: 'door_03_ambushroomdoor', 13: 'door_04_selfsacrificeroomdoor', 14: 'door_07_devilroomdoor', 15: 'door_07_holyroomdoor' };
      const file = 'gfx/grid/' + (F[t] || 'door_01_normaldoor') + '.anm2';
      const anim = (t === 7 || t === 8) ? (open ? 'Opened' : 'Hidden') : open ? 'Opened' : locked ? 'KeyClosed' : 'Closed';
      return { file, anim, frame: 0, rot, ox: off[0], oy: off[1] };
    }
    drawDoors(ctx, rs, M) {
      const room = rs.room, bd = this.backdrop(String(room.backdrop));
      const gw = room.gw, gh = room.gh;
      for (const d of rs.doors.values()) {
        const sp = this.doorSprite(d, rs.dspr, room);
        if (sp.anim === 'Hidden') continue;   // 未炸开的隐藏房门：游戏里不画
        const anm = this.A.anm2(sp.file);
        let rep = this.gridReplace(sp.file, bd);
        if (sp.file === 'gfx/grid/door_01_normaldoor.anm2') {
          // 普通门 anm2 按房间类型换贴图（目标房间是普通房时看当前房间；对照实录核对：献祭房门 = door_00_sacrificeroomdoor）
          const t = (d[2] === 1 || d[2] === 0) ? room.type : d[2];
          const sk = DOOR_SKIN[t];
          if (sk) rep = { 0: sk };
        }
        if (!anm) continue;
        const ci = doorCell(room.shape, d[0], gw, gh);
        const c = ci % gw, r = Math.floor(ci / gw);
        M.at(40 + 40 * c, 120 + 40 * r);
        ctx.translate(sp.ox, sp.oy);
        if (sp.rot) ctx.rotate(sp.rot * DEG);
        if (!drawAnim(ctx, this.A, anm, sp.anim, sp.frame, { replace: rep })) drawAnim(ctx, this.A, anm, anm.def, 0, { replace: rep });
        ctx.restore();
      }
    }

    // ---- 实体
    entityLook(row, app, EI, XI) {
      const t = row[1], v = row[2], s = row[3];
      const rec = this.entRec(t, v, s);
      let file = app && app.file ? normKey(app.file) : (rec ? rec[0] : null);
      const look = { file, anim: row[EI.anim], frame: row[EI.aframe] || 0, ova: app ? app.ov_anim : null, ovf: app ? app.ov_frame : 0,
        flip: app ? app.flip : 0, sx: app ? app.sx : 1, sy: app ? app.sy : 1, rot: app ? app.rot : 0,
        color: app ? mulColor([app.r, app.g, app.b, app.a, app.ro, app.go, app.bo], [app.sr, app.sg, app.sb, app.sa, app.sro, app.sgo, app.sbo]) : null,
        ox: app ? app.ox : 0, oy: app ? app.oy : 0, pox: app ? app.pox : 0, poy: app ? app.poy : 0, shadow: rec ? rec[1] : 0, replace: null };
      if (!app && (t === 2 || t === 9)) { // v1：泪/敌弹的高度
        const xn = XI[t === 2 ? 'tear' : 'projectile'];
        if (xn && row[xn.height] !== null && row[xn.height] !== undefined) look.poy = row[xn.height] / 100;
      }
      if (t === 5 && v === 100 && file) { // 道具台：物品层换成道具图
        const g = (this.T.items_gfx || {})[s];
        if (g) look.replace = { 1: g };
      } else if (t === 5 && v === 350) {
        const g = (this.T.trinkets_gfx || {})[s & 0x7fff];
        if (g) look.replace = { 0: g };
      }
      return look;
    }
    drawEntity(ctx, look, x, y, M) {
      const anm = look.file ? this.A.anm2(look.file) : null;
      if (!anm) { if (!look.file) this.stats.entities_no_anm.add('?'); return false; }
      if (look.shadow > 0) {
        M.at(x, y);
        ctx.globalAlpha *= 0.35; ctx.fillStyle = '#000';
        ctx.beginPath(); ctx.ellipse(0, 0, look.shadow * 0.33 * (look.sx || 1), look.shadow * 0.13 * (look.sy || 1), 0, 0, 7); ctx.fill();
        ctx.restore();
      }
      M.at(x + (look.pox || 0), y + (look.poy || 0));
      if (look.ox || look.oy) ctx.translate(look.ox, look.oy);
      if (look.rot) ctx.rotate(look.rot * DEG);
      const fx = (look.flip & 1) ? -1 : 1, fy = (look.flip & 2) ? -1 : 1;
      const sx = (look.sx === null || look.sx === undefined ? 1 : look.sx) * fx, sy = (look.sy === null || look.sy === undefined ? 1 : look.sy) * fy;
      if (sx !== 1 || sy !== 1) ctx.scale(sx, sy);
      const o = { color: look.color, replace: look.replace };
      let ok = drawAnim(ctx, this.A, anm, look.anim, look.frame, o);
      if (!ok && !getAnim(anm, look.anim)) { this.stats.anim_missing.add(anm.key + '#' + look.anim); ok = drawAnim(ctx, this.A, anm, anm.def, look.frame, o); }
      if (look.ova) drawAnim(ctx, this.A, anm, look.ova, look.ovf || 0, o);
      ctx.restore();
      return ok;
    }

    // ---- 玩家：001.000_player.anm2 换皮肤 + 服装（道具 -> costumes2.xml）
    costumesFor(ptype, items) {
      const out = [];
      const pd = (this.T.players || {})[ptype];
      if (pd && pd.costume && (this.T.costumes_id || {})[pd.costume]) out.push((this.T.costumes_id)[pd.costume]);
      for (const id of Object.keys(items || {})) {
        for (const rec of ((this.T.costumes_item || {})[id] || [])) out.push(rec);
      }
      out.sort((a, b) => a[1] - b[1]);
      return out;
    }
    drawPlayer(ctx, p, app, items, M) {
      const ptype = p.ptype;
      const pd = (this.T.players || {})[ptype];
      const anm = this.A.anm2('gfx/001.000_player.anm2');
      if (!anm) return false;
      const rep = pd && pd.skin ? { 0: pd.skin } : null;
      const x = p.x, y = p.y;
      const look = { flip: app ? app.flip : 0, sx: app ? app.sx : 1, sy: app ? app.sy : 1, rot: app ? app.rot : 0,
        color: app ? mulColor([app.r, app.g, app.b, app.a, app.ro, app.go, app.bo], [app.sr, app.sg, app.sb, app.sa, app.sro, app.sgo, app.sbo]) : null };
      // 受伤无敌闪烁（v1 没有颜色时按 damage_cooldown 估计）
      M.at(x, y);
      ctx.globalAlpha *= 0.35; ctx.fillStyle = '#000';
      ctx.beginPath(); ctx.ellipse(0, 0, 9 * (look.sx || 1), 3.5, 0, 0, 7); ctx.fill();
      ctx.restore();
      M.at(x + (app ? app.pox || 0 : 0), y + (app ? app.poy || 0 : 0));
      if (app && (app.ox || app.oy)) ctx.translate(app.ox, app.oy);
      if (look.rot) ctx.rotate(look.rot * DEG);
      const fx = (look.flip & 1) ? -1 : 1;
      ctx.scale((look.sx || 1) * fx, (look.sy || 1) * ((look.flip & 2) ? -1 : 1));
      if (!app && p.damage_cooldown > 0 && (p.damage_cooldown % 6) < 3) ctx.globalAlpha *= 0.35;
      const o = { color: look.color, replace: rep };
      const body = p.anim, head = p.overlay, bf = p.aframe || 0, hf = app ? (app.ov_frame || 0) : 0;
      const cos = this.costumesFor(ptype, items);
      drawAnim(ctx, this.A, anm, body, bf, o);
      for (const c of cos) { const ca = this.A.anm2(c[0]); if (ca && getAnim(ca, body)) { drawAnim(ctx, this.A, ca, body, bf, { color: look.color }); } }
      if (head) drawAnim(ctx, this.A, anm, head, hf, o);
      for (const c of cos) {
        const ca = this.A.anm2(c[0]);
        if (!ca) { this.stats.costumes_skipped.add(c[0]); continue; }
        if (head && getAnim(ca, head)) { drawAnim(ctx, this.A, ca, head, hf, { color: look.color }); this.stats.costumes_drawn.add(c[0]); }
        else if (getAnim(ca, body)) this.stats.costumes_drawn.add(c[0]);
        else this.stats.costumes_skipped.add(c[0]);
      }
      // 服装的 <动画>_Overlay（如 Spoon Bender 的紫色角）画在最上面
      for (const c of cos) {
        const ca = this.A.anm2(c[0]);
        if (!ca) continue;
        if (getAnim(ca, body + '_Overlay')) drawAnim(ctx, this.A, ca, body + '_Overlay', bf, { color: look.color });
        if (head && getAnim(ca, head + '_Overlay')) drawAnim(ctx, this.A, ca, head + '_Overlay', hf, { color: look.color });
      }
      ctx.restore();
      return true;
    }

    // ---- 一整帧（房间 + 网格 + 门 + 实体 + 玩家），M = 坐标助手（at(x, y) 把变换设到该世界点、单位 = 精灵像素，并 save）
    drawFrame(ctx, F, M) {
      const rs = F.rs, room = rs.room;
      if (!room) return;
      const bg = this.background(room);
      if (bg.ready) {
        M.at(bg.ox, bg.oy); ctx.imageSmoothingEnabled = false;
        ctx.drawImage(bg.canvas, 0, 0); ctx.restore();
      }
      // 第 1 层开局房间地板上的操作说明（gfx/backdrop/controls.anm2）
      const lv = F.level;
      if (lv && lv.stage === 1 && room.room_index === lv.start_room && room.shape === 1) {
        const ca = this.A.anm2('gfx/backdrop/controls.anm2');
        if (ca) { M.at(320, 280); drawAnim(ctx, this.A, ca, 'Controls', 0); ctx.restore(); }
      }
      this.drawGrid(ctx, rs, M);
      // 实体：贴地特效（血迹/creep 等）先画，其余与玩家按 y 排序
      const EI = F.EI, XI = F.XI;
      const items = [];
      const floorFx = new Set([7, 22, 23, 24, 25, 26, 32, 43, 44, 45, 46, 53, 56, 78, 89, 90, 94, 95, 108, 109, 110, 145, 146, 160, 161, 162]);
      const er = F.erows || [], ea = F.eapp || [];
      for (let k = 0; k < er.length; k++) {
        const row = er[k];
        if (row[4] === null || row[5] === null) continue;
        if (!row[EI.visible]) continue;
        if (row[1] === 1000 && !F.effects) continue;
        const app = ea[k] ? this.run.decodeAppearance(ea[k]) : null;
        const look = this.entityLook(row, app, EI, XI);
        const floor = row[1] === 1000 && floorFx.has(row[2]);
        items.push({ y: floor ? -1e9 : row[5] / 100, ent: row, look });
      }
      const pr = F.prows || [], pa = F.papp || [];
      for (let k = 0; k < pr.length; k++) {
        const p = this.run.decodePlayer(pr[k]);
        if (p.x === null || !p.visible) continue;
        items.push({ y: p.y, player: p, app: pa[k] ? this.run.decodeAppearance(pa[k], true) : null, items: (F.items || {})[pr[k][0]] });
      }
      this.drawDoors(ctx, rs, M);
      items.sort((a, b) => a.y - b.y);
      for (const it of items) {
        if (it.player) this.drawPlayer(ctx, it.player, it.app, it.items, M);
        else this.drawEntity(ctx, it.look, it.ent[4] / 100, it.ent[5] / 100, M);
      }
    }

    // ---- HUD（屏幕坐标，u = HUD 像素缩放）
    drawHud(ctx, F, x0, y0, u) {
      const p = F.prows && F.prows[0] ? this.run.decodePlayer(F.prows[0]) : null;
      if (!p) return;
      const A = this.A;
      const at = (x, y) => { ctx.save(); ctx.translate(x0 + x * u, y0 + y * u); ctx.scale(u, u); };
      const hearts = A.anm2('gfx/ui/ui_hearts.anm2');
      const list = [];
      const red = p.hearts || 0, max = p.max_hearts || 0, soul = p.soul || 0, black = p.black || 0, bone = p.bone || 0;
      for (let k = 0; k < max; k += 2) list.push(red - k >= 2 ? 'RedHeartFull' : red - k === 1 ? 'RedHeartHalf' : 'EmptyHeart');
      for (let k = 0, s = 0; k < soul; k += 2, s++) {
        const isBlack = (black >> s) & 1;
        list.push((isBlack ? 'BlackHeart' : 'BlueHeart') + (soul - k >= 2 ? 'Full' : 'Half'));
      }
      for (let k = 0; k < bone; k++) list.push('BoneHeartFull');
      for (let k = 0; k < (p.broken || 0); k++) list.push('BrokenHeart');
      if (hearts) list.forEach((nm, k) => { at(48 + (k % 6) * 12, 12 + Math.floor(k / 6) * 10); drawAnim(ctx, A, hearts, nm, 0); ctx.restore(); });
      // 主动道具
      const act = p.active0;
      if (act > 0) {
        const g = (this.T.items_gfx || {})[act];
        const im = g ? A.image(g) : null;
        if (im) { at(4, 4); ctx.imageSmoothingEnabled = false; ctx.drawImage(im, 0, 0, 32, 32); ctx.restore(); }
        const bar = A.anm2('gfx/ui/ui_chargebar.anm2');
        if (bar) { at(38, 17); drawAnim(ctx, A, bar, 'BarEmpty', 0); ctx.restore(); }
      }
      // 拾取物计数
      const hp = A.anm2('gfx/ui/hudpickups.anm2');
      const rows = [['coins', 0], ['bombs', 2], ['keys', 1]];
      rows.forEach(([k, fr], j) => {
        if (hp) { at(4, 45 + j * 12); drawAnim(ctx, A, hp, 'Idle', fr); ctx.restore(); }
        at(20, 45 + j * 12);
        ctx.font = 'bold 9px "Trebuchet MS", sans-serif'; ctx.textBaseline = 'top'; ctx.lineWidth = 2.5; ctx.strokeStyle = '#111'; ctx.fillStyle = '#f2f2f2';
        const txt = String(p[k] ?? 0).padStart(2, '0') + ((k === 'keys' && p.golden_key) || (k === 'bombs' && p.golden_bomb) ? '*' : '');
        ctx.strokeText(txt, 0, 1); ctx.fillText(txt, 0, 1); ctx.restore();
      });
      // Boss 血条（存活的 boss 实体 HP 之和 / 最大 HP 之和；ui_bosshealthbar.png 下半是空槽）
      const xn = F.XI && F.XI.npc;
      if (xn && F.erows) {
        let hp = 0, mx = 0;
        for (const row of F.erows) {
          if (row[1] >= 10 && row[1] < 1000 && row[xn.boss] === 1 && row[F.EI.hp] > 0) { hp += row[F.EI.hp]; mx += row[F.EI.max_hp]; }
        }
        const im = mx > 0 ? A.image('gfx/ui/ui_bosshealthbar.png') : null;
        if (im && F.W) {
          const bw = 150, bx = (F.W / u - bw) / 2, by = F.H / u - 40;
          at(bx, by); ctx.imageSmoothingEnabled = false; ctx.drawImage(im, 0, 32, 150, 32, 0, 0, 150, 32);
          ctx.fillStyle = '#d01010'; ctx.fillRect(22, 9, 108 * Math.max(0, Math.min(1, hp / mx)), 9); ctx.restore();
        }
      }
      // 饰品 / 卡 / 药
      const tr = p.trinket0;
      if (tr) {
        const g = (this.T.trinkets_gfx || {})[tr & 0x7fff];
        const im = g ? A.image(g) : null;
        if (im) { at(4, 236 - 30); ctx.drawImage(im, 0, 0, 32, 32); ctx.restore(); }
      }
    }
    drawMinimap(ctx, F, x0, y0, u) {
      const lv = F.level, room = F.rs.room;
      if (!lv) return;
      const A = this.A, mm = A.anm2('gfx/ui/minimap1.anm2'), icons = A.anm2('gfx/ui/minimap_icons.anm2');
      if (!mm) return;
      const cur = room && room.desc ? room.desc[0] : null;
      const vis = F.visited || new Set();
      let gxMin = 13, gyMin = 13, gxMax = 0, gyMax = 0;
      for (const d of lv.rooms) { const gi = d[1]; if (gi < 0) continue; gxMin = Math.min(gxMin, gi % 13); gyMin = Math.min(gyMin, Math.floor(gi / 13)); gxMax = Math.max(gxMax, gi % 13); gyMax = Math.max(gyMax, Math.floor(gi / 13)); }
      const ICON = { 2: 'IconShop', 4: 'IconTreasureRoom', 5: 'IconBoss', 6: 'IconMiniboss', 7: 'IconSecretRoom', 8: 'IconSuperSecretRoom', 9: 'IconArcade', 10: 'IconCurseRoom',
        11: 'IconAmbushRoom', 12: 'IconLibrary', 13: 'IconSacrificeRoom', 14: 'IconDevilRoom', 15: 'IconAngelRoom', 18: 'IconIsaacsRoom', 19: 'IconBarrenRoom', 20: 'IconChestRoom', 21: 'IconDiceRoom', 24: 'IconPlanetarium' };
      const foot = { 1: [[0, 0]], 2: [[0, 0]], 3: [[0, 0]], 4: [[0, 0], [0, 1]], 5: [[0, 0], [0, 1]], 6: [[0, 0], [1, 0]], 7: [[0, 0], [1, 0]],
        8: [[0, 0], [1, 0], [0, 1], [1, 1]], 9: [[1, 0], [0, 1], [1, 1]], 10: [[0, 0], [0, 1], [1, 1]], 11: [[0, 0], [1, 0], [1, 1]], 12: [[0, 0], [1, 0], [0, 1]] };
      const CW = 9, CH = 8;
      for (const d of lv.rooms) {
        const gi = d[1]; if (gi < 0) continue;
        const isCur = d[0] === cur, seen = vis.has(d[0]);
        const isSecret = d[6] === 7 || d[6] === 8;
        if (!isCur && !seen && isSecret) continue;
        for (const [dx, dy] of (foot[d[9]] || [[0, 0]])) {
          const gx = gi % 13 + dx - gxMin, gy = Math.floor(gi / 13) + dy - gyMin;
          ctx.save(); ctx.translate(x0 + gx * CW * u, y0 + gy * CH * u); ctx.scale(u, u);
          drawAnim(ctx, A, mm, isCur ? 'RoomCurrent' : seen ? 'RoomVisited' : 'RoomUnvisited', 0);
          ctx.restore();
        }
        if (icons && ICON[d[6]] && (seen || isCur || d[6] === 5 || d[6] === 4 || d[6] === 2)) {
          ctx.save(); ctx.translate(x0 + (gi % 13 - gxMin) * CW * u, y0 + (Math.floor(gi / 13) - gyMin) * CH * u); ctx.scale(u, u);
          drawAnim(ctx, A, icons, ICON[d[6]], 0); ctx.restore();
        }
      }
    }
  }

  function mulColor(a, b) {
    const n = (v, d) => (v === null || v === undefined ? d : v);
    return [n(a[0], 1) * n(b[0], 1), n(a[1], 1) * n(b[1], 1), n(a[2], 1) * n(b[2], 1), n(a[3], 1) * n(b[3], 1),
      n(a[4], 0) + n(b[4], 0), n(a[5], 0) + n(b[5], 0), n(a[6], 0) + n(b[6], 0)];
  }
  // 门槽 -> 网格下标（与 viewer_app.js 的 doorCell 相同，REPORT.md 核对过）
  function doorCell(shape, slot, gw, gh) {
    const base = { 0: [0, 4], 1: [7, 0], 2: [gw - 1, 4], 3: [7, gh - 1], 4: [0, 11], 5: [20, 0], 6: [gw - 1, 11], 7: [20, gh - 1] };
    const adj = { 9: { 0: [13, 4], 1: [7, 7] }, 10: { 5: [20, 7], 2: [14, 4] }, 11: { 3: [7, 8], 4: [13, 11] }, 12: { 6: [14, 11], 7: [20, 8] } };
    const p = (adj[shape] && adj[shape][slot]) || base[slot];
    return p ? p[1] * gw + p[0] : -1;
  }
  function isVoid(shape, c, r) {
    if (shape === 9) return c <= 13 && r <= 7;
    if (shape === 10) return c >= 14 && r <= 7;
    if (shape === 11) return c <= 13 && r >= 8;
    if (shape === 12) return c >= 14 && r >= 8;
    return false;
  }

  root.IsaacArt = { ArtRenderer, Assets, parseAnm2, frameAt, drawAnim, normKey, PX };
})(typeof self !== 'undefined' ? self : this);
