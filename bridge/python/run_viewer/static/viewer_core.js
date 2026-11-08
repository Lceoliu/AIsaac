/* 录制查看器核心：把 build.py 产出的数据还原成任意帧的完整状态。
 * 与 rl/bridge/python/load_run.py 逐条对应（expand_rows / expand_ticks / apply_changes / frame()），
 * 浏览器与 node（verify.py 调 node_dump.js）共用这一个文件。不依赖 DOM。
 */
(function (root) {
  'use strict';
  const C = { n: 0, gf: 1, t: 2, kf: 3, tk: 4, r: 5, p: 6, e: 7, g: 8, dr: 9, mt: 10, dg: 11, nsp: 12, ms: 13,
    pa: 14, ea: 15, gs: 16, ds: 17 };   // 14.. = v2 外观（load_run.expand_sparse / apply_changes 的 gs/ds）
  const N_COMMON = 19;
  const isInt = (v) => typeof v === 'number' && Number.isInteger(v);

  function entityCategory(t) {
    const m = { 2: 'tear', 3: 'familiar', 4: 'bomb', 5: 'pickup', 7: 'laser', 8: 'knife', 9: 'projectile', 1000: 'effect' };
    if (m[t]) return m[t];
    return (t >= 10 && t < 1000) ? 'npc' : null;
  }

  // load_run.expand_rows 的逐行移植。prev: Map(key -> 完整行)
  function expandRows(rows, prev, pcol) {
    const out = [], cur = new Map();
    for (const r of rows) {
      let full, pairs = null;
      if (typeof r === 'number') {
        const pv = prev.get(r);
        if (!pv) throw new Error('差分行引用了上一帧不存在的对象 ' + r);
        // 只有 key 的行：除位置预测（x += vx, y += vy）外与上一帧相同。位置不动时直接复用上一帧的数组
        // （行数组展开后从不被修改），省掉大部分分配。
        if (pcol == null || ((!isInt(pv[pcol]) || !isInt(pv[pcol + 2]) || pv[pcol + 2] === 0) &&
                             (!isInt(pv[pcol + 1]) || !isInt(pv[pcol + 3]) || pv[pcol + 3] === 0))) {
          out.push(pv); cur.set(r, pv);
          continue;
        }
        full = pv.slice();
      } else if (r.length === 2 && Array.isArray(r[1])) {
        const pv = prev.get(r[0]);
        if (!pv) throw new Error('差分行引用了上一帧不存在的对象 ' + r[0]);
        full = pv.slice();
        pairs = new Map();
        for (const cv of r[1]) pairs.set(cv[0], cv[1]);
      } else {
        full = r.slice();
        out.push(full); cur.set(full[0], full);
        continue;
      }
      const pos = pcol == null ? [] : [pcol, pcol + 1];
      if (pairs) {
        for (const [col, v] of pairs) {
          if (pos.includes(col)) continue;
          const a = full[col];
          full[col] = (isInt(a) && isInt(v)) ? a + v : v;
        }
      }
      for (const col of pos) {
        const a = full[col], vel = full[col + 2];
        const present = pairs ? pairs.has(col) : false;
        const v = present ? pairs.get(col) : undefined;
        if (isInt(a) && isInt(vel) && (!present || isInt(v))) full[col] = a + vel + (present ? v : 0);
        else if (present) full[col] = (isInt(a) && isInt(v)) ? a + v : v;
      }
      out.push(full); cur.set(full[0], full);
    }
    return [out, cur];
  }

  // load_run.expand_sparse 的逐行移植（v2 外观行 pa/ea）：完整行 / 差分行 [key, [[列, 值], ...]]；未列出的对象沿用上一帧，
  // 不在 keys（本帧 p/e 的对象）里的丢弃。返回 [按 keys 顺序的行（缺失为 null）, 新 prev Map]
  function expandSparse(rows, prev, keys) {
    const cur = new Map();
    for (const k of keys) if (prev.has(k)) cur.set(k, prev.get(k));
    for (const r of (rows || [])) {
      if (r.length === 2 && Array.isArray(r[1])) {
        const pv = cur.get(r[0]);
        if (!pv) throw new Error('外观差分行引用了上一帧不存在的对象 ' + r[0]);
        const full = pv.slice();
        for (const cv of r[1]) { const a = full[cv[0]], v = cv[1]; full[cv[0]] = (isInt(a) && isInt(v)) ? a + v : v; }
        cur.set(r[0], full);
      } else cur.set(r[0], r.slice());
    }
    return [keys.map((k) => (cur.has(k) ? cur.get(k) : null)), cur];
  }

  // load_run.Run.expand_ticks：state = {玩家编号: 12x3}（就地更新），返回 [{kind, inputs, changed, extra}]
  function expandTicks(tk, state) {
    const out = [];
    for (const e of tk) {
      if (typeof e === 'string') {
        if (e.includes('-')) {
          const [a, b] = e.slice(1).split('-').map(Number);
          for (let j = a; j <= b; j++) out.push({ kind: 'i' + j, inputs: Object.assign({}, state), changed: false, extra: [] });
        } else out.push({ kind: e, inputs: Object.assign({}, state), changed: false, extra: [] });
        continue;
      }
      const kind = e[0], changed = e[1];
      for (const [pnum, arr] of changed) state[pnum] = arr;
      const extra = e.length > 2 ? e[2] : [];
      out.push({ kind, inputs: Object.assign({}, state), changed: changed.length > 0, extra });
    }
    return out;
  }

  function bisectRight(arr, x) { // 第一个 > x 的位置
    let lo = 0, hi = arr.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; }
    return lo;
  }

  class RunData {
    constructor(d) {
      this.d = d;
      this.frames = d.frames;
      this.N = d.frames.length;
      const f = d.header.fields;
      this.fields = f;
      const pp = f.position_prediction || {};
      this.pcolP = pp.player == null ? null : pp.player;
      this.pcolE = pp.entity == null ? null : pp.entity;
      this.pfix = new Set(f.fixed_x100.player);
      this.efix = new Set(f.fixed_x100.entity);
      this.xfix = f.fixed_x100.entity_extra || {};
      this.pIndex = {}; f.player.forEach((nm, i) => { this.pIndex[nm] = i; });
      // v2 外观
      this.format = d.format || 1;
      this.appNames = f.appearance || null;
      this.appPlayerExtra = f.player_appearance_extra || [];
      this.afix = new Set(f.appearance_fixed_x100 || []);
      this.astr = new Set(f.appearance_string_columns || []);
      this.strings = d.strings || {};
      this.aIndex = {}; (this.appNames || []).forEach((nm, i) => { this.aIndex[nm] = i; });
      this.eIndex = {}; f.entity.forEach((nm, i) => { this.eIndex[nm] = i; });
      // 关键帧下标
      this.kfIdx = [];
      for (let i = 0; i < this.N; i++) if (this.frames[i][C.kf]) this.kfIdx.push(i);
      if (!this.kfIdx.length || this.kfIdx[0] !== 0) this.kfIdx.unshift(0);
      // 房间/楼层/道具行的生效下标
      this.roomAt = d.rooms.map((r) => r.at);
      this.levelAt = d.levels.map((r) => r.at);
      // 每次进房期间有网格/门变化的帧
      this.roomChanges = d.rooms.map(() => []);
      for (let i = 0; i < this.N; i++) {
        const fr = this.frames[i];
        if (fr[C.g] || fr[C.dr] || fr[C.gs] || fr[C.ds]) {
          const ri = this.roomIndexAt(i);
          if (ri >= 0) this.roomChanges[ri].push(i);
        }
      }
      // 事件按 n 归帧
      this.eventsByN = new Map();
      for (const ev of d.events) {
        if (!this.eventsByN.has(ev.n)) this.eventsByN.set(ev.n, []);
        this.eventsByN.get(ev.n).push(ev);
      }
      // 输入：顺序过一遍，记每帧之前的状态（状态对象只在变化时替换，存引用即可）
      this.inputBefore = new Array(this.N);
      this.inputSummary = new Array(this.N); // 玩家 0：[u tick 按下位, 本帧任一 tick 按下位, 本帧任一 tick triggered 位]
      let st = {};
      for (let i = 0; i < this.N; i++) {
        this.inputBefore[i] = Object.assign({}, st);
        const ticks = expandTicks(this.frames[i][C.tk], st);
        let uBits = null, anyBits = 0, trig = 0;
        for (const t of ticks) {
          const a = t.inputs[0];
          if (!a) continue;
          let b = 0, tr = 0;
          for (let j = 0; j < 12; j++) { if (a[j] && a[j][1]) b |= 1 << j; if (a[j] && a[j][2]) tr |= 1 << j; }
          anyBits |= b; trig |= tr;
          if (t.kind === 'u') uBits = b;
        }
        if (uBits === null && ticks.length) {
          const a = ticks[ticks.length - 1].inputs[0];
          uBits = 0;
          if (a) for (let j = 0; j < 12; j++) if (a[j] && a[j][1]) uBits |= 1 << j;
        }
        this.inputSummary[i] = [uBits || 0, anyBits, trig];
      }
      this.segCache = new Map(); // 关键帧下标 -> {p: [...], e: [...]}
      this.segOrder = [];
      this.maxSegs = 24;
      this._roomCache = null;
    }

    roomIndexAt(i) { return bisectRight(this.roomAt, i) - 1; }
    levelIndexAt(i) { return bisectRight(this.levelAt, i) - 1; }

    // 关键帧段：从段首关键帧向后逐帧展开，只展开到被请求的帧（之后再请求更后的帧时接着展开），结果缓存。
    _segment(i) {
      const s = bisectRight(this.kfIdx, i) - 1;
      const k0 = this.kfIdx[s];
      let seg = this.segCache.get(k0);
      if (!seg) {
        seg = { p: [], e: [], pp: new Map(), pe: new Map(), pa: [], ea: [], ppa: new Map(), pea: new Map() };
        this.segCache.set(k0, seg);
        this.segOrder.push(k0);
        if (this.segOrder.length > this.maxSegs) this.segCache.delete(this.segOrder.shift());
      }
      for (let j = k0 + seg.p.length; j <= i; j++) {
        const fr = this.frames[j];
        let prows, erows;
        [prows, seg.pp] = expandRows(fr[C.p], j === k0 ? new Map() : seg.pp, this.pcolP);
        if (fr[C.e] !== null && fr[C.e] !== undefined) {
          [erows, seg.pe] = expandRows(fr[C.e], j === k0 ? new Map() : seg.pe, this.pcolE);
        } else erows = null; // IRR_OBS=keyframe 的非关键帧（pe 保留上一个有 e 的帧，与 load_run 相同）
        seg.p.push(prows); seg.e.push(erows);
        if (this.appNames) {
          let parows, earows;
          [parows, seg.ppa] = expandSparse(fr[C.pa], j === k0 ? new Map() : seg.ppa, prows.map((r) => r[0]));
          if (erows) [earows, seg.pea] = expandSparse(fr[C.ea], j === k0 ? new Map() : seg.pea, erows.map((r) => r[0]));
          else { earows = null; seg.pea = new Map(); }
          seg.pa.push(parows); seg.ea.push(earows);
        }
      }
      return [k0, seg];
    }

    playerRows(i) { const [k0, seg] = this._segment(i); return seg.p[i - k0]; }
    entityRows(i) { const [k0, seg] = this._segment(i); return seg.e[i - k0]; }
    // v2：与 playerRows/entityRows 同序的外观完整行（v1 文件返回 null）
    playerAppRows(i) { if (!this.appNames) return null; const [k0, seg] = this._segment(i); return seg.pa[i - k0]; }
    entityAppRows(i) { if (!this.appNames) return null; const [k0, seg] = this._segment(i); return seg.ea[i - k0]; }
    str(id) { return id === null || id === undefined ? null : (this.strings[id] !== undefined ? this.strings[id] : null); }
    decodeAppearance(row, player) {
      if (!row || !this.appNames) return null;
      const names = player ? this.appNames.concat(this.appPlayerExtra) : this.appNames;
      const o = {};
      names.forEach((nm, j) => {
        const v = row[j];
        if (this.afix.has(nm)) o[nm] = v === null ? null : v / 100;
        else if (this.astr.has(nm)) o[nm] = this.str(v);
        else o[nm] = v;
      });
      return o;
    }

    // 当前房间（grid/doors 已应用到第 i 帧的变化）。返回 {ri, room, grid: Map, doors: Map}
    roomState(i) {
      const ri = this.roomIndexAt(i);
      if (ri < 0) return { ri, room: null, grid: new Map(), doors: new Map(), gspr: null, dspr: null };
      const c = this._roomCache;
      let grid, doors, gspr, dspr, from;
      if (c && c.ri === ri && c.i <= i) { grid = c.grid; doors = c.doors; gspr = c.gspr; dspr = c.dspr; from = c.i + 1; }
      else {
        const room = this.d.rooms[ri];
        grid = new Map(room.grid.map((g) => [g[0], g]));
        doors = new Map(room.doors.map((d) => [d[0], d]));
        gspr = room.grid_spr ? new Map(room.grid_spr.map((g) => [g[0], g])) : null;   // v2
        dspr = room.door_spr ? new Map(room.door_spr.map((d) => [d[0], d])) : null;
        from = this.roomAt[ri];
      }
      for (const j of this.roomChanges[ri]) {
        if (j < from) continue;
        if (j > i) break;
        const fr = this.frames[j];
        for (const row of (fr[C.g] || [])) { if (row.length === 1) grid.delete(row[0]); else grid.set(row[0], row); }
        for (const row of (fr[C.dr] || [])) { if (row.length === 1) doors.delete(row[0]); else doors.set(row[0], row); }
        if (gspr) for (const row of (fr[C.gs] || [])) { if (row.length === 1) gspr.delete(row[0]); else gspr.set(row[0], row); }
        if (dspr) for (const row of (fr[C.ds] || [])) { if (row.length === 1) dspr.delete(row[0]); else dspr.set(row[0], row); }
      }
      // 缓存不能被后面倒退的调用改坏：倒退时上面重建了新 Map
      this._roomCache = { ri, i, grid, doors, gspr, dspr };
      return { ri, room: this.d.rooms[ri], grid, doors, gspr, dspr };
    }

    items(i) { // {玩家: {道具 id: 数量}}
      const out = {};
      for (const it of this.d.items) {
        if (it.at > i) break;
        const m = {};
        for (const [id, c] of it.items) m[id] = c;
        out[it.p] = m;
      }
      return out;
    }
    itemsLine(i, p) {
      let best = null;
      for (const it of this.d.items) { if (it.at > i) break; if (it.p === p) best = it; }
      return best;
    }

    level(i) { const li = this.levelIndexAt(i); return li >= 0 ? this.d.levels[li] : null; }
    events(i) { return this.eventsByN.get(this.frames[i][C.n]) || []; }
    ticks(i) { return expandTicks(this.frames[i][C.tk], Object.assign({}, this.inputBefore[i])); }

    decodePlayer(row) {
      const o = {};
      this.fields.player.forEach((nm, j) => { const v = row[j]; o[nm] = (this.pfix.has(nm) && v !== null) ? v / 100 : v; });
      return o;
    }
    decodeEntity(row) {
      const o = {};
      this.fields.entity.forEach((nm, j) => { const v = row[j]; o[nm] = (this.efix.has(nm) && v !== null) ? v / 100 : v; });
      const cat = entityCategory(o.type);
      o.category = cat;
      const names = cat ? this.fields.entity_extra[cat] : null;
      const tail = row.slice(N_COMMON);
      if (names && tail.length === names.length) {
        const fx = new Set(this.xfix[cat] || []);
        o.extra = {};
        names.forEach((nm, j) => { const v = tail[j]; o.extra[nm] = (fx.has(nm) && v !== null) ? v / 100 : v; });
      } else o.extra = null;
      return o;
    }

    // 校验/导出用：第 i 帧的完整状态（与 load_run.Run.frame(n) 的字段对应）
    frameState(i) {
      const fr = this.frames[i];
      const rs = this.roomState(i);
      const prows = this.playerRows(i), erows = this.entityRows(i);
      const lv = this.level(i);
      const rf = {};
      this.fields.room_frame.forEach((nm, j) => { rf[nm] = fr[C.r][j]; });
      const room = rs.room ? Object.assign({}, rs.room) : null;
      const fx = (x) => (x === null ? null : x / 100);
      if (room) { delete room.at; room.top_left = room.top_left.map(fx); room.bottom_right = room.bottom_right.map(fx); }
      const pa = this.playerAppRows(i), ea = this.entityAppRows(i);
      return {
        n: fr[C.n], gf: fr[C.gf], t: fr[C.t], keyframe: !!fr[C.kf], mt: fr[C.mt], digest: fr[C.dg],
        room_frame: rf, player_rows: prows, entity_rows: erows,
        players: prows.map((r) => this.decodePlayer(r)),
        entities: erows === null ? null : erows.map((r) => this.decodeEntity(r)),
        room_head: room ? Object.fromEntries(Object.entries(room).filter(([k]) => !['grid', 'doors', 'grid_spr', 'door_spr'].includes(k))) : null,
        pa_rows: pa, ea_rows: ea,
        player_appearance: pa ? pa.map((r) => this.decodeAppearance(r, true)) : null,
        entity_appearance: ea ? ea.map((r) => this.decodeAppearance(r, false)) : null,
        grid_spr: rs.gspr ? Array.from(rs.gspr.values()) : null,
        door_spr: rs.dspr ? Array.from(rs.dspr.values()) : null,
        grid: rs.room ? Array.from(rs.grid.values()) : null,
        doors: rs.room ? Array.from(rs.doors.values()) : null,
        level_n: lv ? lv.n : null, level_rooms: lv ? lv.rooms : null,
        items: this.items(i), events: this.events(i),
        ticks: this.ticks(i),
        mouse: (fr[C.ms] || []).map((m) => [m[0]].concat(m.slice(1, 5).map((x) => (x === null ? null : x / 100)), [m[5]])),
      };
    }
  }

  const api = { RunData, expandRows, expandSparse, expandTicks, entityCategory, C, N_COMMON };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.IsaacRunCore = api;
})(typeof self !== 'undefined' ? self : this);
