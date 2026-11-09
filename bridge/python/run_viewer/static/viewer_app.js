/* 录制查看器界面：房间俯视图（canvas）、可拖动的全局/房间进度条、跳转列表、当帧数值面板。
 * 数据与展开见 viewer_core.js；坐标系见 REPORT.md「坐标系」：网格第 c 列第 r 行的格子覆盖
 * x ∈ [20+40c, 60+40c]、y ∈ [100+40r, 140+40r]（Isaac 房间像素坐标），实体位置直接是这套坐标。
 */
(function () {
  'use strict';
  const core = window.IsaacRunCore;
  const C = core.C;
  const $ = (id) => document.getElementById(id);
  const RV = (window.__rv = { timing: {}, ready: false });

  // ------------------------------------------------------------------ 常量表
  const ROOM_TYPES = {
    1: '普通', 2: '商店', 3: '错误房', 4: '宝箱房', 5: 'Boss', 6: '小 Boss', 7: '隐藏房', 8: '超级隐藏', 9: '街机',
    10: '诅咒房', 11: '挑战房', 12: '图书馆', 13: '献祭房', 14: '恶魔房', 15: '天使房', 16: '地下通道', 17: 'Boss Rush',
    18: '床房', 19: '贫瘠房', 20: '箱子房', 21: '骰子房', 22: '黑市', 23: '贪婪出口', 24: '星象房', 25: '传送房',
    26: '传送出口', 27: '隐藏出口', 28: '蓝子宫', 29: '究极隐藏',
  };
  const ROOM_COLORS = {
    1: '#6b6f7a', 2: '#3fa66b', 3: '#888', 4: '#e8c547', 5: '#d9443a', 6: '#d9803a', 7: '#8a7bd1', 8: '#a58be6',
    9: '#4fb3c9', 10: '#9b3d6a', 11: '#c95f3a', 12: '#b58a5a', 13: '#a83f3f', 14: '#7a1c1c', 15: '#e8e3c8',
    16: '#444', 17: '#ff7d6b', 18: '#c9a0d6', 19: '#c9a0d6', 20: '#c9a64a', 21: '#e05fa0', 22: '#555', 24: '#5a7be0',
    25: '#4fc9a0', 27: '#5a5a5a', 28: '#4f8fe0', 29: '#c94fe0',
  };
  const SHAPES = { 1: '1x1', 2: 'IH 窄横', 3: 'IV 窄竖', 4: '1x2', 5: 'IIV 长竖', 6: '2x1', 7: 'IIH 长横', 8: '2x2',
    9: 'L 缺左上', 10: 'L 缺右上', 11: 'L 缺左下', 12: 'L 缺右下' };
  const ACTIONS = ['←', '→', '↑', '↓', '射←', '射→', '射↑', '射↓', '炸弹', '主动', '卡/药', '丢弃'];
  const ACTION_NAMES = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'SHOOTLEFT', 'SHOOTRIGHT', 'SHOOTUP', 'SHOOTDOWN', 'BOMB', 'ITEM', 'PILLCARD', 'DROP'];
  const DIRS = { '-1': '无', 0: '左', 1: '上', 2: '右', 3: '下' };
  const GRID_NAMES = { 0: '空', 1: '装饰', 2: '石头', 3: '金属块', 4: '标记石', 5: '炸弹石', 6: '罐/蘑菇', 7: '坑', 8: '尖刺',
    9: '开关尖刺', 10: '蛛网', 11: '钥匙块', 12: 'TNT', 13: '火堆', 14: '便便', 15: '墙', 16: '门', 17: '活板门', 18: '楼梯',
    19: '重力', 20: '压力板', 21: '雕像', 22: '超级石', 23: '传送', 24: '柱', 25: '尖刺石', 26: '罐 2', 27: '金石' };
  const PICKUP = { 10: '心', 20: '硬币', 30: '钥匙', 40: '炸弹', 41: '可投掷炸弹', 42: '便便', 50: '箱子', 51: '炸弹箱',
    52: '尖刺箱', 53: '永恒箱', 54: '拟态箱', 55: '旧箱', 56: '木箱', 57: '巨箱', 58: '鬼箱', 60: '锁箱', 69: '福袋', 70: '药丸',
    90: '电池', 100: '道具', 110: '铲子', 150: '商品', 300: '卡', 340: '大箱', 350: '饰品', 360: '红箱', 370: '奖杯', 380: '床', 390: '妈妈箱' };
  const STAGE_COLORS = ['#7a5c3e', '#5e6b3b', '#3c5a6e', '#6e3c3c', '#6e3c5e', '#3c3c6e', '#5e5e5e', '#3c6e5e', '#6e5e3c', '#4a4a4a', '#6e6e3c', '#3c6e6e', '#5a3c6e', '#2e2e2e'];

  // ------------------------------------------------------------------ 状态
  let run, D, N;
  let cur = 0, playing = false, speed = 1, acc = 0, lastTs = 0;
  let opt = { effects: true, labels: true, vel: false, gridIdx: false, invisible: true, hud: true };
  let selected = null; // {idx, seed}
  let needRender = true;
  let view = { s: 1, ox: 0, oy: 0, bbox: null };
  let hoverEnt = null;
  let names = { entities: {}, items: {}, trinkets: {}, cards: {}, pills: {}, icons: {} };
  let PI, EI, XI = {};
  let roomVisits = [];  // {ri, at, end, level}
  let markers = [];     // {i, kind, label, color}
  let tlCache = null;
  const iconImgs = {};
  // 视图模式：geo = 几何（原来的画法）、art = 真实美术、video = 实录画面、split = 美术 + 实录并排
  let mode = 'geo';
  let art = null;          // IsaacArt.ArtRenderer（有素材时）
  let footage = null;      // {url, fps, first_frame, frames, width, height}
  // 自由视角：zoom 相对"整房适配"；pan = 相对房间中心的世界坐标偏移；follow = 跟随玩家
  const cam = { zoom: 1, panX: 0, panY: 0, follow: false, game: false };
  const IsaacArtPX = () => (window.IsaacArt ? window.IsaacArt.PX : 0.65);
  let dragPan = null, suppressClick = false;
  let videoDriven = false; // 播放时帧号由 <video> 的时间驱动

  // ------------------------------------------------------------------ 工具
  const fmtTime = (frames) => {
    const s = frames / 30, h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = (s % 60).toFixed(2);
    return (h ? h + ':' + String(m).padStart(2, '0') : m) + ':' + ss.padStart(5, '0');
  };
  const fx = (v) => (v === null || v === undefined ? '—' : (v / 100).toFixed(2));
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  function entName(t, v, s) {
    const e = names.entities[`${t}.${v}.${s}`] || names.entities[`${t}.${v}`] || names.entities[`${t}`];
    return e ? e[0] : null;
  }
  function entRadius(t, v, s) {
    const e = names.entities[`${t}.${v}.${s}`] || names.entities[`${t}.${v}`] || names.entities[`${t}`];
    return e && e[1] > 0 ? e[1] : null;
  }
  const itemName = (id) => (id > 0 ? (names.items[id] || `道具 ${id}`) : '—');
  const trinketName = (id) => {
    if (!id) return '—';
    const gold = id & 0x8000, base = id & 0x7fff;
    return (names.trinkets[base] || `饰品 ${base}`) + (gold ? '（金）' : '');
  };
  const cardName = (id) => (id ? (names.cards[id] || `卡 ${id}`) : '—');
  const pillName = (id) => (id >= 0 && id !== null ? (names.pills[id] || `药效 ${id}`) : '—');
  function iconImg(key) {
    if (!names.icons[key]) return null;
    if (!iconImgs[key]) { const im = new Image(); im.src = names.icons[key]; im.onload = () => { needRender = true; }; iconImgs[key] = im; }
    return iconImgs[key];
  }
  function popcount(x) { let c = 0; x = x >>> 0; while (x) { c += x & 1; x >>>= 1; } return c; }
  function eventText(ev) {
    const a = ev.a || [];
    if (ev.k === 'hcmd') return `控制台(${ev.tag}) ${ev.cmd}`;
    if (ev.k === 'mark') return `标记 ${ev.label}`;
    if (ev.k === 'api') {
      // 第三方 mod 的效果调用（recorder 0.4.0）：参数里 {"p":k} = 玩家 k，{"f":x} = 浮点，{"ref":...} = EntityRef
      const fmt = (v) => (v && typeof v === 'object') ? (v.p !== undefined ? `玩家${v.p}` : v.f !== undefined ? String(v.f)
        : v.ref !== undefined ? `ref(${v.ref && v.ref.p !== undefined ? '玩家' + v.ref.p : JSON.stringify(v.ref)})` : JSON.stringify(v)) : JSON.stringify(v);
      const m = String(ev.m || '');
      const args = (ev.a || []).map(fmt).join(', ');
      return `mod(${ev.src || '?'}) ${ev.set ? m + ' = ' + args : m + '(' + args + ')'}${ev.rp ? '（回放执行）' : ''}`;
    }
    switch (ev.ev) {
      case 'damage': {
        const src = a[3] ? (entName(a[3], a[4], 0) || `${a[3]}.${a[4]}`) : '无来源/环境';
        return `受伤 ${(a[1] / 100).toFixed(1)}（${src}，flags ${a[2]}）`;
      }
      case 'use_item': return `使用 ${itemName(a[0])}`;
      case 'use_card': return `用卡 ${cardName(a[0])}`;
      case 'use_pill': return `吃药 ${pillName(a[0])}`;
      case 'pause': return '暂停';
      case 'unpause': return '继续';
      case 'game_end': return a[0] ? '游戏结束（死亡）' : '游戏结束（胜利）';
      case 'exit': return `退出${a[0] ? '（存档）' : ''}`;
      case 'cmd': return `命令 ${a.join(' ')}`;
      default: return `${ev.ev} ${JSON.stringify(a)}`;
    }
  }
  const API_QUIET = new Set(['EntityPlayer.SetShootingCooldown', 'EntityPlayer.AddControlsCooldown', 'SFXManager.Play']);
  const EV_COLORS = { api: '#26c6da', damage: '#ff4d4d', use_item: '#4fc3f7', use_card: '#c77dff', use_pill: '#7bd88f', game_end: '#ffffff', exit: '#ffffff', cmd: '#ffb74d', hcmd: '#ffb74d', mark: '#ffb74d' };

  // ------------------------------------------------------------------ 房间几何
  // 门槽 -> 网格坐标（由 human1/机器人录制里门槽与门格的对应关系核对过，见 REPORT.md）
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
  function layout(room) {
    const cv = $('room');
    const W = cv.clientWidth, H = cv.clientHeight;
    const gw = room ? room.gw : 15, gh = room ? room.gh : 9;
    const x0 = 20, y0 = 100, x1 = 20 + 40 * gw, y1 = 100 + 40 * gh;
    const pad = mode === 'geo' ? 46 : 8; // 几何模式留出门外侧的目标房间标签
    const ex = mode === 'geo' ? 0 : 60;  // 美术模式把墙外沿（1.5 格）也放进视野
    // 画布刚从隐藏切回或窗格极窄时 W/H 可能为 0 或小于 2*pad，比例会变成负数，roundRect 会抛错、渲染循环中断；夹到一个小正数
    const s0 = Math.max(0.02, Math.min((W - 2 * pad) / (x1 - x0 + 2 * ex), (H - 2 * pad) / (y1 - y0 + 2 * ex)));
    const s = s0 * cam.zoom;
    let cx = (x0 + x1) / 2 + cam.panX, cy = (y0 + y1) / 2 + cam.panY;
    if (cam.follow && run) {
      const pr = run.playerRows(cur)[0];
      if (pr && pr[PI.x] !== null) { cx = pr[PI.x] / 100; cy = pr[PI.y] / 100; }
    }
    if (cam.game && run) {
      // 游戏镜头（C）：视野 = 游戏 480x270 画面 = 738x415 世界单位；跟随玩家、夹在房间（含 1 格外沿）以内（估计游戏的镜头规则）
      const Vw = 480 / IsaacArtPX(), Vh = 270 / IsaacArtPX();
      const sg = Math.min(W / Vw, H / Vh);
      const pr = run.playerRows(cur)[0];
      let px = (x0 + x1) / 2, py = (y0 + y1) / 2;
      if (pr && pr[PI.x] !== null) { px = pr[PI.x] / 100; py = pr[PI.y] / 100; }
      const clamp = (v, a, b) => (a > b ? (a + b) / 2 : Math.max(a, Math.min(b, v)));
      const gx = clamp(px, x0 - 40 + Vw / 2, x1 + 40 - Vw / 2), gy = clamp(py, y0 - 40 + Vh / 2, y1 + 40 - Vh / 2);
      view = { s: sg, s0: sg, ox: W / 2 - gx * sg, oy: H / 2 - gy * sg, bbox: [x0, y0, x1, y1] };
      return;
    }
    view = { s, s0, ox: W / 2 - cx * s, oy: H / 2 - cy * s, bbox: [x0, y0, x1, y1] };
  }
  const WX = (x) => view.ox + x * view.s, WY = (y) => view.oy + y * view.s;

  // ------------------------------------------------------------------ 画房间
  function drawGridCell(ctx, g, c, r, room) {
    const [idx, type, variant, state, coll] = g;
    const x = WX(20 + 40 * c), y = WY(100 + 40 * r), w = 40 * view.s;
    const inset = w * 0.08;
    const dead = coll === 0 && [2, 3, 4, 5, 6, 11, 12, 13, 14, 22, 24, 25, 26, 27].includes(type);
    ctx.save();
    if (dead) ctx.globalAlpha = 0.28;
    switch (type) {
      case 15: ctx.fillStyle = '#2a1f19'; ctx.fillRect(x, y, w + 0.5, w + 0.5); break;
      case 16: ctx.fillStyle = '#2a1f19'; ctx.fillRect(x, y, w + 0.5, w + 0.5); break; // 门在 drawDoors 里画
      case 7: ctx.fillStyle = coll === 0 ? '#3a2f28' : '#050505'; ctx.fillRect(x + 1, y + 1, w - 2, w - 2); break;
      case 8: case 9: {
        ctx.fillStyle = '#5a5552'; ctx.fillRect(x + inset, y + inset, w - 2 * inset, w - 2 * inset);
        const on = type === 8 || state === 0 || coll !== 0;
        ctx.fillStyle = on ? '#d8d8d8' : '#777';
        for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) {
          const cx = x + w * (0.22 + 0.28 * k), cy = y + w * (0.22 + 0.28 * j), q = w * 0.08;
          ctx.beginPath(); ctx.moveTo(cx - q, cy + q); ctx.lineTo(cx, cy - q * 1.4); ctx.lineTo(cx + q, cy + q); ctx.fill();
        }
        break;
      }
      case 10: {
        ctx.strokeStyle = 'rgba(240,240,240,0.7)'; ctx.lineWidth = Math.max(1, view.s);
        ctx.beginPath();
        ctx.moveTo(x + inset, y + inset); ctx.lineTo(x + w - inset, y + w - inset);
        ctx.moveTo(x + w - inset, y + inset); ctx.lineTo(x + inset, y + w - inset);
        ctx.moveTo(x + w / 2, y + inset); ctx.lineTo(x + w / 2, y + w - inset);
        ctx.moveTo(x + inset, y + w / 2); ctx.lineTo(x + w - inset, y + w / 2);
        ctx.stroke(); break;
      }
      case 17: case 18: {
        ctx.fillStyle = '#000'; ctx.beginPath(); ctx.arc(x + w / 2, y + w / 2, w * 0.42, 0, 7); ctx.fill();
        ctx.strokeStyle = type === 17 ? '#c9a64a' : '#888'; ctx.lineWidth = 2; ctx.stroke(); break;
      }
      case 20: ctx.fillStyle = state ? '#4a7' : '#999'; ctx.fillRect(x + w * 0.25, y + w * 0.25, w * 0.5, w * 0.5); break;
      case 1: break;
      default: {
        const col = { 2: '#8d8378', 3: '#5d6a78', 4: '#8d8378', 5: '#8d5a40', 6: '#a39066', 11: '#c9a64a', 12: '#b83a2e',
          13: variant === 1 ? '#d33' : (variant === 2 ? '#4a7ad3' : (variant === 3 ? '#a04ad3' : '#e2701f')),
          14: ({ 1: '#a33', 2: '#c9b05a', 3: '#e8c547', 4: '#e05fa0', 5: '#222', 6: '#eee' })[variant] || '#7a4b22',
          21: '#999', 22: '#7a8aa0', 23: '#4fc9a0', 24: '#666', 25: '#8d6f6f', 26: '#a39066', 27: '#e8c547' }[type] || '#a0a';
        ctx.fillStyle = col;
        const rr = Math.max(0, w * 0.18);
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(x + inset, y + inset, w - 2 * inset, w - 2 * inset, rr) : ctx.rect(x + inset, y + inset, w - 2 * inset, w - 2 * inset);
        ctx.fill();
        if (type === 4) { // 标记石：蓝 X
          ctx.strokeStyle = '#4fa3ff'; ctx.lineWidth = 2; ctx.beginPath();
          ctx.moveTo(x + w * 0.3, y + w * 0.3); ctx.lineTo(x + w * 0.7, y + w * 0.7); ctx.moveTo(x + w * 0.7, y + w * 0.3); ctx.lineTo(x + w * 0.3, y + w * 0.7); ctx.stroke();
        }
        if (type === 14 && !dead && state > 0) { // 便便受损程度（state 0..1000）
          ctx.fillStyle = 'rgba(0,0,0,0.45)'; ctx.fillRect(x + inset, y + w - inset - (w - 2 * inset) * Math.min(1, state / 1000), w - 2 * inset, (w - 2 * inset) * Math.min(1, state / 1000));
        }
        if (type === 12 || type === 11) {
          ctx.fillStyle = '#fff'; ctx.font = `bold ${Math.max(8, w * 0.28)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillText(type === 12 ? 'TNT' : '锁', x + w / 2, y + w / 2);
        }
      }
    }
    ctx.restore();
    if (opt.gridIdx) {
      ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.font = `${Math.max(7, w * 0.22)}px monospace`; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(idx, x + 2, y + 2);
    }
  }

  function drawRoom(ctx, rs) {
    const room = rs.room;
    const W = ctx.canvas.clientWidth, H = ctx.canvas.clientHeight;
    ctx.fillStyle = '#0d0d10'; ctx.fillRect(0, 0, W, H);
    if (!room) return;
    const gw = room.gw, gh = room.gh, shape = room.shape;
    // 地板：top_left..bottom_right 内部，L 形扣掉缺角
    const [tlx, tly] = room.top_left.map((v) => v / 100), [brx, bry] = room.bottom_right.map((v) => v / 100);
    for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) {
      const x = 20 + 40 * c, y = 100 + 40 * r;
      if (x < tlx || y < tly || x + 40 > brx || y + 40 > bry || isVoid(shape, c, r)) continue;
      ctx.fillStyle = (c + r) % 2 ? '#4b3a2e' : '#47372b';
      ctx.fillRect(WX(x), WY(y), 40 * view.s + 0.5, 40 * view.s + 0.5);
    }
    for (const g of rs.grid.values()) drawGridCell(ctx, g, g[0] % gw, Math.floor(g[0] / gw), room);
    // 门
    for (const d of rs.doors.values()) {
      const [slot, target, ttype, open, locked] = d;
      const ci = doorCell(shape, slot, gw, gh);
      const c = ci % gw, r = Math.floor(ci / gw);
      const x = WX(20 + 40 * c), y = WY(100 + 40 * r), w = 40 * view.s;
      const col = ROOM_COLORS[ttype] || '#999';
      const side = slot % 4; // 0 左 1 上 2 右 3 下
      ctx.fillStyle = open ? '#0b0b0b' : col;
      ctx.strokeStyle = col; ctx.lineWidth = Math.max(2, 3 * view.s);
      const a = w * 0.18;
      ctx.beginPath(); ctx.rect(x + a, y + a, w - 2 * a, w - 2 * a); ctx.fill(); ctx.stroke();
      if (!open) { // 关门：画门闩
        ctx.strokeStyle = '#111'; ctx.lineWidth = Math.max(1.5, 2 * view.s); ctx.beginPath();
        if (side === 0 || side === 2) { ctx.moveTo(x + w / 2, y + a); ctx.lineTo(x + w / 2, y + w - a); }
        else { ctx.moveTo(x + a, y + w / 2); ctx.lineTo(x + w - a, y + w / 2); }
        ctx.stroke();
      }
      if (locked) { ctx.fillStyle = '#ffd54f'; ctx.font = `bold ${Math.max(9, w * 0.32)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('🔒', x + w / 2, y + w / 2); }
      if (opt.labels) {
        ctx.fillStyle = '#ddd'; ctx.font = `${Math.max(8, w * 0.22)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        const lx = x + w / 2 + (side === 0 ? -w * 0.9 : side === 2 ? w * 0.9 : 0), ly = y + w / 2 + (side === 1 ? -w * 0.65 : side === 3 ? w * 0.65 : 0);
        ctx.fillText((ROOM_TYPES[ttype] || ttype) + (target >= 0 ? '' : ` ${target}`), lx, ly);
      }
    }
  }

  // ------------------------------------------------------------------ 画实体
  function catOrder(t) { return t === 1000 ? 0 : t === 5 ? 1 : t === 6 ? 1 : t === 4 ? 2 : (t >= 10 && t < 1000) ? 3 : t === 3 ? 4 : t === 7 ? 5 : 6; }
  function entGeom(row) { // 返回 {x, y, r, dy}（dy = 泪弹高度偏移）
    const t = row[1];
    const x = row[4] / 100, y = row[5] / 100;
    let r = entRadius(t, row[2], row[3]) || 10, dy = 0;
    if (t === 2 || t === 9) {
      const xn = XI[t === 2 ? 'tear' : 'projectile'];
      const sc = row[xn.scale] != null ? row[xn.scale] / 100 : 1;
      r = Math.max(2.5, 6.5 * sc);
      dy = row[xn.height] != null ? row[xn.height] / 100 : 0;
    } else if (t === 1000) r = Math.min(r || 4, 8);
    return { x, y, r, dy };
  }
  function drawEntities(ctx, erows) {
    if (!erows) return [];
    const list = erows.slice().sort((a, b) => catOrder(a[1]) - catOrder(b[1]));
    const hits = [];
    const s = view.s;
    for (const row of list) {
      const t = row[1], v = row[2], st = row[3];
      if (row[4] === null || row[5] === null) continue;
      if (t === 1000 && !opt.effects) continue;
      const vis = row[EI.visible];
      if (!vis && !opt.invisible) continue;
      const g = entGeom(row);
      const X = WX(g.x), Y = WY(g.y), R = g.r * s;
      ctx.save();
      if (!vis) ctx.setLineDash([3, 3]);
      const isSel = selected && selected.idx === row[0] && selected.seed === row[EI.init_seed];
      if (t === 1000) {
        ctx.globalAlpha = vis ? 0.35 : 0.2;
        ctx.fillStyle = '#9aa'; ctx.beginPath(); ctx.arc(X, Y, Math.max(1.5, R * 0.6), 0, 7); ctx.fill();
      } else if (t === 2 || t === 9) {
        ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.beginPath(); ctx.ellipse(X, Y, R * 0.8, R * 0.4, 0, 0, 7); ctx.fill();
        ctx.fillStyle = t === 2 ? '#8fd3ff' : '#ff3d6b';
        ctx.strokeStyle = t === 2 ? '#1d6fa5' : '#7a0f2a'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(X, Y + g.dy * s, R, 0, 7); ctx.fill(); ctx.stroke();
      } else if (t === 7) { // 激光：从位置沿角度画到 max_distance（0 = 到墙，画到房间外）
        const xn = XI.laser;
        const ang = (row[xn.angle] || 0) / 100 * Math.PI / 180;
        let len = (row[xn.max_distance] || 0) / 100; if (!len) len = 1200;
        const rad = Math.max(3, ((row[xn.radius] || 0) / 100 || 8));
        ctx.strokeStyle = 'rgba(255,60,60,0.55)'; ctx.lineWidth = rad * s; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X + Math.cos(ang) * len * s, Y + Math.sin(ang) * len * s); ctx.stroke();
      } else if (t === 5) {
        drawPickup(ctx, row, X, Y, s);
      } else if (t === 4) {
        ctx.fillStyle = '#1b1b1b'; ctx.strokeStyle = '#ff9a3c'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X, Y, Math.max(4, 10 * s), 0, 7); ctx.fill(); ctx.stroke();
      } else if (t === 3) {
        ctx.fillStyle = '#5fbf5f'; ctx.strokeStyle = '#1f5f1f'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X, Y, Math.max(3, R), 0, 7); ctx.fill(); ctx.stroke();
      } else if (t === 6) {
        ctx.fillStyle = '#9b59b6'; ctx.fillRect(X - 14 * s, Y - 18 * s, 28 * s, 26 * s);
      } else if (t === 8) {
        ctx.fillStyle = '#ccc'; ctx.beginPath(); ctx.arc(X, Y, Math.max(3, 6 * s), 0, 7); ctx.fill();
      } else { // NPC
        const xn = XI.npc;
        const boss = row[xn.boss] === 1 || row[xn.boss] === true;
        const champ = row[xn.champion];
        const dead = row[EI.hp] !== null && row[EI.hp] <= 0;
        ctx.globalAlpha = dead ? 0.35 : (row[EI.coll] === 0 ? 0.6 : 0.92);
        ctx.fillStyle = boss ? '#b3261e' : (t === 33 ? '#e2701f' : '#e06650');
        ctx.strokeStyle = champ !== null && champ >= 0 ? '#ffe066' : (boss ? '#ffb4ab' : '#5a1810');
        ctx.lineWidth = champ !== null && champ >= 0 ? 3 : (boss ? 2.5 : 1.5);
        ctx.beginPath(); ctx.arc(X, Y, Math.max(3, R), 0, 7); ctx.fill(); ctx.stroke();
        const hp = row[EI.hp], mhp = row[EI.max_hp];
        if (mhp > 0 && hp !== null && (hp < mhp || boss) && !dead) {
          const bw = Math.max(18, R * 2), bh = 3;
          ctx.globalAlpha = 1;
          ctx.fillStyle = '#300'; ctx.fillRect(X - bw / 2, Y - R - 7, bw, bh);
          ctx.fillStyle = '#e33'; ctx.fillRect(X - bw / 2, Y - R - 7, bw * Math.max(0, Math.min(1, hp / mhp)), bh);
        }
      }
      ctx.restore();
      if (opt.vel && (row[6] || row[7])) arrow(ctx, X, Y, X + row[6] / 100 * s * 6, Y + row[7] / 100 * s * 6, '#7fffd4');
      if (opt.labels && t !== 1000 && t !== 2 && t !== 9 && t !== 7) {
        const nm = t === 5 ? pickupLabel(row) : (entName(t, v, st) || `${t}.${v}`);
        if (nm) {
          ctx.font = '11px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
          ctx.fillStyle = 'rgba(0,0,0,0.6)'; const tw = ctx.measureText(nm).width;
          const ly = Y - Math.max(R, 8) - (t >= 10 && t < 1000 ? 9 : 3);
          ctx.fillRect(X - tw / 2 - 2, ly - 12, tw + 4, 13);
          ctx.fillStyle = t >= 10 && t < 1000 ? '#ffd0c8' : '#e8e8e8'; ctx.fillText(nm, X, ly);
        }
      }
      if (isSel) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.arc(X, Y + g.dy * s, Math.max(R, 6) + 5, 0, 7); ctx.stroke(); ctx.setLineDash([]); }
      hits.push({ row, X, Y: Y + g.dy * s, R: Math.max(R, 5) });
    }
    return hits;
  }
  function pickupLabel(row) {
    const v = row[2], st = row[3];
    const price = row[XI.pickup.price];
    let s;
    if (v === 100) s = st ? itemName(st) : '空台';
    else if (v === 350) s = trinketName(st);
    else if (v === 300) s = cardName(st);
    else if (v === 70) s = `药丸 ${st}`;
    else s = PICKUP[v] || `拾取 ${v}`;
    if (price) s += price > 0 ? ` $${price}` : ` (${price})`;
    return s;
  }
  function drawPickup(ctx, row, X, Y, s) {
    const v = row[2], st = row[3];
    const r = 9 * s;
    if (v === 100) {
      ctx.fillStyle = '#6d6a66'; ctx.fillRect(X - 14 * s, Y - 4 * s, 28 * s, 14 * s);
      const im = st ? iconImg('c' + st) : null;
      if (im && im.complete && im.naturalWidth) { ctx.imageSmoothingEnabled = false; ctx.drawImage(im, X - 16 * s, Y - 34 * s, 32 * s, 32 * s); }
      else if (st) { ctx.fillStyle = '#e8c547'; ctx.fillRect(X - 10 * s, Y - 28 * s, 20 * s, 20 * s); ctx.fillStyle = '#000'; ctx.font = `${Math.max(8, 9 * s)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(st, X, Y - 18 * s); }
      return;
    }
    if (v === 350) {
      const im = iconImg('t' + (st & 0x7fff));
      if (im && im.complete && im.naturalWidth) { ctx.imageSmoothingEnabled = false; ctx.drawImage(im, X - 14 * s, Y - 14 * s, 28 * s, 28 * s); return; }
    }
    const col = { 10: '#e53935', 20: '#f4c430', 30: '#b0b0b0', 40: '#333', 41: '#333', 42: '#7a4b22', 69: '#8d6e63', 70: '#f5f5f5', 90: '#ffeb3b', 300: '#d7ccc8', 350: '#bcaaa4' }[v]
      || (v >= 50 && v <= 60 || v === 340 || v === 360 || v === 390 ? '#a0522d' : '#bbb');
    ctx.fillStyle = col; ctx.strokeStyle = '#111'; ctx.lineWidth = 1;
    if (v === 300) { ctx.fillRect(X - 6 * s, Y - 8 * s, 12 * s, 16 * s); ctx.strokeRect(X - 6 * s, Y - 8 * s, 12 * s, 16 * s); }
    else if (v >= 50 && v <= 60 || v === 340 || v === 360 || v === 390) { ctx.fillRect(X - 11 * s, Y - 8 * s, 22 * s, 16 * s); ctx.strokeRect(X - 11 * s, Y - 8 * s, 22 * s, 16 * s); }
    else if (v === 10) { heart(ctx, X, Y, r, col); }
    else { ctx.beginPath(); ctx.arc(X, Y, Math.max(3, (v === 20 ? 6 : 7) * s), 0, 7); ctx.fill(); ctx.stroke(); }
  }
  function heart(ctx, x, y, r, col) {
    ctx.fillStyle = col; ctx.beginPath();
    ctx.moveTo(x, y + r * 0.8);
    ctx.bezierCurveTo(x - r * 1.2, y - r * 0.1, x - r * 0.6, y - r, x, y - r * 0.35);
    ctx.bezierCurveTo(x + r * 0.6, y - r, x + r * 1.2, y - r * 0.1, x, y + r * 0.8);
    ctx.fill();
  }
  function arrow(ctx, x0, y0, x1, y1, col) {
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0), h = 6;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 - h * Math.cos(a - 0.5), y1 - h * Math.sin(a - 0.5)); ctx.lineTo(x1 - h * Math.cos(a + 0.5), y1 - h * Math.sin(a + 0.5)); ctx.fill();
  }
  function drawPlayers(ctx, prows) {
    const s = view.s, hits = [];
    for (const row of prows) {
      const x = row[PI.x], y = row[PI.y];
      if (x === null || y === null) continue;
      const X = WX(x / 100), Y = WY(y / 100), R = Math.max(5, (row[PI.size] || 1000) / 100 * s);
      ctx.save();
      const inv = row[PI.damage_cooldown] > 0;
      ctx.globalAlpha = inv && (cur % 4 < 2) ? 0.45 : 1;
      ctx.fillStyle = '#f2d0b0'; ctx.strokeStyle = '#3fa9f5'; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.arc(X, Y, R, 0, 7); ctx.fill(); ctx.stroke();
      ctx.globalAlpha = 1;
      // 射击方向（fire_dir）与瞄准向量
      const fd = row[PI.fire_dir];
      if (fd !== null && fd >= 0) {
        const d = [[-1, 0], [0, -1], [1, 0], [0, 1]][fd];
        arrow(ctx, X, Y, X + d[0] * (R + 16 * s), Y + d[1] * (R + 16 * s), '#ffeb3b');
      }
      if (opt.vel && (row[PI.vx] || row[PI.vy])) arrow(ctx, X, Y, X + row[PI.vx] / 100 * s * 6, Y + row[PI.vy] / 100 * s * 6, '#7fffd4');
      ctx.fillStyle = '#123'; ctx.font = `bold ${Math.max(9, R)}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(prows.length > 1 ? 'P' + row[0] : 'P', X, Y + 1);
      ctx.restore();
      hits.push({ player: true, row, X, Y, R });
    }
    return hits;
  }

  let hitList = [];
  function renderRoom() {
    const cv = $('room'), ctx = cv.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(cv.clientWidth * dpr) || cv.height !== Math.round(cv.clientHeight * dpr)) {
      cv.width = Math.round(cv.clientWidth * dpr); cv.height = Math.round(cv.clientHeight * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (mode === 'art' || mode === 'split') return renderArt(cv, ctx, dpr);
    const q0 = performance.now();
    const rs = run.roomState(cur);
    layout(rs.room);
    drawRoom(ctx, rs);
    const q1 = performance.now();
    const erows = run.entityRows(cur);
    const q2 = performance.now();
    const h1 = drawEntities(ctx, erows);
    const h2 = drawPlayers(ctx, run.playerRows(cur));
    RV.roomParts = [q1 - q0, q2 - q1, performance.now() - q2, erows ? erows.length : 0];
    hitList = h1.concat(h2);
    if (!erows) { ctx.fillStyle = '#fc6'; ctx.font = '13px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('本帧没有实体行（keyframe 观测模式）', 10, 10); }
    return rs;
  }

  // ------------------------------------------------------------------ 真实美术模式
  function renderArt(cv, ctx, dpr) {
    const q0 = performance.now();
    const rs = run.roomState(cur);
    layout(rs.room);
    const W = cv.clientWidth, H = cv.clientHeight;
    ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
    const erows = run.entityRows(cur), prows = run.playerRows(cur);
    if (!art || !rs.room) { RV.roomParts = [0, 0, 0, 0]; hitList = []; return rs; }
    const k = view.s / IsaacArt.PX;   // 屏幕像素 / 精灵像素
    const M = { at: (x, y) => { ctx.save(); ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * (view.ox + x * view.s), dpr * (view.oy + y * view.s)); ctx.imageSmoothingEnabled = false; } };
    const visited = new Set();
    const lv = run.level(cur);
    if (lv) for (const v of roomVisits) { if (v.at > cur) break; if (v.at >= lv.at && v.room.desc) visited.add(v.room.desc[0]); }
    const F = { W: cv.clientWidth, H: cv.clientHeight, rs, erows, prows, eapp: run.entityAppRows(cur), papp: run.playerAppRows(cur), items: run.items(cur), EI, XI, effects: opt.effects, level: lv, visited };
    const q1 = performance.now();
    art.drawFrame(ctx, F, M);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // 命中表（悬停/点选用，与几何模式相同的语义）
    const hits = [];
    const ea = F.eapp || [];
    (erows || []).forEach((row, j) => {
      if (row[4] === null || row[5] === null) return;
      if (row[1] === 1000 && !opt.effects) return;
      const g = entGeom(row);
      let dy = g.dy;
      if (ea[j]) { const a = run.decodeAppearance(ea[j]); if (a && a.poy !== null) dy = a.poy; }
      const X = WX(g.x), Y = WY(g.y + dy), R = Math.max(5, g.r * view.s);
      hits.push({ row, X, Y, R });
      if (selected && !selected.player && selected.idx === row[0] && selected.seed === row[EI.init_seed]) ring(ctx, X, Y, Math.max(R, 6) + 5);
      if (opt.labels && row[1] >= 10 && row[1] < 1000) {
        const hp = row[EI.hp], mhp = row[EI.max_hp];
        if (mhp > 0 && hp !== null && hp > 0 && hp < mhp) {
          const bw = Math.max(18, R * 2);
          ctx.fillStyle = '#300'; ctx.fillRect(X - bw / 2, Y - R - 14, bw, 3);
          ctx.fillStyle = '#e33'; ctx.fillRect(X - bw / 2, Y - R - 14, bw * Math.max(0, Math.min(1, hp / mhp)), 3);
        }
      }
    });
    for (const row of prows) {
      if (row[PI.x] === null) continue;
      const X = WX(row[PI.x] / 100), Y = WY(row[PI.y] / 100), R = Math.max(8, 12 * view.s);
      hits.push({ player: true, row, X, Y, R });
      if (selected && selected.player && selected.idx === row[0]) ring(ctx, X, Y - 10 * view.s, R + 8);
    }
    hitList = hits;
    // HUD 与小地图（屏幕坐标；按画面大小缩放）
    const u = Math.max(1, Math.min(W / 480, H / 270)) * 0.9;
    if (opt.hud) {
      art.drawHud(ctx, F, 6, 4, u);
      art.drawMinimap(ctx, F, W - 13 * 9 * u * 0.62 - 10, 8, u * 0.62);
    }
    const q2 = performance.now();
    if (run.format < 2) {
      ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(W / 2 - 150, H - 26, 300, 20);
      ctx.fillStyle = '#ffd38a'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('旧格式（v1），翻转/颜色/叠加帧/网格变体为估计', W / 2, H - 16); ctx.textAlign = 'left';
    }
    if (art.A.pending > 0) { ctx.fillStyle = '#aaa'; ctx.font = '11px sans-serif'; ctx.textBaseline = 'bottom'; ctx.fillText(`载入素材… ${art.A.pending}`, 8, H - 6); }
    RV.roomParts = [q1 - q0, 0, q2 - q1, erows ? erows.length : 0];
    return rs;
  }
  function ring(ctx, X, Y, R) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.arc(X, Y, R, 0, 7); ctx.stroke(); ctx.setLineDash([]); }

  // ------------------------------------------------------------------ 实录画面层（isaac-footage-v1）
  function vid() { return $('vid'); }
  function footageActive() { return footage && (mode === 'video' || mode === 'split'); }
  function footageTime(i) { return (i + 1 - footage.first_frame + 0.5) / footage.fps; }   // i = 帧下标（n = i + 1）
  function syncVideo() {
    if (!footageActive() || videoDriven) return;
    const v = vid();
    const t = footageTime(cur);
    const inRange = t >= 0 && (cur + 1 - footage.first_frame) < footage.frames;
    $('vidmsg').hidden = inRange;
    if (inRange && Math.abs(v.currentTime - t) > 0.4 / footage.fps) { try { v.currentTime = t; } catch (e) { /* 还没有元数据 */ } }
  }
  function applyMode() {
    const main = document.querySelector('main');
    main.dataset.mode = mode;
    $('room').hidden = mode === 'video';
    $('vidwrap').hidden = !(mode === 'video' || mode === 'split');
    document.body.classList.toggle('artui', mode !== 'geo');
    $('mode').value = mode;
    if (!footageActive() && vid()) { vid().pause(); videoDriven = false; }
    tlCache = null; needRender = true;
    setTimeout(() => { needRender = true; syncVideo(); }, 30);
  }

  // ------------------------------------------------------------------ 进度条
  function buildIndex() {
    const rooms = D.rooms;
    roomVisits = rooms.map((r, k) => ({ ri: k, at: r.at, end: (k + 1 < rooms.length ? rooms[k + 1].at : N), room: r, level: run.levelIndexAt(r.at) }))
      .filter((v) => v.at < N);
    markers = [];
    for (const ev of D.events) {
      if (ev.ev === 'pause' || ev.ev === 'unpause') continue;
      if (ev.k === 'api' && API_QUIET.has(ev.m)) continue;   // 每次渲染都调用的那几种（按住地图键时的射击冷却、音效）不上时间轴
      markers.push({ i: ev.n - 1, kind: ev.ev || ev.k, label: eventText(ev), color: EV_COLORS[ev.ev || ev.k] || '#bbb' });
    }
    markers.sort((a, b) => a.i - b.i);
    // 暂停区间（只用于进度条底纹）
    RV.pauses = [];
    let ps = null;
    for (const ev of D.events) {
      if (ev.ev === 'pause') ps = ev.n - 1;
      else if (ev.ev === 'unpause' && ps !== null) { RV.pauses.push([ps, ev.n - 1]); ps = null; }
    }
    // 每次进房的受伤数
    for (const v of roomVisits) v.dmg = markers.filter((m) => m.kind === 'damage' && m.i >= v.at && m.i < v.end).length;
  }
  function drawTimeline() {
    const cv = $('tl'), dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (!tlCache || tlCache.W !== W || tlCache.H !== H || tlCache.dpr !== dpr) {
      const off = document.createElement('canvas'); off.width = W * dpr; off.height = H * dpr;
      const c = off.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.fillStyle = mode === 'geo' ? '#16161b' : '#3a2a1a'; c.fillRect(0, 0, W, H);
      const X = (i) => i / Math.max(1, N - 1) * W;
      // 层
      D.levels.forEach((lv, k) => {
        const a = X(lv.at), b = X(k + 1 < D.levels.length ? D.levels[k + 1].at : N);
        c.fillStyle = STAGE_COLORS[(lv.stage - 1 + 14) % 14]; c.fillRect(a, 0, b - a, 14);
        c.fillStyle = '#fff'; c.font = '10px sans-serif'; c.textBaseline = 'middle'; c.textAlign = 'left';
        c.save(); c.beginPath(); c.rect(a, 0, b - a, 14); c.clip(); c.fillText(lv.name || ('stage ' + lv.stage), a + 3, 7); c.restore();
        c.fillStyle = '#000'; c.fillRect(a, 0, 1, H);
      });
      // 房间
      roomVisits.forEach((v, k) => {
        const a = X(v.at), b = X(v.end);
        const t = v.room.type;
        c.fillStyle = t === 1 ? (k % 2 ? '#3a3d45' : '#4a4e58') : (ROOM_COLORS[t] || '#777');
        c.fillRect(a, 16, Math.max(1, b - a), 12);
      });
      // 暂停底纹
      c.fillStyle = 'rgba(255,255,255,0.10)';
      for (const [a, b] of RV.pauses) c.fillRect(X(a), 30, Math.max(1, X(b) - X(a)), H - 30);
      // 事件
      for (const m of markers) {
        c.fillStyle = m.color;
        const x = Math.round(X(m.i));
        const y0 = m.kind === 'damage' ? 30 : (m.kind === 'use_item' ? 38 : 46);
        c.fillRect(x - 1, y0, 2, 8);
      }
      tlCache = { W, H, dpr, img: off };
    }
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const ctx = cv.getContext('2d'); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(tlCache.img, 0, 0); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const x = cur / Math.max(1, N - 1) * W;
    ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, H);
    ctx.beginPath(); ctx.moveTo(x - 5, 0); ctx.lineTo(x + 5, 0); ctx.lineTo(x, 6); ctx.fill();
  }
  function curVisit() {
    const ri = run.roomIndexAt(cur);
    let lo = 0, hi = roomVisits.length - 1, best = 0;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (roomVisits[m].at <= cur) { best = m; lo = m + 1; } else hi = m - 1; }
    return roomVisits[best] || { at: 0, end: N, ri };
  }
  function drawRoomTimeline() {
    const cv = $('tl2'), dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const v = curVisit();
    const a = v.at, b = Math.max(v.at + 1, v.end - 1);
    const X = (i) => (i - a) / Math.max(1, b - a) * W;
    ctx.fillStyle = mode === 'geo' ? '#1d1d24' : '#4a3520'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = 'rgba(255,255,255,0.10)';
    for (const [p, q] of RV.pauses) if (q >= a && p <= b) ctx.fillRect(X(p), 0, Math.max(1, X(q) - X(p)), H);
    // 每秒刻度
    ctx.fillStyle = '#444';
    const step = (b - a) > 30 * 120 ? 300 : (b - a) > 30 * 20 ? 30 : 10;
    for (let i = Math.ceil(a / step) * step; i <= b; i += step) ctx.fillRect(X(i), H - 6, 1, 6);
    for (const m of markers) if (m.i >= a && m.i <= b) { ctx.fillStyle = m.color; ctx.fillRect(X(m.i) - 1, 2, 3, H - 10); }
    const x = X(cur);
    ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, H);
    ctx.fillStyle = '#aaa'; ctx.font = '10px sans-serif'; ctx.textBaseline = 'top';
    ctx.textAlign = 'left'; ctx.fillText(`房间段 帧 ${a + 1}`, 4, 2);
    ctx.textAlign = 'right'; ctx.fillText(`${v.end} · ${((v.end - v.at) / 30).toFixed(1)} s`, W - 4, 2);
  }

  // ------------------------------------------------------------------ 面板
  let lastItemsKey = null, lastVisitKey = null, lastLevelKey = null, lastEntKey = 0;
  function renderPanels(rs) {
    const fr = run.frames[cur];
    const lv = run.level(cur);
    const room = rs.room;
    const rf = fr[C.r];
    $('fnum').textContent = `${fr[C.n]} / ${N}`;
    $('ftime').textContent = fmtTime(fr[C.n]);
    const t0 = run.frames[0][C.t];
    $('fwall').textContent = fmtTime((fr[C.t] - t0) / 1000 * 30);
    $('fgf').textContent = fr[C.gf];
    $('fkf').textContent = fr[C.kf] ? '是' : '否';
    $('fmt').textContent = fr[C.mt] ? `[${fr[C.mt][0]}, ${fr[C.mt][1]}]` : '—';
    $('fdg').textContent = fr[C.dg] ? fr[C.dg].join(' / ') : '—';
    $('lvl').textContent = lv ? `${lv.name}（stage ${lv.stage}${lv.stage_type ? '.' + lv.stage_type : ''}${lv.curses ? '，诅咒 ' + lv.curses : ''}）` : '—';
    if (room) {
      const desc = room.desc || [];
      $('rinfo').textContent = `${ROOM_TYPES[room.type] || room.type} · ${SHAPES[room.shape] || room.shape} · 房间号 ${room.room_index} · 变体 ${desc[7]} · 难度 ${desc[11]}${desc[10] ? ' · “' + desc[10] + '”' : ''}`;
      $('rstate').textContent = `${rf[1] ? '已清' : '未清'} · 存活敌人 ${rf[2]} · 房间帧 ${rf[0]}`;
    }
    $('top-room').textContent = room ? `${lv ? lv.name : ''} · ${ROOM_TYPES[room.type] || room.type}（${room.room_index}）` : '';
    // 玩家
    const prows = run.playerRows(cur);
    const p = prows[0] ? run.decodePlayer(prows[0]) : null;
    if (p) {
      const hearts = [];
      const red = p.hearts || 0, max = p.max_hearts || 0;
      for (let k = 0; k < max; k += 2) hearts.push(`<span class="h ${red - k >= 2 ? 'full' : red - k === 1 ? 'half' : 'empty'}"></span>`);
      const soul = p.soul || 0, blackN = popcount(p.black || 0);
      for (let k = 0; k < soul; k += 2) hearts.push(`<span class="h soul${soul - k === 1 ? ' half' : ''}"></span>`);
      for (let k = 0; k < (p.bone || 0); k++) hearts.push('<span class="h bone"></span>');
      for (let k = 0; k < (p.broken || 0); k++) hearts.push('<span class="h broken"></span>');
      $('hearts').innerHTML = hearts.join('') +
        `<div class="sub">红 ${red}/${max} · 魂 ${soul}（黑 ${blackN} 格）· 骨 ${p.bone} · 永恒 ${p.eternal} · 金 ${p.golden} · 碎 ${p.broken} · 腐 ${p.rotten}${p.extra_lives ? ' · 额外命 ' + p.extra_lives : ''}</div>`;
      $('res').textContent = `¢ ${p.coins}   💣 ${p.bombs}${p.golden_bomb ? '★' : ''}   🔑 ${p.keys}${p.golden_key ? '★' : ''}`;
      const tears = p.max_fire_delay !== null ? (30 / (p.max_fire_delay + 1)).toFixed(2) : '—';
      $('stats').innerHTML = [
        ['伤害', p.damage], ['射速', tears], ['射程', p.tear_range !== null ? (p.tear_range / 40).toFixed(2) : '—'], ['弹速', p.shot_speed],
        ['移速', p.move_speed], ['幸运', p.luck], ['冷却', p.fire_delay], ['飞行', p.can_fly ? '是' : '否'],
      ].map(([k, v]) => `<span>${k}</span><b>${typeof v === 'number' ? +v.toFixed(2) : v}</b>`).join('');
      const act = [];
      for (let k = 0; k < 4; k++) {
        const id = p['active' + k];
        if (id > 0 || k === 0) act.push(`<div class="slot">${iconHtml('c' + id)}<span>${k === 0 ? '主动' : '槽' + k}: ${itemName(id)} · 充能 ${p['charge' + k] ?? '—'}${k === 0 && p.battery0 ? '+' + p.battery0 : ''}</span></div>`);
      }
      act.push(`<div class="slot"><span>卡: ${cardName(p.card0)}${p.card1 ? ' / ' + cardName(p.card1) : ''} · 药: ${p.pill0 || '—'}${p.pill1 ? ' / ' + p.pill1 : ''}</span></div>`);
      act.push(`<div class="slot">${p.trinket0 ? iconHtml('t' + (p.trinket0 & 0x7fff)) : ''}<span>饰品: ${trinketName(p.trinket0)}${p.trinket1 ? ' / ' + trinketName(p.trinket1) : ''}</span></div>`);
      $('actives').innerHTML = act.join('');
      $('pmisc').textContent = `位置 (${p.x}, ${p.y}) · 速度 (${p.vx}, ${p.vy}) · 头 ${DIRS[p.head_dir] ?? p.head_dir} · 射 ${DIRS[p.fire_dir] ?? p.fire_dir} · 移 ${DIRS[p.move_dir] ?? p.move_dir} · 无敌 ${p.damage_cooldown} · 累计受伤 ${p.total_damage_taken} · 动画 ${p.anim}#${p.aframe}${p.overlay ? ' / ' + p.overlay : ''}`;
    }
    // 输入
    const [uBits, anyBits, trig] = run.inputSummary[cur];
    const btns = $('inputs').children;
    for (let j = 0; j < 12; j++) {
      const b = btns[j];
      b.className = 'key' + ((uBits >> j) & 1 ? ' on' : (anyBits >> j) & 1 ? ' tick' : '') + ((trig >> j) & 1 ? ' trig' : '');
    }
    const tk = fr[C.tk];
    $('tkinfo').textContent = `本帧 tick: ${tk.map((e) => (typeof e === 'string' ? e : e[0] + '*')).join(' ')}`;
    // 道具（只在道具行变化时重画）
    const il = run.itemsLine(cur, 0);
    const ikey = il ? il.at + ':' + il.n : 'none';
    if (ikey !== lastItemsKey) {
      lastItemsKey = ikey;
      const its = il ? il.items : [];
      $('items-h').textContent = `道具（${il ? il.count : 0}）`;
      $('items').innerHTML = its.map(([id, c]) => `<div class="it" title="${esc(itemName(id))}（${id}）">${iconHtml('c' + id) || `<span class="noicon">${id}</span>`}${c > 1 ? `<i>×${c}</i>` : ''}<span class="nm">${esc(itemName(id))}</span></div>`).join('');
    }
    // 本帧事件
    const evs = run.events(cur);
    $('fev').innerHTML = evs.length ? evs.map((e) => `<div>${esc(eventText(e))}</div>`).join('') : '<span class="dim">无</span>';
    // 实体列表（播放时限频）
    const now = performance.now();
    if (!playing || now - lastEntKey > 250) { lastEntKey = now; renderEntityList(); }
    renderSelected();
    // 左侧房间列表高亮 + 小地图
    const v = curVisit();
    const vkey = v.ri;
    if (vkey !== lastVisitKey) {
      lastVisitKey = vkey;
      document.querySelectorAll('#rooms .rv.cur').forEach((e) => e.classList.remove('cur'));
      const el = document.querySelector(`#rooms .rv[data-k="${v.ri}"]`);
      if (el) { el.classList.add('cur'); if (!RV.noScroll) el.scrollIntoView({ block: 'nearest' }); }
    }
    drawMinimap(lv, rs);
  }
  function iconHtml(key) { return names.icons[key] ? `<img class="ic" src="${names.icons[key]}">` : ''; }
  function renderEntityList() {
    const erows = run.entityRows(cur);
    const el = $('ents');
    if (!erows) { el.innerHTML = '<span class="dim">本帧未记录实体</span>'; return; }
    const counts = {};
    let rows = [];
    for (const r of erows) {
      const t = r[1];
      if (t === 1000) { counts.effect = (counts.effect || 0) + 1; continue; }
      if (t === 2) { counts.tear = (counts.tear || 0) + 1; continue; }
      if (t === 9) { counts.proj = (counts.proj || 0) + 1; continue; }
      rows.push(r);
    }
    $('ents-h').textContent = `实体（${erows.length}：泪 ${counts.tear || 0} · 敌弹 ${counts.proj || 0} · 特效 ${counts.effect || 0}）`;
    rows.sort((a, b) => catOrder(b[1]) - catOrder(a[1]));
    el.innerHTML = rows.slice(0, 80).map((r) => {
      const t = r[1];
      const nm = t === 5 ? pickupLabel(r) : (entName(t, r[2], r[3]) || '');
      const hp = r[EI.max_hp] > 0 ? ` ${fx(r[EI.hp])}/${fx(r[EI.max_hp])}` : '';
      const sel = selected && selected.idx === r[0] && selected.seed === r[EI.init_seed] ? ' sel' : '';
      return `<div class="er${sel}" data-idx="${r[0]}" data-seed="${r[EI.init_seed]}"><code>${t}.${r[2]}.${r[3]}</code> ${esc(nm)}<span class="dim">${hp}</span></div>`;
    }).join('') + (rows.length > 80 ? `<div class="dim">…还有 ${rows.length - 80} 个</div>` : '');
  }
  function renderSelected() {
    const el = $('selent');
    if (!selected) { el.innerHTML = '<span class="dim">点击画面里的实体或上面列表查看全部字段</span>'; return; }
    let row = null;
    if (selected.player) row = run.playerRows(cur).find((r) => r[0] === selected.idx) || null;
    else row = (run.entityRows(cur) || []).find((r) => r[0] === selected.idx && r[EI.init_seed] === selected.seed) || null;
    if (!row) { el.innerHTML = `<span class="dim">所选对象（Index ${selected.idx}）本帧不存在</span>`; return; }
    if (selected.player) {
      const p = run.decodePlayer(row);
      el.innerHTML = '<div class="kv">' + Object.entries(p).map(([k, v]) => `<span>${k}</span><b>${esc(v === null ? 'null' : v)}</b>`).join('') + '</div>';
      return;
    }
    const e = run.decodeEntity(row);
    const nm = e.type === 5 ? pickupLabel(row) : entName(e.type, e.variant, e.subtype);
    let h = `<div class="selname">${esc(nm || '')} <code>${e.type}.${e.variant}.${e.subtype}</code> · ${e.category || '?'}</div><div class="kv">`;
    for (const [k, v] of Object.entries(e)) if (k !== 'extra' && k !== 'category') h += `<span>${k}</span><b>${esc(v === null ? 'null' : v)}</b>`;
    if (e.extra) for (const [k, v] of Object.entries(e.extra)) h += `<span>${e.category}.${k}</span><b>${esc(v === null ? 'null' : v)}</b>`;
    el.innerHTML = h + '</div>';
  }
  function drawMinimap(lv, rs) {
    const cv = $('mini'), dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#121216'; ctx.fillRect(0, 0, W, H);
    if (!lv) return;
    const cs = Math.floor(Math.min(W, H) / 13);
    const ox = (W - cs * 13) / 2, oy = (H - cs * 13) / 2;
    const curList = rs.room && rs.room.desc ? rs.room.desc[0] : null;
    // 本层到当前帧为止进过的房间（按 list_index）
    const visited = new Set();
    for (const v of roomVisits) { if (v.at > cur) break; if (v.at >= lv.at && v.room.desc) visited.add(v.room.desc[0]); }
    const foot = { 1: [[0, 0]], 2: [[0, 0]], 3: [[0, 0]], 4: [[0, 0], [0, 1]], 5: [[0, 0], [0, 1]], 6: [[0, 0], [1, 0]], 7: [[0, 0], [1, 0]],
      8: [[0, 0], [1, 0], [0, 1], [1, 1]], 9: [[1, 0], [0, 1], [1, 1]], 10: [[0, 0], [0, 1], [1, 1]], 11: [[0, 0], [1, 0], [1, 1]], 12: [[0, 0], [1, 0], [0, 1]] };
    for (const d of lv.rooms) {
      const gi = d[1]; if (gi < 0) continue;
      const gx = gi % 13, gy = Math.floor(gi / 13);
      const isCur = d[0] === curList, vis = visited.has(d[0]);
      const col = ROOM_COLORS[d[6]] || '#888';
      for (const [dx, dy] of (foot[d[9]] || [[0, 0]])) {
        ctx.fillStyle = isCur ? '#fff' : (vis ? (d[6] === 1 ? '#8a8f9a' : col) : 'rgba(120,120,130,0.25)');
        ctx.fillRect(ox + (gx + dx) * cs + 1, oy + (gy + dy) * cs + 1, cs - 2, cs - 2);
        if (d[6] !== 1 && !isCur) { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.strokeRect(ox + (gx + dx) * cs + 1.5, oy + (gy + dy) * cs + 1.5, cs - 3, cs - 3); }
      }
    }
    if (rs.room && rs.room.room_index < 0) { ctx.fillStyle = '#fc6'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(`特殊房间 ${rs.room.room_index}（不在地图格上）`, 4, 4); }
  }

  // ------------------------------------------------------------------ 左侧列表
  function buildLists() {
    const el = $('rooms');
    const parts = [];
    let lastLv = -2;
    roomVisits.forEach((v) => {
      if (v.level !== lastLv) {
        lastLv = v.level;
        const lv = D.levels[v.level];
        if (lv) parts.push(`<div class="lvh" data-at="${lv.at}">${esc(lv.name)} <span class="dim">帧 ${lv.at + 1} · ${fmtTime(lv.at)}</span></div>`);
      }
      const r = v.room, t = r.type;
      parts.push(`<div class="rv" data-k="${v.ri}" data-at="${v.at}"><i style="background:${ROOM_COLORS[t] || '#888'}"></i>${esc(ROOM_TYPES[t] || t)} <span class="dim">#${r.room_index} ${SHAPES[r.shape] || ''}</span>` +
        `<span class="rt">${fmtTime(v.at)} · ${((v.end - v.at) / 30).toFixed(1)}s${v.dmg ? ` · <b class="dmg">♥-${v.dmg}</b>` : ''}</span></div>`);
    });
    el.innerHTML = parts.join('');
    el.addEventListener('click', (e) => { const d = e.target.closest('[data-at]'); if (d) { RV.noScroll = true; setFrame(+d.dataset.at); setTimeout(() => { RV.noScroll = false; }, 50); } });
    const ev = $('events');
    ev.innerHTML = markers.map((m) => `<div class="evr" data-i="${m.i}"><i style="background:${m.color}"></i><span class="dim">${fmtTime(m.i + 1)}</span> ${esc(m.label)}</div>`).join('');
    ev.addEventListener('click', (e) => { const d = e.target.closest('[data-i]'); if (d) setFrame(+d.dataset.i); });
  }

  // ------------------------------------------------------------------ 控制
  function setFrame(i) {
    i = Math.max(0, Math.min(N - 1, Math.round(i)));
    if (i === cur && !needRender) return;
    cur = i; needRender = true;
  }
  function render() {
    const t0 = performance.now();
    const rs = renderRoom();
    const t1 = performance.now();
    renderPanels(rs);
    const t2 = performance.now();
    drawTimeline();
    drawRoomTimeline();
    RV.lastParts = [t1 - t0, t2 - t1, performance.now() - t2];
    syncVideo();
    $('playbtn').textContent = playing ? '❚❚' : '▶';
    const dt = performance.now() - t0;
    RV.lastRenderMs = dt;
    RV.renderMs = RV.renderMs || []; RV.renderMs.push(dt); if (RV.renderMs.length > 600) RV.renderMs.shift();
    RV.cur = cur;
  }
  let hashTimer = null;
  function loop(ts) {
    const v = footage ? vid() : null;
    const useVideo = playing && footageActive() && speed <= 16 && v && v.readyState >= 2 &&
      (cur + 1 - footage.first_frame) >= 0 && (cur + 1 - footage.first_frame) < footage.frames - 1;
    if (useVideo) {
      // 播放时用 <video> 的时间驱动帧号（倍速 = playbackRate；浏览器上限 16x，更快时退化为逐帧 seek）
      videoDriven = true;
      if (v.playbackRate !== speed) v.playbackRate = speed;
      if (v.paused) v.play().catch(() => {});
      const n = footage.first_frame + Math.floor(v.currentTime * footage.fps + 1e-6);
      const i = Math.max(0, Math.min(N - 1, n - 1));
      if (i !== cur) { cur = i; needRender = true; }
      if (n - footage.first_frame >= footage.frames - 1) { playing = false; v.pause(); videoDriven = false; }
    } else {
      if (videoDriven) { videoDriven = false; if (v && !v.paused) v.pause(); }
      if (playing) {
        acc += Math.min(250, ts - lastTs) / 1000 * 30 * speed;
        const k = Math.floor(acc);
        if (k > 0) { acc -= k; if (cur + k >= N - 1) { setFrame(N - 1); playing = false; } else setFrame(cur + k); }
      }
    }
    lastTs = ts;
    if (needRender) {
      needRender = false; render();
      if (!playing) { clearTimeout(hashTimer); hashTimer = setTimeout(() => { try { history.replaceState(null, '', '#f=' + (cur + 1)); } catch (e) { /* file:// 某些浏览器不允许 */ } }, 300); }
    }
    requestAnimationFrame(loop);
  }
  function nextMarker(dir) {
    if (dir > 0) { const m = markers.find((m) => m.i > cur); if (m) setFrame(m.i); }
    else { for (let k = markers.length - 1; k >= 0; k--) if (markers[k].i < cur) { setFrame(markers[k].i); break; } }
  }
  function nextRoom(dir) {
    const v = curVisit();
    const k = roomVisits.indexOf(v);
    if (dir > 0 && k + 1 < roomVisits.length) setFrame(roomVisits[k + 1].at);
    if (dir < 0) setFrame(cur > v.at ? v.at : (k > 0 ? roomVisits[k - 1].at : 0));
  }
  const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32];
  function setSpeed(s) { speed = s; $('speed').value = String(s); }
  function bindControls() {
    $('playbtn').onclick = () => { playing = !playing; acc = 0; needRender = true; };
    $('prevf').onclick = () => setFrame(cur - 1);
    $('nextf').onclick = () => setFrame(cur + 1);
    $('prevr').onclick = () => nextRoom(-1);
    $('nextr').onclick = () => nextRoom(1);
    $('preve').onclick = () => nextMarker(-1);
    $('nexte').onclick = () => nextMarker(1);
    $('speed').innerHTML = SPEEDS.map((s) => `<option value="${s}">${s}×</option>`).join('');
    $('speed').value = '1';
    $('speed').onchange = (e) => { speed = +e.target.value; };
    $('goto').onkeydown = (e) => { if (e.key === 'Enter') { setFrame(+e.target.value - 1); e.target.blur(); } };
    for (const [id, k] of [['o-fx', 'effects'], ['o-lb', 'labels'], ['o-vel', 'vel'], ['o-gi', 'gridIdx'], ['o-inv', 'invisible']]) {
      $(id).checked = opt[k];
      $(id).onchange = (e) => { opt[k] = e.target.checked; needRender = true; };
    }
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
      const big = e.shiftKey ? 30 : 1;
      switch (e.key) {
        case ' ': playing = !playing; acc = 0; needRender = true; break;
        case 'ArrowLeft': playing = false; setFrame(cur - big); break;
        case 'ArrowRight': playing = false; setFrame(cur + big); break;
        case 'ArrowUp': nextRoom(-1); break;
        case 'ArrowDown': nextRoom(1); break;
        case '[': nextMarker(-1); break;
        case ']': nextMarker(1); break;
        case 'Home': setFrame(0); break;
        case 'End': setFrame(N - 1); break;
        case '+': case '=': { const k = SPEEDS.indexOf(speed); setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, k + 1)]); break; }
        case '-': { const k = SPEEDS.indexOf(speed); setSpeed(SPEEDS[Math.max(0, k - 1)]); break; }
        case 'e': case 'E': opt.effects = !opt.effects; $('o-fx').checked = opt.effects; needRender = true; break;
        case 'l': case 'L': opt.labels = !opt.labels; $('o-lb').checked = opt.labels; needRender = true; break;
        case 'v': case 'V': opt.vel = !opt.vel; $('o-vel').checked = opt.vel; needRender = true; break;
        case 'g': case 'G': opt.gridIdx = !opt.gridIdx; $('o-gi').checked = opt.gridIdx; needRender = true; break;
        case 'f': case 'F': cam.follow = !cam.follow; if (!cam.follow) { cam.panX = 0; cam.panY = 0; } $('o-follow').checked = cam.follow; needRender = true; break;
        case 'r': case 'R': cam.zoom = 1; cam.panX = 0; cam.panY = 0; cam.follow = false; cam.game = false; $('o-follow').checked = false; needRender = true; break;
        case 'c': case 'C': cam.game = !cam.game; needRender = true; break;
        case 'h': case 'H': opt.hud = !opt.hud; $('o-hud').checked = opt.hud; needRender = true; break;
        case 'm': case 'M': { const ms = Array.from($('mode').options).filter((o) => !o.disabled).map((o) => o.value); mode = ms[(ms.indexOf(mode) + 1) % ms.length]; applyMode(); break; }
        default: return;
      }
      e.preventDefault();
    });
    // 进度条拖动
    function scrub(cv, toFrame) {
      let down = false;
      const at = (e) => { const r = cv.getBoundingClientRect(); return toFrame((e.clientX - r.left) / r.width, e.clientY - r.top); };
      cv.addEventListener('pointerdown', (e) => { down = true; cv.setPointerCapture(e.pointerId); playing = false; setFrame(at(e)); });
      cv.addEventListener('pointermove', (e) => {
        if (down) setFrame(at(e));
        const i = at(e);
        const v = roomVisits.find((q) => q.at <= i && i < q.end);
        const near = markers.filter((m) => Math.abs(m.i - i) < Math.max(3, (cv === $('tl') ? N : 600) / cv.clientWidth * 3));
        cv.title = `帧 ${i + 1} · ${fmtTime(i + 1)}` + (v ? ` · ${ROOM_TYPES[v.room.type] || ''} #${v.room.room_index}` : '') + (near.length ? '\n' + near.slice(0, 4).map((m) => m.label).join('\n') : '');
      });
      cv.addEventListener('pointerup', () => { down = false; });
    }
    scrub($('tl'), (fr, y) => {
      let i = Math.round(fr * (N - 1));
      if (y >= 30) { // 事件行：吸附到 3 px 内的事件
        const px = (N - 1) / $('tl').clientWidth * 3;
        let best = null;
        for (const m of markers) if (Math.abs(m.i - i) <= px && (!best || Math.abs(m.i - i) < Math.abs(best.i - i))) best = m;
        if (best) i = best.i;
      }
      return Math.max(0, Math.min(N - 1, i));
    });
    scrub($('tl2'), (fr) => { const v = curVisit(); const b = Math.max(v.at + 1, v.end - 1); return Math.max(v.at, Math.min(b, Math.round(v.at + fr * (b - v.at)))); });
    // 画面：悬停提示与点选
    const cv = $('room');
    cv.addEventListener('mousemove', (e) => {
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = null, bd = 1e9;
      for (const h of hitList) { const d = Math.hypot(h.X - mx, h.Y - my); if (d <= h.R + 4 && d < bd) { bd = d; best = h; } }
      const tip = $('tip');
      const wx = (mx - view.ox) / view.s, wy = (my - view.oy) / view.s;
      if (best) {
        const row = best.row;
        let txt;
        if (best.player) txt = `玩家 ${row[0]} (${row[PI.x] / 100}, ${row[PI.y] / 100})`;
        else {
          const t = row[1];
          const nm = t === 5 ? pickupLabel(row) : (entName(t, row[2], row[3]) || '');
          txt = `${nm} ${t}.${row[2]}.${row[3]} · Index ${row[0]}\n位置 (${fx(row[4])}, ${fx(row[5])}) 速度 (${fx(row[6])}, ${fx(row[7])})` +
            (row[EI.max_hp] > 0 ? `\nHP ${fx(row[EI.hp])}/${fx(row[EI.max_hp])}` : '') + `\n动画 ${row[EI.anim]}#${row[EI.aframe]}`;
        }
        tip.textContent = txt;
      } else {
        const rs = run.roomState(cur);
        let gtxt = '';
        if (rs.room) {
          const c = Math.floor((wx - 20) / 40), rr = Math.floor((wy - 100) / 40);
          if (c >= 0 && rr >= 0 && c < rs.room.gw && rr < rs.room.gh) {
            const gi = rr * rs.room.gw + c, g = rs.grid.get(gi);
            gtxt = `格 ${gi} (${c}, ${rr})` + (g ? ` ${GRID_NAMES[g[1]] || g[1]} var ${g[2]} state ${g[3]} coll ${g[4]}` : '');
          }
        }
        tip.textContent = `(${wx.toFixed(1)}, ${wy.toFixed(1)}) ${gtxt}`;
      }
      tip.style.left = (mx + 14) + 'px'; tip.style.top = (my + 14) + 'px'; tip.hidden = false;
    });
    cv.addEventListener('mouseleave', () => { $('tip').hidden = true; });
    cv.addEventListener('click', (e) => {
      if (suppressClick) { suppressClick = false; return; }
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      let best = null, bd = 1e9;
      for (const h of hitList) { const d = Math.hypot(h.X - mx, h.Y - my); if (d <= h.R + 4 && d < bd) { bd = d; best = h; } }
      selected = best ? (best.player ? { player: true, idx: best.row[0] } : { idx: best.row[0], seed: best.row[EI.init_seed] }) : null;
      needRender = true;
    });
    // 自由视角：滚轮 = 以鼠标为中心缩放；Alt+滚轮 = 逐帧（Shift = 30 帧）；拖拽 = 平移；F 跟随玩家；R 复位整房
    const stepWheel = (e) => { e.preventDefault(); playing = false; setFrame(cur + (e.deltaY > 0 ? 1 : -1) * (e.shiftKey ? 30 : 1)); };
    cv.addEventListener('wheel', (e) => {
      if (e.altKey) return stepWheel(e);
      e.preventDefault();
      const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const wx = (mx - view.ox) / view.s, wy = (my - view.oy) / view.s;
      const z = Math.max(0.25, Math.min(12, cam.zoom * Math.pow(1.0015, -e.deltaY)));
      if (!cam.follow) {
        // 缩放后鼠标下的世界点不动：ox' = mx - wx*s'，center' = (W/2 - ox')/s'
        const s2 = view.s0 * z, W = cv.clientWidth, H = cv.clientHeight;
        const cx = (W / 2 - (mx - wx * s2)) / s2, cy = (H / 2 - (my - wy * s2)) / s2;
        const [x0, y0, x1, y1] = view.bbox;
        cam.panX = cx - (x0 + x1) / 2; cam.panY = cy - (y0 + y1) / 2;
      }
      cam.zoom = z; needRender = true;
    }, { passive: false });
    $('tl').addEventListener('wheel', stepWheel, { passive: false });
    $('tl2').addEventListener('wheel', stepWheel, { passive: false });
    cv.addEventListener('pointerdown', (e) => { if (e.button === 0) dragPan = { x: e.clientX, y: e.clientY, moved: false, id: e.pointerId }; });
    cv.addEventListener('pointermove', (e) => {
      if (!dragPan || !(e.buttons & 1)) return;
      const dx = e.clientX - dragPan.x, dy = e.clientY - dragPan.y;
      if (!dragPan.moved && Math.hypot(dx, dy) < 4) return;
      if (!dragPan.moved) { dragPan.moved = true; try { cv.setPointerCapture(dragPan.id); } catch (err) { /* ignore */ } if (cam.follow) { cam.follow = false; $('o-follow').checked = false; const pr = run.playerRows(cur)[0]; const [x0, y0, x1, y1] = view.bbox; if (pr) { cam.panX = pr[PI.x] / 100 - (x0 + x1) / 2; cam.panY = pr[PI.y] / 100 - (y0 + y1) / 2; } } }
      cam.panX -= dx / view.s; cam.panY -= dy / view.s;
      dragPan.x = e.clientX; dragPan.y = e.clientY; needRender = true;
    });
    cv.addEventListener('pointerup', () => { if (dragPan && dragPan.moved) suppressClick = true; dragPan = null; });
    $('mode').onchange = (e) => { mode = e.target.value; applyMode(); };
    $('o-follow').onchange = (e) => { cam.follow = e.target.checked; cam.panX = 0; cam.panY = 0; needRender = true; };
    $('o-hud').onchange = (e) => { opt.hud = e.target.checked; needRender = true; };
    $('camreset').onclick = () => { cam.zoom = 1; cam.panX = 0; cam.panY = 0; cam.follow = false; cam.game = false; $('o-follow').checked = false; needRender = true; };
    $('ents').addEventListener('click', (e) => { const d = e.target.closest('[data-idx]'); if (d) { selected = { idx: +d.dataset.idx, seed: +d.dataset.seed }; needRender = true; } });
    window.addEventListener('resize', () => { tlCache = null; needRender = true; });
    window.addEventListener('hashchange', () => { const m = /f=(\d+)/.exec(location.hash); if (m && +m[1] - 1 !== cur) { playing = false; setFrame(+m[1] - 1); } });
  }

  // ------------------------------------------------------------------ 加载
  async function loadData() {
    const emb = document.getElementById('rv-data');
    let bytes;
    const t0 = performance.now();
    if (emb && emb.textContent.trim().length > 10) {
      const bin = atob(emb.textContent.trim());
      bytes = new Uint8Array(bin.length);
      for (let k = 0; k < bin.length; k++) bytes[k] = bin.charCodeAt(k);
    } else {
      const r = await fetch('data.json.gz');
      if (!r.ok) throw new Error('读数据失败 ' + r.status);
      bytes = new Uint8Array(await r.arrayBuffer());
    }
    const t1 = performance.now();
    let text;
    if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
      const ds = new DecompressionStream('gzip');
      text = await new Response(new Blob([bytes]).stream().pipeThrough(ds)).text();
    } else text = new TextDecoder().decode(bytes);
    const t2 = performance.now();
    const d = JSON.parse(text);
    const t3 = performance.now();
    RV.timing.fetch_ms = Math.round(t1 - t0); RV.timing.gunzip_ms = Math.round(t2 - t1); RV.timing.parse_ms = Math.round(t3 - t2);
    RV.timing.gz_bytes = bytes.length; RV.timing.json_bytes = text.length;
    return d;
  }
  async function main() {
    const T0 = performance.now();
    const msg = $('loading');
    try {
      msg.textContent = '读取并解压录制数据…';
      D = await loadData();
      msg.textContent = '建立索引…';
      await new Promise((r) => setTimeout(r, 0));
      const t = performance.now();
      run = new core.RunData(D);
      RV.timing.index_ms = Math.round(performance.now() - t);
      RV.run = run;
      N = run.N;
      names = Object.assign(names, D.names || {});
      PI = run.pIndex; EI = run.eIndex;
      // 真实美术（有素材表时）与实录画面层（有 footage 时）
      if (D.art && D.art.tables && window.IsaacArt) {
        art = new IsaacArt.ArtRenderer({ tables: D.art.tables, assets: D.art.assets || null, base: D.art.base || 'asset/', run, onload: () => { needRender = true; } });
        RV.art = art;
      }
      footage = D.footage || null;
      const mo = $('mode').options;
      for (const o of mo) o.disabled = (o.value === 'art' && !art) || (o.value === 'video' && !footage) || (o.value === 'split' && !(footage && art));
      if (footage) {
        const v = vid();
        v.src = footage.url;
        v.addEventListener('loadedmetadata', () => { RV.video = { duration: v.duration, w: v.videoWidth, h: v.videoHeight }; syncVideo(); });
        v.addEventListener('seeked', () => { RV.lastSeek = performance.now(); });
        $('vidinfo').textContent = `实录 ${footage.frames} 帧 @ ${footage.fps} fps，第 1 帧 = 逻辑帧 ${footage.first_frame}`;
      }
      const mm = /[#&?]mode=(geo|art|video|split)/.exec(location.hash + '&' + location.search);
      const okMode = (m) => Array.from(mo).some((o) => o.value === m && !o.disabled);
      mode = mm && okMode(mm[1]) ? mm[1] : (art ? 'art' : 'geo');
      applyMode();
      for (const [cat, list] of Object.entries(run.fields.entity_extra)) { XI[cat] = {}; list.forEach((nm, j) => { XI[cat][nm] = core.N_COMMON + j; }); }
      buildIndex();
      buildLists();
      bindControls();
      const h = D.header, end = D.end;
      const seeds = (h.meta && h.meta.seeds) || {};
      $('title').textContent = `${D.source.name}`;
      document.title = `${D.source.name} · 录制回放`;
      $('meta').textContent = `种子 ${seeds.start_seed_string || '?'} · ${((h.meta || {}).players || []).map((p) => p.name).join('/')} · ${N} 帧（${fmtTime(N)} 游戏时间）· ${D.levels.length} 层 · ${roomVisits.length} 次进房 · ` +
        `结束：${end ? end.reason : '无 end 行（截断）'} · MT ${h.mt && h.mt.available ? '有' : '无'} · recorder ${h.recorder}${h.continued ? ' · 续局' : ''}`;
      const m = /[#&?]f=(\d+)/.exec(location.hash + '&' + location.search);
      cur = m ? Math.max(0, Math.min(N - 1, +m[1] - 1)) : 0;
      const tr = performance.now();
      render();
      RV.timing.first_render_ms = Math.round(performance.now() - tr);
      RV.timing.total_ms = Math.round(performance.now() - T0);
      msg.hidden = true;
      $('loadinfo').textContent = `加载 ${(RV.timing.total_ms / 1000).toFixed(2)} s（解压 ${RV.timing.gunzip_ms} ms · 解析 ${RV.timing.parse_ms} ms · 索引 ${RV.timing.index_ms} ms）`;
      RV.ready = true;
      document.body.dataset.ready = '1';
      requestAnimationFrame((ts) => { lastTs = ts; loop(ts); });
    } catch (err) {
      msg.textContent = '加载失败：' + (err && err.stack || err);
      RV.error = String(err && err.stack || err);
      document.body.dataset.ready = 'error';
    }
  }
  // 调试/测量入口（REPORT 里的拖动测速用）
  RV.setCam = (c) => { Object.assign(cam, c); needRender = true; return Object.assign({}, cam); };
  RV.setFrame = (i) => { setFrame(i); render(); needRender = false; return RV.lastRenderMs; };
  RV.bench = (n = 2000, mode = 'random') => {
    const out = [];
    for (let k = 0; k < n; k++) {
      const i = mode === 'random' ? Math.floor(Math.random() * N) : (mode === 'seq' ? k % N : Math.floor(k / n * N));
      cur = i; const t = performance.now(); render(); out.push(performance.now() - t);
    }
    out.sort((a, b) => a - b);
    return { n, mode, mean: out.reduce((a, b) => a + b, 0) / n, p50: out[Math.floor(n * 0.5)], p95: out[Math.floor(n * 0.95)], p99: out[Math.floor(n * 0.99)], max: out[n - 1] };
  };
  window.addEventListener('DOMContentLoaded', main);
})();
