// verify.py 用：node node_dump.js DATA.json.gz FRAMES.json OUT.json
// 用与浏览器同一个 viewer_core.js 展开指定帧（数组下标），把完整状态写成 JSON。
'use strict';
const fs = require('fs');
const zlib = require('zlib');
const path = require('path');
const core = require(path.join(__dirname, 'viewer_core.js'));

const [dataPath, framesPath, outPath] = process.argv.slice(2);
const t0 = Date.now();
const d = JSON.parse(zlib.gunzipSync(fs.readFileSync(dataPath)).toString('utf8'));
const t1 = Date.now();
const run = new core.RunData(d);
const t2 = Date.now();
const idx = JSON.parse(fs.readFileSync(framesPath, 'utf8'));
const out = {};
for (const i of idx) out[i] = run.frameState(i);
const t3 = Date.now();
// 顺序全量展开一遍（检查每帧都能展开、不抛异常），并计时
let rows = 0;
for (let i = 0; i < run.N; i++) { const e = run.entityRows(i); rows += e ? e.length : 0; run.playerRows(i); run.roomState(i); }
const t4 = Date.now();
fs.writeFileSync(outPath, JSON.stringify({
  frames: out,
  timing_ms: { parse: t1 - t0, index: t2 - t1, sample: t3 - t2, full_sequential: t4 - t3 },
  total_entity_rows: rows, N: run.N, keyframes: run.kfIdx.length,
}));
