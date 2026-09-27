// Runs the floor generators (Python, web/api.py) in the browser with Pyodide. A module worker (Pyodide
// loads its own files with import()); the page sends {type: 'init', pyodide: <distribution URL>, game}
// once, then {type: 'call', id, method, params}. The generator code and the game's room data download
// while the runtime loads; numpy is not needed (isaac_macro/f32.py).
let pyodide = null;
let ready = null;
const games = new Map();       // game -> Promise that its data is unpacked and configured
const downloads = new Map();   // path -> Promise<ArrayBuffer>

function progress(step, state, text) {
  postMessage({ type: 'progress', step, state, text });
}

function fetchBuffer(path) {
  if (!downloads.has(path)) {
    downloads.set(path, fetch(new URL(path, self.location.href)).then((res) => {
      if (!res.ok) throw new Error(`下载 ${path} 失败（HTTP ${res.status}）`);
      return res.arrayBuffer();
    }));
  }
  return downloads.get(path);
}

async function init(url, game) {
  const code = fetchBuffer('../py/macro.zip');
  if (game) fetchBuffer(`../data/${game}.zip`).catch(() => {});
  progress('runtime', 'run', '下载 Python 运行环境');
  const { loadPyodide } = await import(url + 'pyodide.mjs');
  pyodide = await loadPyodide({ indexURL: url });
  progress('runtime', 'done');
  progress('code', 'run', '加载生成器');
  pyodide.unpackArchive(await code, 'zip', { extractDir: '/app' });
  pyodide.runPython("import sys\nsys.path.insert(0, '/app')\nimport api");
  progress('code', 'done');
}

function loadGame(game) {
  if (!games.has(game)) {
    games.set(game, (async () => {
      progress('data-' + game, 'run', `加载${game === 'repplus' ? '忏悔+' : '胎衣†'}的房间数据`);
      pyodide.unpackArchive(await fetchBuffer(`../data/${game}.zip`), 'zip', { extractDir: `/data/${game}` });
      pyodide.globals.set('_game', game);
      pyodide.runPython("api.configure({_game: '/data/' + _game})");
      progress('data-' + game, 'done');
    })());
  }
  return games.get(game);
}

self.onmessage = async (e) => {
  const m = e.data;
  if (m.type === 'init') {
    ready = init(m.pyodide, m.game);
    ready.catch((err) => progress('runtime', 'error', String(err && err.message || err)));
    return;
  }
  if (m.type !== 'call') return;
  try {
    await ready;
    if (m.method !== 'random') await loadGame(m.params.game || 'abplus');
    pyodide.globals.set('_method', m.method);
    pyodide.globals.set('_params', JSON.stringify(m.params));
    const out = JSON.parse(pyodide.runPython('api.call(_method, _params)'));
    if ('error' in out) postMessage({ id: m.id, error: out.error });
    else postMessage({ id: m.id, result: out.ok });
  } catch (err) {
    postMessage({ id: m.id, error: String(err && err.message || err) });
  }
};
