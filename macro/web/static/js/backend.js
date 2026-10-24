// Where the floor generators run: the local Python server (api/<method>) or, on the static site,
// Pyodide in a Web Worker (js/worker.js). Both answer the same methods (web/api.py METHODS).
// The worker starts loading as soon as the page does; `game` is downloaded along with it.
const CONFIG = window.MAPGEN_CONFIG || { backend: 'server' };

export function createBackend(onProgress, game = null) {
  return CONFIG.backend === 'pyodide' ? pyodideBackend(onProgress, game) : serverBackend();
}

function serverBackend() {
  return {
    kind: 'server',
    async call(method, params = {}) {
      const res = await fetch(`api/${method}?${new URLSearchParams(params)}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || res.statusText);
      return data;
    },
  };
}

function pyodideBackend(onProgress, game) {
  const worker = new Worker('js/worker.js', { type: 'module' });
  const pending = new Map();
  let next = 1;
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'progress') {
      if (onProgress) onProgress(m);
      return;
    }
    const p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.error) p.reject(new Error(m.error));
    else p.resolve(m.result);
  };
  worker.onerror = (e) => {
    for (const p of pending.values()) p.reject(new Error(e.message || '运行环境加载失败'));
    pending.clear();
  };
  worker.postMessage({ type: 'init', pyodide: CONFIG.pyodide, game });
  return {
    kind: 'pyodide',
    call(method, params = {}) {
      const id = next++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ type: 'call', id, method, params });
      });
    },
  };
}
