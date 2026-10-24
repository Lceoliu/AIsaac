// Where the floor generators run. The local server answers /api/...; tools/build_site.py replaces
// this file with { backend: 'pyodide', pyodide: '<distribution URL>' } for the static site.
window.MAPGEN_CONFIG = { backend: 'server' };
