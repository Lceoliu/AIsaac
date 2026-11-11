#!/bin/sh
# bm1 (kept for older callers): run.sh under its old name. Everything host-specific comes from bm1.env.
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
exec "$BM/run.sh" "$@"
