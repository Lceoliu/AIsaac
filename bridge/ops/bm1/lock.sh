# bm1 lane lock (sourced). take "<purpose>": atomic (noclobber) LOCK_node0 with "<session> <time> <purpose>", waiting
# while another holder has it; give: removes it if it is ours. LOCK=$BM/LOCK_node0 by default; SESSION names us.
LOCK=${LOCK:-$BM/LOCK_node0}
SESSION=${SESSION:-bm1-a1}
take() {
  while ! ( set -C; echo "$SESSION $(date '+%F %T') $1" > "$LOCK" ) 2>/dev/null; do sleep 30; done
  LOCKLINE=$(cat "$LOCK")
}
give() { [ "$(cat "$LOCK" 2>/dev/null)" = "${LOCKLINE:-}" ] && rm -f "$LOCK"; LOCKLINE=; }
# capmode <tree>: the graph capture mode a run of <tree> uses: ISAAC_CAPTURE_MODE from ENVS when given, else the tree's
# default (block_training.CAPTURE_MODE's), for the bm1.log lines
capmode() {
  m=$(echo "${ENVS:-}" | grep -o 'ISAAC_CAPTURE_MODE=[a-z_]*' | cut -d= -f2)
  [ -n "$m" ] || m=$(grep -o "ISAAC_CAPTURE_MODE', '[a-z_]*'" $BM/trees/$1/bridge/python/isaac_bridge/block_training.py 2>/dev/null | cut -d"'" -f3)
  echo "${m:-global (no switch in this tree)}"
}
