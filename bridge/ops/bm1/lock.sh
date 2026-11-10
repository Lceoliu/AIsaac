# bm1 lane lock (sourced). take "<purpose>": atomic (noclobber) LOCK_node0 with "<session> <time> <purpose>", waiting
# while another holder has it; give: removes it if it is ours. LOCK=$BM/LOCK_node0 by default; SESSION names us.
LOCK=${LOCK:-$BM/LOCK_node0}
SESSION=${SESSION:-bm1-a1}
take() {
  while ! ( set -C; echo "$SESSION $(date '+%F %T') $1" > "$LOCK" ) 2>/dev/null; do sleep 30; done
  LOCKLINE=$(cat "$LOCK")
}
give() { [ "$(cat "$LOCK" 2>/dev/null)" = "${LOCKLINE:-}" ] && rm -f "$LOCK"; LOCKLINE=; }
