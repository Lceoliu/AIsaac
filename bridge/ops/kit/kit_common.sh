# kit common (sourced by kit_build.sh, kit_install.sh, kit_doctor.sh). Loads the host file hosts/<hostname>.env
# (or $KIT_ENV), defines klog / kdie / kit_lock / kit_unlock / kit_trash / kit_manifest_get.
# The lock is bm1's reader-writer lock (bridge/ops/bm1/common.sh, sourced unchanged): every heavy kit step takes it
# SHARED as session "kit", so it never overlaps a timed run (exclusive holder); bm1's log lines of the kit go to
# <ROOT>/kit/lockshim/reports/bm1.log, never into a session's reports.
KITDIR=${KITDIR:-$(cd "$(dirname "$0")" && pwd)}
KIT_ENV=${KIT_ENV:-$KITDIR/hosts/$(hostname).env}
[ -f "$KIT_ENV" ] || { echo "kit: no host file $KIT_ENV (add hosts/$(hostname).env)" >&2; exit 2; }
. "$KIT_ENV"
: "${KIT_ROOT:?}" "${KIT_HOST:?}"
KIT_LOG=$KIT_ROOT/kit/kit.log
mkdir -p "$KIT_ROOT/kit"

klog() { echo "$(date -u '+%F %T') [$KIT_HOST] $*" | tee -a "$KIT_LOG"; }
kdie() { klog "FAILED: $*"; exit 1; }

# pinned: run a command on the kit's CPUs, low CPU and I/O priority
pinned() {
  if [ -n "${KIT_CPUSET:-}" ]; then nice -n 10 ionice -c2 -n7 taskset -c "$KIT_CPUSET" "$@"; else nice -n 10 "$@"; fi
}

# KIT_LOCKDIRS: one or more bm1 lock directories (space-separated; KIT_LOCKDIR = one). kit_lock takes the SHARED lock
# in every one of them: it waits until none has a live exclusive file X (a timed run holding or queued), then takes them
# in order; when an X appears in a later directory meanwhile, it releases what it took and waits again, so the kit never
# sits on one directory's shared lock while waiting for another's timed run.
_x_live() {   # _x_live <dir>: an X whose holder pid is alive
  [ -f "$1/X" ] || return 1
  p=$(sed -n 's/^pid \([0-9]*\).*/\1/p' "$1/X")
  [ -n "$p" ] && kill -0 "$p" 2>/dev/null
}
KIT_HELD_F=(); KIT_HELD_L=()
kit_lock() {   # kit_lock "<purpose>"
  dirs=${KIT_LOCKDIRS:-${KIT_LOCKDIR:-}}
  if [ -z "$dirs" ]; then klog "lock: no lock directory on this host, not locking ($1)"; return 0; fi
  KBM=$KIT_ROOT/kit/lockshim
  mkdir -p "$KBM"
  printf 'BM1_LOCKDIR=%s\nBM1_SESSION=kit\nBM1_TRASH=%s\n' "${dirs%% *}" "${KIT_TRASH:-$KIT_ROOT/kit/trash}" > "$KBM/bm1.env"
  BM=$KBM
  . "$KITDIR/../bm1/common.sh"
  # bm1/common.sh (72e0ac74...) bug: _stale_sweep loops with the global variable f, which take_shared uses for its own
  # file name S.<session>.<pid>; after the sweep f is the last S.* entry (or the literal "S.*"), so every shared holder
  # writes the SAME file and may overwrite another holder's. Same protocol, loop variable renamed (this process only;
  # bm1/common.sh itself is not edited -- reported to the coordinator).
  _stale_sweep() {
    for _kf in "$BM1_LOCKDIR"/X "$BM1_LOCKDIR"/S.*; do
      [ -f "$_kf" ] || continue
      _kp=$(sed -n 's/^pid \([0-9]*\).*/\1/p' "$_kf")
      if [ -z "$_kp" ] || ! kill -0 "$_kp" 2>/dev/null; then
        log "lock: stale $(basename "$_kf") removed (holder pid ${_kp:-?} gone): $(head -1 "$_kf")"
        rm -f "$_kf"
      fi
    done
  }
  klog "lock: waiting for the shared lock in [$dirs] ($1); holders now: $(for d in $dirs; do cat "$d"/* 2>/dev/null | grep -v '^pid'; done | tr '\n' ';')"
  t0=$(date +%s)
  while :; do
    busy=1
    while [ $busy = 1 ]; do busy=0; for d in $dirs; do _x_live "$d" && busy=1; done; [ $busy = 1 ] && sleep 30; done
    ok=1
    for d in $dirs; do
      mkdir -p "$d"
      if _x_live "$d"; then ok=0; break; fi
      BM1_LOCKDIR=$d
      take_shared "kit $1"
      KIT_HELD_F+=("$LOCKFILE"); KIT_HELD_L+=("$LOCKLINE")
    done
    [ $ok = 1 ] && break
    klog "lock: an exclusive lock appeared in $d meanwhile: releasing and waiting again"
    kit_unlock
    sleep 30
  done
  klog "lock: shared taken in [$dirs] after $(( $(date +%s) - t0 )) s: ${KIT_HELD_F[*]}"
}
kit_unlock() {
  [ ${#KIT_HELD_F[@]} -gt 0 ] || return 0
  for i in "${!KIT_HELD_F[@]}"; do LOCKFILE=${KIT_HELD_F[$i]}; LOCKLINE=${KIT_HELD_L[$i]}; give; done
  klog "lock: released ${KIT_HELD_F[*]}"
  KIT_HELD_F=(); KIT_HELD_L=()
}

kit_trash() {   # move into $KIT_TRASH/<time>/ (nothing is deleted)
  [ -e "$1" ] || return 0
  d=${KIT_TRASH:-$KIT_ROOT/kit/trash}/$(date -u +%m%d-%H%M%S)
  mkdir -p "$d" && mv "$1" "$d/" && klog "moved to trash: $1 -> $d/"
}

kit_manifest() { echo "$KIT_ROOT/kit/$1.manifest.json"; }   # $1 = kit-YYYYMMDD.N
# kit_manifest_get <version> <dotted.key>: one value of the version's manifest entry
kit_manifest_get() {
  python3 -c 'import json,sys
v=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): v=v[k]
print(v)' "$(kit_manifest "$1")" "$2"
}
