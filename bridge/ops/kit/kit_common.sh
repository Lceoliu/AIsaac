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

kit_lock() {   # kit_lock "<purpose>": the shared bm1 lock (waits while a timed run holds or wants it)
  if [ -z "${KIT_LOCKDIR:-}" ]; then klog "lock: no KIT_LOCKDIR on this host, not locking ($1)"; return 0; fi
  KBM=$KIT_ROOT/kit/lockshim
  mkdir -p "$KBM"
  printf 'BM1_LOCKDIR=%s\nBM1_SESSION=kit\nBM1_TRASH=%s\n' "$KIT_LOCKDIR" "${KIT_TRASH:-$KIT_ROOT/kit/trash}" > "$KBM/bm1.env"
  klog "lock: waiting for the shared lock ($1); holders now: $(cat "$KIT_LOCKDIR"/* 2>/dev/null | grep -v '^pid' | tr '\n' ';')"
  BM=$KBM
  . "$KITDIR/../bm1/common.sh"
  take_shared "kit $1"
  klog "lock: shared taken: $LOCKLINE"
}
kit_unlock() {
  if [ -n "${LOCKFILE:-}" ]; then give; klog "lock: released"; fi
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
