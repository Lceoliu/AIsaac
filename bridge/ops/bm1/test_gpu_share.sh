#!/bin/sh
# Test gpus_free's BM1_GPU_SHARE_WITH (2026-11-20, A1 / the coordinator; Codex round 27) through the actual runner, with
# stubs only: no device query, no container launch, no request ever reaches a real GPU (least of all the forbidden one
# -- in this fixture GPU 1 is the forbidden device). The evidence functions (share_proc_cgroup, share_on_host) are
# replaced in this test's own copy of common.sh only; the production file reads the host's /proc.
#   usage: test_gpu_share.sh <scratch dir>
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
T=$(cd "$1" && pwd)
rm -rf "$T/bin" "$T/proc" "$T/reports" "$T/containers.reg" "$T/queried"
mkdir -p "$T/bin"
cp "$HERE/common.sh" "$HERE/run.sh" "$T/"
cat >> "$T/common.sh" <<'EOF'
# (test_gpu_share.sh's stand-ins, in this copy only)
share_proc_cgroup() { cat "$FAKE_PROC/$1/cgroup"; }
share_on_host() { [ "${FAKE_IN_CONTAINER:-0}" = 0 ]; }
EOF
CID=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
OTHER=fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/bin/sh
dev=; prev=
for a in "$@"; do [ "$prev" = -i ] && dev=$a; prev=$a; done
k=$(printf '%s' "$dev" | tr -c 'a-zA-Z0-9\n' '_')
case "$*" in
  *--query-gpu=index,uuid*) printf '%s\n' "$DEV_MAP" ;;
  -L) printf 'GPU 0: a\nGPU 1: b\nGPU 2: c\n' ;;
  *--query-gpu=memory.used*) echo "$dev mem" >> "$BM/queried"; [ "${SMI_FAIL_MEM:-0}" = 0 ] || exit 124
     printf '%s\n' "${SMI_MEM:-100}" ;;
  *--query-compute-apps=pid*) echo "$dev apps" >> "$BM/queried"; [ "${SMI_FAIL_APPS:-0}" = 0 ] || exit 124
     eval "v=\${SMI_PIDS_$k-\${SMI_PIDS:-}}"; [ -z "$v" ] || printf '%s\n' "$v" ;;
esac
EOF
cat > "$T/bin/docker" <<'EOF'
#!/bin/sh
case "$1" in
  run) cf=; prev=; for a in "$@"; do [ "$prev" = --cidfile ] && cf=$a; prev=$a; done
       [ -z "$cf" ] || printf '%s' "$CID" > "$cf"
       printf '%s\n' "$@" > "$BM/started" ;;
  inspect) [ "${INSPECT_FAIL:-0}" = 0 ] || exit 1
           case " ${RUNNING_NAMES:-aisaac-bench-bm1-a1-train-seq2} " in *" $4 "*) ;; *) exit 1 ;; esac
           if [ "${STOPPED:-0}" = 1 ]; then echo ''; else echo "${INSPECT_ID:-$CID}"; fi ;;
  *) exit 1 ;;
esac
EOF
for cmd in journalctl dmesg; do printf '#!/bin/sh\nexit 0\n' > "$T/bin/$cmd"; done
chmod +x "$T/bin/"*
cg() { mkdir -p "$T/proc/$1"; printf '%s\n' "$2" > "$T/proc/$1/cgroup"; }
cg 101 "0::/docker/$CID"
cg 102 "0::/docker/$CID/nested/child"
cg 103 "0::/system.slice/docker-$CID.scope"
cg 104 "$(printf '12:pids:/docker/%s\n11:memory:/docker/%s' "$CID" "$CID")"
cg 999 "0::/docker/$OTHER"
cg 201 "0::/docker/${CID}abc"
cg 202 "0::/other/docker/$CID"
cg 203 "0::/docker/$(printf '%s' "$CID" | cut -c1-12)"
cg 204 "0::/docker/$OTHER/$CID"
cg 205 "$(printf '12:pids:/docker/%s\n11:memory:/' "$CID")"
cg 206 ""
cat > "$T/bm1.env" <<'EOF'
BM1_IN=$BM
BM1_MODE=${MODE:-docker}
BM1_PY=python3
BM1_IMAGE=fixture
BM1_GAME_RUNTIME=.
BM1_DOCKER_BASE=.
BM1_GPUS_OK=${ALLOW:-0,1,2}
BM1_GPU_FORBID_REQUIRED=1
BM1_GPU_FORBID_UUID=GPU-bad
EOF
SHARE=aisaac-bench-bm1-a1-train-seq2
printf '2026-11-20 11:51:12 %s %s\n' "$SHARE" "$CID" > "$T/containers.reg"
DEV_MAP=$(printf '0, GPU-a\n1, GPU-bad\n2, GPU-c')
export BM=$T PATH="$T/bin:$PATH" DEV_MAP CID FAKE_PROC="$T/proc"
run() {   # run <expected rc> <gpu list> [reason expected on stderr], with the caller's environment
  expected=$1; gpus=$2; why=${3:-}
  rm -f "$T/started" "$T/queried"
  actual=0; sh "$T/run.sh" probe "$gpus" true > "$T/out" 2>&1 || actual=$?
  [ "$expected" = "$actual" ] || { cat "$T/out"; echo "expected $expected, got $actual ($gpus)"; exit 1; }
  if [ "$expected" = 0 ]; then [ -f "$T/started" ]; else [ ! -e "$T/started" ]; fi
  [ -z "$why" ] || grep -q -- "$why" "$T/out" || { cat "$T/out"; echo "reason '$why' not given"; exit 1; }
}
# --- the switch empty: exactly as before
run 0 2
SMI_PIDS=101 run 4 2
SMI_MEM=600 run 4 2
SMI_MEM=600 BM1_GPU_FREE_MB=1000 SMI_PIDS=101 run 4 2
# --- every container run.sh starts is registered (docker's --cidfile)
run 0 2
i=0; until grep -q " aisaac-bench-bm1-a1-probe $CID\$" "$T/containers.reg"; do i=$((i+1)); [ $i -lt 50 ] || { echo 'not registered'; exit 1; }; sleep 0.1; done
# --- set: the container's own processes (cgroupfs, nested below it, systemd's scope, every v1 line) -- started
SMI_PIDS="$(printf '101\n102\n103\n104')" BM1_GPU_SHARE_WITH=$SHARE run 0 2
grep -q "GPU GPU-c SHARED with $SHARE (memory used 100 MiB < 500, 4 compute process(es), all its own)" "$T/reports/bm1.log"
[ "$(grep -c 'GPU-c apps' "$T/queried")" = 2 ]                           # (checked, and checked again before the start)
BM1_GPU_SHARE_WITH=$SHARE run 0 2                                        # (no process at all: fine too)
SMI_PIDS=101 BM1_GPU_SHARE_WITH=$SHARE run 0 0,2
# --- two cards: the first its own, the second with someone else's process -- refused, both queried, the second named
SMI_PIDS_GPU_a=101 SMI_PIDS_GPU_c=999 BM1_GPU_SHARE_WITH=$SHARE run 4 0,2 "GPU GPU-c: process 999 is not inside"
grep -q 'GPU-a apps' "$T/queried" && grep -q 'GPU-c apps' "$T/queried"
# --- not the container's: another container, a lookalike id, an ancestor, a short id, a mixed v1 file, no cgroup
for p in 999 201 202 203 204 205 206 555; do
  SMI_PIDS=$(printf '101\n%s' $p) BM1_GPU_SHARE_WITH=$SHARE run 4 2 "process $p"
done
# --- process lists that do not parse: never read as empty
SMI_PIDS='10 1' BM1_GPU_SHARE_WITH=$SHARE run 4 2 'a process id does not parse (10 1)'
SMI_PIDS='[N/A]' BM1_GPU_SHARE_WITH=$SHARE run 4 2 'a process id does not parse'
SMI_PIDS='101*' BM1_GPU_SHARE_WITH=$SHARE run 4 2 'a process id does not parse'
# --- queries that fail or time out, a memory reading that does not parse
SMI_PIDS=101 SMI_FAIL_APPS=1 BM1_GPU_SHARE_WITH=$SHARE run 4 2 'nvidia-smi failed (processes)'
SMI_PIDS=101 SMI_FAIL_MEM=1 BM1_GPU_SHARE_WITH=$SHARE run 4 2 'nvidia-smi failed (memory)'
SMI_PIDS=101 SMI_MEM='N/A' BM1_GPU_SHARE_WITH=$SHARE run 4 2 'memory used does not parse'
# --- the memory threshold still holds
SMI_PIDS=101 SMI_MEM=600 BM1_GPU_SHARE_WITH=$SHARE run 4 2 'memory used 600 MiB >= 500'
SMI_PIDS=101 SMI_MEM=600 BM1_GPU_FREE_MB=8000 BM1_GPU_SHARE_WITH=$SHARE run 0 2
# --- the container: not ours by name, a bad name, not there, not running, not registered (same prefix), a short id
SMI_PIDS=101 BM1_GPU_SHARE_WITH=other-container run 4 2 'not an aisaac-bench-'
SMI_PIDS=101 BM1_GPU_SHARE_WITH='aisaac-bench-x;true' run 4 2 'invalid container name'
SMI_PIDS=101 BM1_GPU_SHARE_WITH=aisaac-bench-bm1-a1-gone run 4 2 'docker inspect failed'
SMI_PIDS=101 INSPECT_FAIL=1 BM1_GPU_SHARE_WITH=$SHARE run 4 2 'docker inspect failed'
SMI_PIDS=101 STOPPED=1 BM1_GPU_SHARE_WITH=$SHARE run 4 2 'not running'
SMI_PIDS=101 RUNNING_NAMES="$SHARE aisaac-bench-bm1-a1-train-seq9" BM1_GPU_SHARE_WITH=aisaac-bench-bm1-a1-train-seq9 \
  run 4 2 'is not registered'
SMI_PIDS=101 INSPECT_ID=0123456789ab BM1_GPU_SHARE_WITH=$SHARE run 4 2 'not a full 64-hex id'
SMI_PIDS=101 INSPECT_ID=$OTHER BM1_GPU_SHARE_WITH=$SHARE run 4 2 'is not registered'
# --- not on the host, not docker mode
SMI_PIDS=101 FAKE_IN_CONTAINER=1 BM1_GPU_SHARE_WITH=$SHARE run 4 2 'not on the host itself'
SMI_PIDS=101 MODE=direct BM1_GPU_SHARE_WITH=$SHARE run 4 2 'docker mode only'
# --- an empty device list (gpus_shared on its own: run.sh's gpu_forbid refuses one before)
rc=0; BM1_GPU_SHARE_WITH=$SHARE sh -c ". '$T/common.sh'; gpus_shared ''" > "$T/out" 2>&1 || rc=$?
[ "$rc" = 4 ] && grep -q 'empty device list' "$T/out"
# --- bm1.log not writable: still refused, the reason on stderr
chmod a-w "$T/reports/bm1.log"
SMI_PIDS=999 BM1_GPU_SHARE_WITH=$SHARE run 4 2 'process 999 is not inside'
chmod u+w "$T/reports/bm1.log"
# --- the forbidden device stays forbidden and the lane's list stays the lane's (gpu_forbid first)
SMI_PIDS=101 BM1_GPU_SHARE_WITH=$SHARE run 5 1
SMI_PIDS=101 BM1_GPU_SHARE_WITH=$SHARE run 5 GPU-bad
SMI_PIDS=101 BM1_GPU_SHARE_WITH=$SHARE run 5 1,2
SMI_PIDS=101 ALLOW=0,1 BM1_GPU_SHARE_WITH=$SHARE run 2 2
echo 'test_gpu_share: PASS'
