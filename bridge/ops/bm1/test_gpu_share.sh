#!/bin/sh
# Test gpus_free's BM1_GPU_SHARE_WITH (2026-11-20, A1 / the coordinator) through the actual runner, with stubs only:
# no device query, no container launch, and no request ever reaches a real GPU (least of all the forbidden one -- in
# this fixture GPU 1 is the forbidden device).   usage: test_gpu_share.sh <scratch dir>
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
T=$(cd "$1" && pwd)
rm -rf "$T/bin" "$T/proc" "$T/reports"
mkdir -p "$T/bin"
cp "$HERE/common.sh" "$HERE/run.sh" "$T/"
CID=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
OTHER=fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/bin/sh
case "$*" in
  *--query-gpu=index,uuid*) printf '%s\n' "$DEV_MAP" ;;
  -L) printf 'GPU 0: a\nGPU 1: b\nGPU 2: c\n' ;;
  *--query-gpu=memory.used*) [ "${SMI_FAIL_MEM:-0}" = 0 ] || exit 124; printf '%s\n' "${SMI_MEM:-100}" ;;
  *--query-compute-apps=pid*) [ "${SMI_FAIL_APPS:-0}" = 0 ] || exit 124; [ -z "${SMI_PIDS:-}" ] || printf '%s\n' $SMI_PIDS ;;
esac
EOF
cat > "$T/bin/docker" <<'EOF'
#!/bin/sh
case "$1" in
  run) printf '%s\n' "$@" > "$BM/started" ;;
  inspect) [ "$4" = "${RUNNING_NAME:-aisaac-bench-bm1-a1-train-seq2}" ] || exit 1
           [ "${STOPPED:-0}" = 1 ] && echo '' || echo "$CID" ;;
  *) exit 1 ;;
esac
EOF
for cmd in journalctl dmesg; do printf '#!/bin/sh\nexit 0\n' > "$T/bin/$cmd"; done
chmod +x "$T/bin/"*
for p in 101 102; do mkdir -p "$T/proc/$p"; printf '0::/docker/%s\n' "$CID" > "$T/proc/$p/cgroup"; done
mkdir -p "$T/proc/999"; printf '0::/docker/%s\n' "$OTHER" > "$T/proc/999/cgroup"
cat > "$T/bm1.env" <<'EOF'
BM1_IN=$BM
BM1_MODE=docker
BM1_PY=python3
BM1_IMAGE=fixture
BM1_GAME_RUNTIME=.
BM1_DOCKER_BASE=.
BM1_GPUS_OK=${ALLOW:-0,1,2}
BM1_GPU_FORBID_REQUIRED=1
BM1_GPU_FORBID_UUID=GPU-bad
EOF
DEV_MAP=$(printf '0, GPU-a\n1, GPU-bad\n2, GPU-c')
export BM=$T PATH="$T/bin:$PATH" DEV_MAP CID BM1_PROC="$T/proc"
SHARE=aisaac-bench-bm1-a1-train-seq2
run() {   # run <expected rc> <gpu list>, with the caller's environment
  expected=$1; shift
  rm -f "$T/started"
  actual=0; sh "$T/run.sh" probe "$1" true > "$T/out" 2>&1 || actual=$?
  [ "$expected" = "$actual" ] || { cat "$T/out"; echo "expected $expected, got $actual"; exit 1; }
  if [ "$expected" = 0 ]; then [ -f "$T/started" ]; else [ ! -e "$T/started" ]; fi
}
# the switch empty: exactly as before -- a card with a process, or over the memory threshold, refused
run 0 2
SMI_PIDS=101 run 4 2
SMI_MEM=600 run 4 2
SMI_MEM=600 BM1_GPU_FREE_MB=1000 SMI_PIDS=101 run 4 2
# set: every process on the card the container's -- started, the share logged
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS='101 102' run 0 2
grep -q "GPU GPU-c SHARED with $SHARE" "$T/reports/bm1.log"
BM1_GPU_SHARE_WITH=$SHARE run 0 2                                # (no process at all: fine too)
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS='101 102' run 0 0,2
# anyone else's process, a process id that does not parse: refused
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS='101 999' run 4 2
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS='101 [N/A]' run 4 2
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS='101 555' run 4 2             # (no /proc entry: not its own)
# nvidia-smi failing or timing out, a memory reading that does not parse: refused
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 SMI_FAIL_APPS=1 run 4 2
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 SMI_FAIL_MEM=1 run 4 2
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 SMI_MEM='N/A' run 4 2
# the memory threshold still holds
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 SMI_MEM=600 run 4 2
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 SMI_MEM=600 BM1_GPU_FREE_MB=8000 run 0 2
# the container: not aisaac-bench-*, a bad name, not there, not running -- refused
BM1_GPU_SHARE_WITH=other-container SMI_PIDS=101 run 4 2
BM1_GPU_SHARE_WITH='aisaac-bench-x;true' SMI_PIDS=101 run 4 2
BM1_GPU_SHARE_WITH=aisaac-bench-bm1-a1-gone SMI_PIDS=101 run 4 2
BM1_GPU_SHARE_WITH=$SHARE STOPPED=1 SMI_PIDS=101 run 4 2
grep -q 'GPU SHARE REFUSED' "$T/reports/bm1.log"
# the forbidden device stays forbidden and the lane's list stays the lane's, the switch set or not (gpu_forbid first)
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 run 5 1
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 run 5 GPU-bad
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 run 5 1,2
BM1_GPU_SHARE_WITH=$SHARE SMI_PIDS=101 ALLOW=0,1 run 2 2
echo 'test_gpu_share: PASS'
