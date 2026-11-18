#!/bin/sh
# Test the actual runner without external device queries or container launches.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
T=$(cd "$1" && pwd)
mkdir -p "$T/bin"
cp "$HERE/common.sh" "$HERE/run.sh" "$T/"
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/bin/sh
case "$*" in
  *--query-gpu=index,uuid*) [ "${NO_MAP:-0}" = 0 ] || exit 1; printf '%s\n' "$DEV_MAP" ;;
  -L) printf 'GPU 0: a\nGPU 1: b\nGPU 2: c\n' ;;
  *--query-gpu=memory.used*) echo 0 ;;
esac
EOF
cat > "$T/bin/docker" <<'EOF'
#!/bin/sh
[ "$1" = run ] || exit 1
printf '%s\n' "$@" > "$BM/started"
EOF
cat > "$T/bin/setsid" <<'EOF'
#!/bin/sh
printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES" > "$BM/started"
EOF
for cmd in journalctl dmesg; do printf '#!/bin/sh\nexit 0\n' > "$T/bin/$cmd"; done
chmod +x "$T/bin/"*
cat > "$T/bm1.env" <<'EOF'
BM1_IN=$BM
BM1_MODE=${MODE:-docker}
BM1_PY=python3
BM1_IMAGE=fixture
BM1_GAME_RUNTIME=.
BM1_DOCKER_BASE=.
BM1_GPUS_OK=${ALLOW:-0,1,2}
BM1_TIMED_GPUS_OK=${TIMED:-}
BM1_GPU_FORBID_REQUIRED=${REQUIRED:-1}
BM1_GPU_FORBID_UUID=${DENY_UUID-GPU-bad}
BM1_GPU_FORBID=${DENY_INDEX:-}
EOF
export BM=$T PATH="$T/bin:$PATH"
DEV_MAP=$(printf '0, GPU-a\n1, GPU-bad\n2, GPU-c'); export DEV_MAP
run() {
  expected=$1; shift
  if [ -e "$T/started" ]; then mv "$T/started" "$T/started.previous"; fi
  actual=0; sh "$T/run.sh" probe "$1" true > "$T/out" 2>&1 || actual=$?
  [ "$expected" = "$actual" ] || { cat "$T/out"; echo "expected $expected, got $actual"; return 1; }
  if [ "$expected" = 0 ]; then
    [ -f "$T/started" ]
  else
    [ ! -e "$T/started" ]
  fi
}
run 0 2,0; grep -qx 'NVIDIA_VISIBLE_DEVICES=GPU-c,GPU-a' "$T/started"
MODE=direct run 0 0,2; grep -qx 'CUDA_VISIBLE_DEVICES=GPU-a,GPU-c' "$T/started"
run 5 0,1
run 5 GPU-bad
DENY_UUID= DENY_INDEX=1 run 5 GPU-bad
run 0 GPU-c
run 5 all
run 5 ''
run 5 0,,2
NO_MAP=1 run 5 0
DENY_UUID= run 5 0
ALLOW=2 run 2 0
RUNENV='NVIDIA_VISIBLE_DEVICES=all' run 2 0
DOCK_ENV='-e CUDA_VISIBLE_DEVICES=1' run 2 0
MODE=direct RUNENV='CUDA_VISIBLE_DEVICES=1' run 2 0
MODE=direct RUNENV='NVIDIA_VISIBLE_DEVICES=all' run 2 0
# A caller's GPUS_OK narrows the lane (2026-11-18, the coordinator) ...
GPUS_OK=2 run 0 2; grep -qx 'NVIDIA_VISIBLE_DEVICES=GPU-c' "$T/started"
GPUS_OK=2 run 2 0
GPUS_OK=0,2 run 0 2,0; grep -qx 'NVIDIA_VISIBLE_DEVICES=GPU-c,GPU-a' "$T/started"
GPUS_OK=GPU-c run 2 2                 # spelled otherwise than bm1.env's lane: dropped (fails closed)
# ... and never widens it
ALLOW=0 GPUS_OK=0,2 run 2 2
ALLOW=0 GPUS_OK=0,2 run 0 0
ALLOW=0 GPUS_OK=2 run 2 0             # nothing in common: every launch refused
grep -q 'no GPU allowed on this lane' "$T/out"
REQUIRED=0 DENY_UUID= NO_MAP=1 ALLOW=0 GPUS_OK=0,2 run 2 2
# bm1.env's timed GPUs belong to the lane too, when the caller names them (accept_g.sh, shardk.sh); never by default
ALLOW=0 TIMED=2 GPUS_OK=0,2 run 0 0,2; grep -qx 'NVIDIA_VISIBLE_DEVICES=GPU-a,GPU-c' "$T/started"
ALLOW=0 TIMED=2 run 2 2
ALLOW=0 TIMED=1,2 GPUS_OK=0,1,2 run 5 1   # the forbidden device stays forbidden
DEV_MAP=$(printf '0, GPU-bad\n1, GPU-a\n2, GPU-c')
run 5 0
run 0 1,2; grep -qx 'NVIDIA_VISIBLE_DEVICES=GPU-a,GPU-c' "$T/started"
REQUIRED=0 DENY_UUID= NO_MAP=1 run 0 1,2
grep -qx 'NVIDIA_VISIBLE_DEVICES=1,2' "$T/started"
echo 'test_gpu_launch: PASS'
