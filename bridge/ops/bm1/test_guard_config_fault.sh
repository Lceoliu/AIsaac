#!/bin/sh
# train_guard.sh's cause config_fault (2026-11-18, the coordinator): two attempts in a row that end the same way early
# (rc != 0 within EARLY_S of the start, no progress row, the same rc and rank 0's last stderr line) stop the guard, no
# resume; a different last line, or an attempt that ran longer than EARLY_S, is retried as before. Every launch is a
# stub (no GPU, no docker, no game): the runner writes rank 0's stderr and exits 1.
#   usage: sh test_guard_config_fault.sh <scratch dir>
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
T=$(cd "$1" && pwd)
mkdir -p "$T/bin" "$T/g"
cp "$HERE/common.sh" "$HERE/train_guard.sh" "$T/g/"
printf 'BM1_IN=/bench\nBM1_PY=python3\nBM1_MODE=docker\nBM1_IMAGE=probe\nBM1_DOCKER_BASE=.\nBM1_GAME_RUNTIME=.\nBM1_GPUS_OK=0,1\nBM1_GPU_FORBID_REQUIRED=1\nBM1_GPU_FORBID_UUID=GPU-bad\nBM1_GPU_EXPECT=3\nBM1_TRAIN=$BM/train\nBM1_TRAIN_IN=/bench/train\nBM1_DISK=.\n' > "$T/g/bm1.env"
cat > "$T/bin/nvidia-smi" <<'EOF'
#!/bin/sh
case "$*" in
  *--query-gpu=index,uuid*) printf '0, GPU-good\n1, GPU-bad\n2, GPU-other\n' ;;
  -L) printf 'GPU 0: good\nGPU 1: bad\nGPU 2: other\n' ;;
  *--query-gpu=memory.used*) echo 0 ;;
esac
EOF
printf '#!/bin/sh\nexit 0\n' > "$T/bin/docker"
printf '#!/bin/sh\necho 999\n' > "$T/bin/df"
cat > "$T/bin/python3" <<'EOF'
#!/bin/sh
case "$*" in *'float(sys.argv[1] or 0)'*) echo 1 ;; esac
EOF
for c in journalctl dmesg; do printf '#!/bin/sh\nexit 0\n' > "$T/bin/$c"; done
# the runner: rank 0's stderr -- the same line every attempt (MSG=same) or one per attempt (MSG=vary) -- after SLOW s
cat > "$T/g/run.sh" <<'EOF'
#!/bin/sh
n=$(cat "$RD/.n" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$RD/.n"
sleep "${SLOW:-0}"
{ echo 'Traceback (most recent call last):'; echo '  File "train_tok.py", line 1'
  if [ "$MSG" = same ]; then echo 'ValueError: TokPolicy(frame=v3) needs no item_tokens'; else echo "OSError: transient $n"; fi; } > "$RD/probe.stderr" 2>&1
mv "$RD/probe.stderr" "$RD/probe-s0.stderr"
exit 1
EOF
chmod +x "$T/bin/"* "$T/g/run.sh"
printf 'LANE_GPUS=0\nLANE_CPUSET=\nLANE_MEMS=\nLANE_LEARNER_CPUS=\nLANE_DRIVER_CPUS=\nLANE_MEM=1g\nLANE_NAME=probe\nLANE_PORT=41000\n' > "$T/lane.env"
export PATH="$T/bin:$PATH"
case_() {   # <MSG> <EARLY_S> <SLOW> <expected attempts> <config_fault expected 1/0>
  rm -rf "$T/g/train"
  RD=$T/g/train/probe; mkdir -p "$RD"
  env -u GPUS_OK BM="$T/g" RD="$RD" MSG=$1 EARLY_S=$2 SLOW=$3 NOLOCK=1 HOURS=1 BACKOFF=1 MAX_RETRY=2 \
    timeout 120 sh "$T/g/train_guard.sh" probe tree "$T/lane.env" > "$T/out" 2>&1 || true
  a=$(grep -c ' attempt [0-9]*: ' "$RD/guard.log" || true)
  f=$(grep -c 'cause config_fault' "$RD/guard.log" || true)
  ok=0; if [ "$5" = 1 ]; then [ "$f" -ge 1 ] && ok=1; else [ "$f" = 0 ] && ok=1; fi
  [ "$a" = "$4" ] && [ $ok = 1 ] || {
    cat "$RD/guard.log"; echo "test_guard_config_fault: MSG $1 EARLY_S $2: attempts $a (want $4), config_fault lines $f (want $5)"; exit 1; }
  grep -q '(early: rc=1, ' "$RD/guard.log" || [ "$2" = 0 ] || { cat "$RD/guard.log"; exit 1; }
}
case_ same 300 0 2 1    # the same early end twice: stopped after the second attempt
case_ vary 300 0 3 0    # different last lines: retried until cause other's MAX_RETRY
case_ same 0 0 3 0      # not early (EARLY_S 0): retried as before
echo 'test_guard_config_fault: PASS'
