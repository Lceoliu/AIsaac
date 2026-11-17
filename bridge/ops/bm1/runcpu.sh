#!/bin/sh
# bm1: run.sh's CPU-only variant (2026-11-16, the coordinator: GPUs 4-7 all held by another tenant; CPU work only).
# One command in a container $BM1_NAME-cpu-<purpose> of $BM1_IMAGE with NO GPU at all: no --runtime=nvidia, no NVIDIA_*
# variables, CUDA_VISIBLE_DEVICES empty -- so no gpus_free check. Every other docker option as run.sh's docker mode
# (--rm --init, --network none, --read-only, /tmp tmpfs, --pids-limit, the caller's uid, /bench, the game runtime
# read-only, no proxy); no added privilege (no --privileged / --cap-add / seccomp change). Limits: --cpus 8 (on the
# caller's CPUSET when given), --memory 24g at most, torch / OMP threads 8 (BM1_CPU_THREADS). One such container at a
# time (another $BM1_NAME-cpu-* running: refused, exit 5). Docker mode only.
#   usage: [RUNENV="K=V ..."] runcpu.sh <purpose> <command> [args...]
set -eu
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
P=$1; shift
[ "$BM1_MODE" = docker ] || { echo "runcpu.sh: docker mode only" >&2; exit 2; }
# (2026-11-17, the coordinator: GPU fault -- nothing of ours touches the NVIDIA driver; this container maps no GPU, so
#  no preflight / nvidia-smi here; bm1/STOP_QUEUE (a GPU stop) is logged, not obeyed, for this CPU-only entry)
[ -f $BM/STOP_QUEUE ] && log "runcpu $P: STOP_QUEUE present ($(cat $BM/STOP_QUEUE)); CPU-only container started anyway (no GPU, no driver call)"
if docker ps --format '{{.Names}}' | grep -q "^$BM1_NAME-cpu-"; then
  log "runcpu $P: another CPU-only container runs ($(docker ps --format '{{.Names}}' | grep "^$BM1_NAME-cpu-" | tr '\n' ' ')): not started"; exit 5
fi
MEMC=${BM1_MEM:-24g}
case $MEMC in *g) [ "${MEMC%g}" -le 24 ] || MEMC=24g ;; *) MEMC=24g ;; esac
TH=${BM1_CPU_THREADS:-8}
KV="BM1_IN=$BM1_IN BM1_PY=$BM1_PY $(echo "${DOCK_ENV:-}" | sed 's/-e //g') ${RUNENV:-}"
mkdir -p "$BM/home"
EV=""; for kv in $KV; do EV="$EV -e $kv"; done
PP=${PYTHONPATH_EXTRA:-$BM1_PYTHONPATH}
lock_container "$BM1_NAME-cpu-$P"
log "runcpu $P: CPU-only container (cpus 8, memory $MEMC, threads $TH, cpuset ${BM1_CPUSET:-any})"
exec docker run --rm --init --name "$BM1_NAME-cpu-$P" --network none \
  --cpus 8 ${BM1_CPUSET:+--cpuset-cpus "$BM1_CPUSET"} ${BM1_MEMS:+--cpuset-mems "$BM1_MEMS"} \
  --memory "$MEMC" --memory-swap "$MEMC" --shm-size 8g --pids-limit "$BM1_PIDS" \
  --log-opt max-size=10m --log-opt max-file=2 --read-only --tmpfs /tmp:rw,exec,size=4g \
  --user "$(id -u):$(id -g)" \
  -v "$BM1_DOCKER_BASE:/bench" -v "$BM1_GAME_RUNTIME:/bench/home/isaac-abplus/runtime:ro" \
  -e CUDA_VISIBLE_DEVICES= \
  -e HOME=$BM1_IN/home -e ABP_HOME=$BM1_IN/home/isaac-abplus -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=$TH -e MKL_NUM_THREADS=$TH -e OPENBLAS_NUM_THREADS=$TH -e PYTHONPATH="$PP" \
  -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= -e all_proxy= \
  $EV --entrypoint /bin/bash "$BM1_IMAGE" -lc 'exec "$@"' run "$@"
