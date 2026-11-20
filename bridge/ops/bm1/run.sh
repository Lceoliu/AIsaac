#!/bin/sh
# bm1: run one command on this lane's CPUs / memory node / GPU(s), host-independently (bm1.env):
#   usage: [RUNENV="K=V ..."] run.sh <purpose> <gpu[,gpu...]> <command> [args...]
#   (DOCK_ENV="-e K=V ..." of the older scripts is accepted too: its "-e" are dropped)
# The command sees BM1_IN (bm1's directory as the runner sees it), BM1_PY, HOME=$BM1_IN/home (every cache),
# ABP_HOME=$BM1_IN/home/isaac-abplus (run_instance.sh starts the game from $HOME/isaac-abplus), PYTHONPATH=$BM1_IN/pydeps
# (when it exists), OMP / MKL / OpenBLAS threads 1, no proxy; its GPUs as cuda:0, cuda:1, ... in the order given (UUIDs on guarded hosts).
#   BM1_MODE=docker: a container $BM1_NAME-<purpose> (default aisaac-bench-bm1-a1-<purpose>) of $BM1_IMAGE (--rm, --cpuset-cpus / --cpuset-mems,
#     --memory, --pids-limit, read-only root, /tmp tmpfs, --network none, the caller's uid, $BM1_DOCKER_BASE as /bench,
#     the game runtime read-only).
#   BM1_MODE=direct: the current system. CPU affinity by taskset (BM1_CPUSET), memory node by numactl --membind
#     (BM1_MEMS, when numactl exists), the GPUs by CUDA_VISIBLE_DEVICES, its own session (setsid). The memory limit
#     BM1_MEM: a cgroup v2 child with memory.max where the cgroup can be written, else memguard.py, which sums the
#     process tree's RSS every 2 s and, over the limit, logs to bm1.log and kills the tree (never silently).
#     No network isolation and no read-only root in this mode: the runtime's files must not be writable by the run.
set -eu
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}
. $BM/common.sh
P=$1; G=$2; shift 2
DEVICES=$(gpu_forbid "$G" "$P") || exit $?
preflight "$P"
gpus_free "$DEVICES"
KV="BM1_IN=$BM1_IN BM1_PY=$BM1_PY $(echo "${DOCK_ENV:-}" | sed 's/-e //g') ${RUNENV:-}"
# Do not let a later generic -e/export override the checked device selection.
set -f
for kv in $KV; do
  case "$kv" in
    CUDA_VISIBLE_DEVICES=*|NVIDIA_VISIBLE_DEVICES=*|CUDA_DEVICE_ORDER=*)
      printf 'run.sh: device selection must use the GPU argument, not %s\n' "${kv%%=*}" >&2; exit 2 ;;
    *=*)
      case "${kv%%=*}" in ''|[0-9]*|*[!a-zA-Z0-9_]*)
        echo "run.sh: invalid environment variable name" >&2; exit 2 ;;
      esac ;;
    *) printf 'run.sh: invalid environment assignment: %s\n' "$kv" >&2; exit 2 ;;
  esac
done
mkdir -p "$BM/home"
if [ "$BM1_MODE" = docker ]; then
  EV=""; for kv in $KV; do EV="$EV -e $kv"; done
  PP=${PYTHONPATH_EXTRA:-$BM1_PYTHONPATH}
  lock_container "$BM1_NAME-$P"   # (the lock this job runs under stays held while its container runs)
  # (2026-11-20, Codex round 27) BM1_GPU_SHARE_WITH: the occupancy checked again right before the start (narrows the
  #  window between the check and the start; no lease covers it -- common.sh's known deviation)
  [ -z "${BM1_GPU_SHARE_WITH:-}" ] || gpus_free "$DEVICES"
  # (2026-11-20, Codex round 27) every container this runner starts is registered: $BM/containers.reg, one line
  #  "<date> <time> <name> <full id>" from docker's --cidfile (the registry gpus_shared requires of a shared container)
  CIDF=$(mktemp -u "$R/.cid-$P-XXXXXX")
  ( i_=0
    while [ $i_ -lt 600 ]; do
      if [ -s "$CIDF" ]; then
        c_=$(cat "$CIDF")
        case "$c_" in *[!0-9a-f]*) ;; *) [ ${#c_} -eq 64 ] && \
          printf '%s %s %s\n' "$(date '+%F %T')" "$BM1_NAME-$P" "$c_" >> "$BM/containers.reg" ;; esac
        break
      fi
      sleep 0.2; i_=$((i_ + 1))
    done
    rm -f "$CIDF" ) &
  exec docker run --rm --init --cidfile "$CIDF" --name "$BM1_NAME-$P" --runtime=nvidia --network none \
    ${BM1_CPUSET:+--cpuset-cpus "$BM1_CPUSET"} ${BM1_MEMS:+--cpuset-mems "$BM1_MEMS"} \
    ${BM1_MEM:+--memory "$BM1_MEM" --memory-swap "$BM1_MEM"} --shm-size 8g --pids-limit "$BM1_PIDS" \
    --log-opt max-size=10m --log-opt max-file=2 --read-only --tmpfs /tmp:rw,exec,size=4g \
    --user "$(id -u):$(id -g)" \
    -v "$BM1_DOCKER_BASE:/bench" -v "$BM1_GAME_RUNTIME:/bench/home/isaac-abplus/runtime:ro" \
    -e NVIDIA_VISIBLE_DEVICES="$DEVICES" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HOME=$BM1_IN/home -e ABP_HOME=$BM1_IN/home/isaac-abplus -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONPATH="$PP" \
    -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= -e all_proxy= \
    $EV --entrypoint /bin/bash "$BM1_IMAGE" -lc 'exec "$@"' run "$@"
fi
# direct
export HOME=$BM1_IN/home ABP_HOME=$BM1_IN/home/isaac-abplus PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$DEVICES"
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
[ -d "$BM1_IN/pydeps" ] && export PYTHONPATH="${PYTHONPATH_EXTRA:-$BM1_PYTHONPATH}"
for kv in $KV; do export "$kv"; done
PRE=""
[ -n "$BM1_CPUSET" ] && PRE="taskset -c $BM1_CPUSET"
[ -n "$BM1_MEMS" ] && command -v numactl > /dev/null 2>&1 && PRE="numactl --membind=$BM1_MEMS $PRE"
setsid $PRE "$@" &
CH=$!
GU=
if [ -n "$BM1_MEM" ]; then
  python3 $BM/memguard.py "$CH" "$BM1_MEM" "$LOG" "$P" &
  GU=$!
fi
set +e
wait $CH
rc=$?
[ -n "$GU" ] && kill $GU 2>/dev/null
exit $rc
