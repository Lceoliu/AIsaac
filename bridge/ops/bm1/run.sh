#!/bin/sh
# bm1: run one command on this lane's CPUs / memory node / GPU(s), host-independently (bm1.env):
#   usage: [RUNENV="K=V ..."] run.sh <purpose> <gpu[,gpu...]> <command> [args...]
#   (DOCK_ENV="-e K=V ..." of the older scripts is accepted too: its "-e" are dropped)
# The command sees BM1_IN (bm1's directory as the runner sees it), BM1_PY, HOME=$BM1_IN/home (every cache),
# ABP_HOME=$BM1_IN/home/isaac-abplus (run_instance.sh starts the game from $HOME/isaac-abplus), PYTHONPATH=$BM1_IN/pydeps
# (when it exists), OMP / MKL / OpenBLAS threads 1, no proxy; its GPUs as cuda:0, cuda:1, ... in the order given.
#   BM1_MODE=docker: a container aisaac-bench-bm1-a1-<purpose> of $BM1_IMAGE (--rm, --cpuset-cpus / --cpuset-mems,
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
for g in $(echo "$G" | tr ',' ' '); do
  case ",$BM1_GPUS_OK," in *",$g,"*) ;; *) echo "run.sh: GPU $g is not on this lane ($BM1_GPUS_OK)" >&2; exit 2 ;; esac
done
preflight "$P"
gpus_free "$G"
KV="BM1_IN=$BM1_IN BM1_PY=$BM1_PY $(echo "${DOCK_ENV:-}" | sed 's/-e //g') ${RUNENV:-}"
mkdir -p "$BM/home"
if [ "$BM1_MODE" = docker ]; then
  EV=""; for kv in $KV; do EV="$EV -e $kv"; done
  PP=${PYTHONPATH_EXTRA:-$BM1_IN/pydeps}
  exec docker run --rm --init --name "aisaac-bench-bm1-a1-$P" --runtime=nvidia --network none \
    ${BM1_CPUSET:+--cpuset-cpus "$BM1_CPUSET"} ${BM1_MEMS:+--cpuset-mems "$BM1_MEMS"} \
    ${BM1_MEM:+--memory "$BM1_MEM" --memory-swap "$BM1_MEM"} --shm-size 8g --pids-limit "$BM1_PIDS" \
    --log-opt max-size=10m --log-opt max-file=2 --read-only --tmpfs /tmp:rw,exec,size=4g \
    --user "$(id -u):$(id -g)" \
    -v "$BM1_DOCKER_BASE:/bench" -v "$BM1_GAME_RUNTIME:/bench/home/isaac-abplus/runtime:ro" \
    -e NVIDIA_VISIBLE_DEVICES="$G" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e HOME=$BM1_IN/home -e ABP_HOME=$BM1_IN/home/isaac-abplus -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e PYTHONPATH="$PP" \
    -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= -e all_proxy= \
    $EV --entrypoint /bin/bash "$BM1_IMAGE" -lc 'exec "$@"' run "$@"
fi
# direct
export HOME=$BM1_IN/home ABP_HOME=$BM1_IN/home/isaac-abplus PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$G"
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= all_proxy=
[ -d "$BM1_IN/pydeps" ] && export PYTHONPATH="${PYTHONPATH_EXTRA:-$BM1_IN/pydeps}"
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
