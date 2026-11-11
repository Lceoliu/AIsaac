#!/bin/sh
# bm1 (8x3090 dgene): run one command in the aisaac-bench:20261019 image, on this lane's CPUs / NUMA node / GPU(s).
#   usage: dock.sh <purpose> <gpu[,gpu...]> <command> [args...]
#   container: aisaac-bench-bm1-a1-<purpose>, --rm, 64 GB, /bench = cpu-wide (the game runtime read-only), the
#   caller's uid, HOME=/bench/bm1/home (every cache: isaac-actor, learn-native, visible, Inductor, Triton).
#   PYTHONPATH=/bench/bm1/pydeps (packages the shared venv lacks or has at another version than the 4090; README).
#   Lane (environment, defaults = node0): CPUSET=0-11,24-35 MEMS=0 GPUS_OK=1,2,3; MEM=64g.
#   Lane (lane.sh: LANE=a1 node0 / LANE=cpp node1; environment overrides): LANE=cpp: container
#   aisaac-bench-bm1-cpp-<purpose>, HOME=/bench/bm1/cpp/home, PYTHONPATH bm1/pydeps then bm1/cpp/pydeps.
set -eu
. "$(dirname "$0")/lane.sh"
P=$1; G=$2; shift 2   # (G: one GPU, or several comma-separated: the container sees them as cuda:0, cuda:1, ...)
for g in $(echo "$G" | tr ',' ' '); do
  case ",$GPUS_OK," in *",$g,"*) ;; *) echo "dock.sh: GPU $g is not on this lane ($GPUS_OK)" >&2; exit 2 ;; esac
done
# (pre-check: every GPU of the host listed by nvidia-smi -L (GPUS_EXPECT, default 8) and the requested ones among them;
# else stop with exit 3: one card off the bus leaves every card unusable for new contexts, 3090 01:40 Xid 79 / 154)
NL=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU [0-9]*: ')
if [ "$NL" -lt "${GPUS_EXPECT:-8}" ] || nvidia-smi -L 2>&1 | grep -q 'Unable to determine'; then
  echo "dock.sh: nvidia-smi -L lists $NL GPU(s) of ${GPUS_EXPECT:-8} (or a device handle error): not starting $P" >&2; exit 3
fi
for g in $(echo "$G" | tr ',' ' '); do
  nvidia-smi -L | grep -q "^GPU $g: " || { echo "dock.sh: GPU $g not listed by nvidia-smi -L: not starting $P" >&2; exit 3; }
done
mkdir -p "$LD/home"
exec docker run --rm --init --name "$NAME-$P" --runtime=nvidia --network none \
  --cpuset-cpus "$CPUSET" --cpuset-mems "$MEMS" --memory "$MEM" --memory-swap "$MEM" --shm-size 8g --pids-limit ${PIDS:-4096} \
  --log-opt max-size=10m --log-opt max-file=2 --read-only --tmpfs /tmp:rw,exec,size=4g \
  --user "$(id -u):$(id -g)" \
  -v "$B:/bench" -v "$B/home/isaac-abplus/runtime:/bench/home/isaac-abplus/runtime:ro" \
  -e NVIDIA_VISIBLE_DEVICES="$G" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HOME=$LC/home -e ABP_HOME=$LC/home/isaac-abplus -e BM_LANE=$LC -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
  -e PYTHONPATH="${PYTHONPATH_EXTRA:-$LANE_PYTHONPATH}" \
  -e http_proxy= -e https_proxy= -e HTTP_PROXY= -e HTTPS_PROXY= -e ALL_PROXY= -e all_proxy= \
  ${DOCK_ENV:-} \
  --entrypoint /bin/bash aisaac-bench:20261019 -lc 'exec "$@"' dock "$@"
