#!/bin/sh
# bm1 (8x3090 dgene): run one command in the aisaac-bench:20261019 image, on this lane's CPUs / NUMA node / one GPU.
#   usage: dock.sh <purpose> <gpu> <command> [args...]
#   container: aisaac-bench-bm1-a1-<purpose>, --rm, 64 GB, /bench = cpu-wide (the game runtime read-only), the
#   caller's uid, HOME=/bench/bm1/home (every cache: isaac-actor, learn-native, visible, Inductor, Triton).
#   PYTHONPATH=/bench/bm1/pydeps (packages the shared venv lacks or has at another version than the 4090; README).
#   Lane (lane.sh: LANE=a1 node0 / LANE=cpp node1; environment overrides): CPUSET, MEMS, GPUS_OK, MEM=64g. LANE=cpp:
#   container aisaac-bench-bm1-cpp-<purpose>, HOME=/bench/bm1/cpp/home, PYTHONPATH bm1/pydeps then bm1/cpp/pydeps.
set -eu
. "$(dirname "$0")/lane.sh"
P=$1; G=$2; shift 2
case ",$GPUS_OK," in *",$G,"*) ;; *) echo "dock.sh: GPU $G is not on this lane ($GPUS_OK)" >&2; exit 2 ;; esac
mkdir -p "$LD/home"
exec docker run --rm --init --name "$NAME-$P" --runtime=nvidia --network none \
  --cpuset-cpus "$CPUSET" --cpuset-mems "$MEMS" --memory "$MEM" --memory-swap "$MEM" --shm-size 8g --pids-limit 4096 \
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
