# bm1 lane settings (sourced by dock.sh, prof.sh, gate.sh, probe.sh, deploy.sh, lock.sh). Two lanes share one host:
#   LANE=a1  (default): node0, GPUs 1-3, everything at bm1/ (the scripts, home/, reports/, trees/, LOCK_node0).
#   LANE=cpp           : node1, GPUs 4-5, everything at bm1/cpp/ (ops/ = these scripts, home/, reports/, trees/,
#                        pydeps/ after the shared bm1/pydeps), the lock bm1/LOCK_node1.
# Shared read-only by both: the image, the shared venv and game runtime (/bench/home/isaac-abplus), bm1/pydeps,
# bm1/tools (nsys, cf), bm1/payloads. Any variable below set in the environment wins (CPUSET, MEMS, GPUS_OK, LOCK, ...).
#   BM   host bm1/ (the shared top)          LD   host lane dir        LC  the lane dir in the container
#   OPS  host dir of the lane's scripts      OPC  the same in the container
#   NAME container name prefix               B    host dir mounted as /bench (dirname of BM)
BM=${BM:-/home/runaiwusheng/aisaac-bench-20261019/cpu-wide/bm1}
B=$(dirname "$BM")
LANE=${LANE:-a1}
case $LANE in
  a1)
    LD=$BM; LC=/bench/bm1; OPS=$BM; OPC=/bench/bm1
    NAME=aisaac-bench-bm1-a1
    CPUSET=${CPUSET:-0-11,24-35}; MEMS=${MEMS:-0}; GPUS_OK=${GPUS_OK:-1,2,3}
    LOCK=${LOCK:-$BM/LOCK_node0}; SESSION=${SESSION:-bm1-a1}
    LANE_PYTHONPATH=/bench/bm1/pydeps
    RUNTIME_LINK=../../../home/isaac-abplus/runtime
    ;;
  cpp)
    LD=$BM/cpp; LC=/bench/bm1/cpp; OPS=$BM/cpp/ops; OPC=/bench/bm1/cpp/ops
    NAME=aisaac-bench-bm1-cpp
    CPUSET=${CPUSET:-12-23,36-47}; MEMS=${MEMS:-1}; GPUS_OK=${GPUS_OK:-4,5}
    LOCK=${LOCK:-$BM/LOCK_node1}; SESSION=${SESSION:-bm1-cpp}
    LANE_PYTHONPATH=/bench/bm1/pydeps:/bench/bm1/cpp/pydeps
    RUNTIME_LINK=../../../../home/isaac-abplus/runtime
    ;;
  *) echo "lane.sh: LANE=$LANE (a1 | cpp)" >&2; exit 2 ;;
esac
MEM=${MEM:-64g}
