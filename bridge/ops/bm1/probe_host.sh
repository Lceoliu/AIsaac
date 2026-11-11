#!/bin/sh
# bm1: print a host's facts for writing its bm1.env (reads only; writes nothing): CPUs and their NUMA nodes, memory,
# GPUs with their NUMA affinity and current use, the driver, the kernel log's Xid count, the tools bm1 uses.
#   usage: sh probe_host.sh
echo "== host"; uname -a; (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")
echo "== CPUs"; nproc --all; lscpu 2>/dev/null | grep -E '^(Model name|Socket|Core|Thread|NUMA node)'
echo "== memory"; free -g 2>/dev/null | head -2
echo "== NUMA"; (numactl -H 2>/dev/null || ls -d /sys/devices/system/node/node* 2>/dev/null) | head -20
echo "== GPUs"; nvidia-smi -L 2>&1
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version --format=csv 2>&1
echo "== GPU topology (CPU / NUMA affinity)"; nvidia-smi topo -m 2>&1 | head -20
echo "== Xid lines in the kernel log"; ( journalctl -k --no-pager 2>/dev/null || dmesg 2>/dev/null ) | grep -c 'NVRM: Xid'
echo "== cgroup v2 (direct mode's memory limit)"; cat /proc/self/cgroup; ls -ld /sys/fs/cgroup 2>&1
echo "== tools"; for t in python3 python3.10 g++ gcc xvfb-run taskset numactl mpstat pkill setsid gio zstd docker; do
  printf '%-10s %s\n' $t "$(command -v $t || echo MISSING)"; done
echo "== disk"; df -h ~ 2>/dev/null | tail -1
