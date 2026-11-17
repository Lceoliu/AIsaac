#!/bin/bash
# Copy A only (3080 -> HPC160 /101063/AIsaac/archive/host1-3080/), re-run after rsync was installed on HPC160
# (the first attempt failed with "rsync: command not found" on the receiver). Needs the forwarded ssh agent (ssh -A).
set -u
cd /home/eolc
[ -f /tmp/host1_filter_a.txt ] || { echo "filters missing: run host1_migrate.sh first"; exit 1; }
HPC="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p 30575"
$HPC root@10.15.171.204 "mkdir -p /101063/AIsaac/archive/host1-3080/isaac-abplus /101063/AIsaac/archive/host1-3080/isaac-rl-runs"
echo "== A1 isaac-abplus -> HPC160 $(date +%T)"
rsync -a --info=progress2 --no-inc-recursive --filter="merge /tmp/host1_filter_a.txt" -e "$HPC" isaac-abplus/ root@10.15.171.204:/101063/AIsaac/archive/host1-3080/isaac-abplus/ > /tmp/host1_migrate_a.log 2>&1; echo "A1 rc=$?"
echo "== A2 isaac-rl/runs -> HPC160 $(date +%T)"
rsync -a --info=progress2 --no-inc-recursive --filter="merge /tmp/host1_filter_rl.txt" -e "$HPC" isaac-rl/runs/ root@10.15.171.204:/101063/AIsaac/archive/host1-3080/isaac-rl-runs/ >> /tmp/host1_migrate_a.log 2>&1; echo "A2 rc=$?"
$HPC root@10.15.171.204 "du -sh /101063/AIsaac/archive/host1-3080/*; find /101063/AIsaac/archive/host1-3080 -name '*.pt' | wc -l"
tail -c 400 /tmp/host1_migrate_a.log | tr '\r' '\n' | tail -n 3
echo "== done $(date +%T)"
