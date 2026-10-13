#!/bin/bash
# Archive the 3080 (host 1) before it goes away (user 2026-10-13): run from the 3080 with an ssh agent forwarded from
# the PC (ssh -A). Copy A (full) -> HPC160 /101063/AIsaac/archive/host1-3080/: everything under ~/isaac-abplus except
# game instances / runtime / caches, every checkpoint of the key experiment lines, only the final checkpoints elsewhere;
# plus ~/isaac-rl/runs with only last/best/final checkpoints. Copy B (compact) -> 3090 ~/isaac-abplus/archive-host1/:
# the same without any intermediate checkpoints. Logs: /tmp/host1_migrate_{a,b}.log. usage: host1_migrate.sh
set -u
cd /home/eolc
cat > /tmp/host1_filter_a.txt <<'EOF'
+ /fk2/runs/c67-run/checkpoints/**
+ /ch1/runs/c74-chars/checkpoints/**
+ /dt/runs/c75-death/checkpoints/**
+ /lf/runs/c80-fastlearner/checkpoints/**
+ /tv/runs/c83-teacher2/checkpoints/**
+ /ch2/runs/c86-chars2/checkpoints/**
+ /mv2/runs/c89-v2mem/checkpoints/**
+ /bld/runs/c70-build/checkpoints/**
- update-*.pt
- /instances/
- /runtime/
- /backup-v1.06-linux-build22878971/
- cache/
- __pycache__/
- *.pyc
- *.sock
- core
EOF
cat > /tmp/host1_filter_b.txt <<'EOF'
- update-*.pt
- /instances/
- /runtime/
- /backup-v1.06-linux-build22878971/
- cache/
- __pycache__/
- *.pyc
- *.sock
- core
EOF
cat > /tmp/host1_filter_rl.txt <<'EOF'
+ last.pt
+ best*.pt
+ final*.pt
- *.pt
- __pycache__/
- *.pyc
EOF
HPC="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p 30575"
H2="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
$HPC root@10.15.171.204 "mkdir -p /101063/AIsaac/archive/host1-3080/isaac-abplus /101063/AIsaac/archive/host1-3080/isaac-rl-runs"
echo "== A1 isaac-abplus -> HPC160 $(date +%T)"
rsync -a --info=progress2 --no-inc-recursive --filter="merge /tmp/host1_filter_a.txt" -e "$HPC" isaac-abplus/ root@10.15.171.204:/101063/AIsaac/archive/host1-3080/isaac-abplus/ > /tmp/host1_migrate_a.log 2>&1; echo "A1 rc=$?"
echo "== A2 isaac-rl/runs -> HPC160 $(date +%T)"
rsync -a --info=progress2 --no-inc-recursive --filter="merge /tmp/host1_filter_rl.txt" -e "$HPC" isaac-rl/runs/ root@10.15.171.204:/101063/AIsaac/archive/host1-3080/isaac-rl-runs/ >> /tmp/host1_migrate_a.log 2>&1; echo "A2 rc=$?"
cp -n isaac-rl/README.md /tmp/ 2>/dev/null
$HPC root@10.15.171.204 "du -sh /101063/AIsaac/archive/host1-3080/* ; find /101063/AIsaac/archive/host1-3080 -name '*.pt' | wc -l"
echo "== B isaac-abplus (compact) -> 3090 $(date +%T)"
$H2 rhythmo@10.19.131.132 "mkdir -p ~/isaac-abplus/archive-host1/isaac-abplus ~/isaac-abplus/archive-host1/isaac-rl-runs"
rsync -a --info=progress2 --no-inc-recursive --filter="merge /tmp/host1_filter_b.txt" -e "$H2" isaac-abplus/ rhythmo@10.19.131.132:~/isaac-abplus/archive-host1/isaac-abplus/ > /tmp/host1_migrate_b.log 2>&1; echo "B1 rc=$?"
rsync -a --info=progress2 --no-inc-recursive --filter="merge /tmp/host1_filter_rl.txt" -e "$H2" isaac-rl/runs/ rhythmo@10.19.131.132:~/isaac-abplus/archive-host1/isaac-rl-runs/ >> /tmp/host1_migrate_b.log 2>&1; echo "B2 rc=$?"
$H2 rhythmo@10.19.131.132 "du -sh ~/isaac-abplus/archive-host1/*; find ~/isaac-abplus/archive-host1 -name '*.pt' | wc -l"
echo "== done $(date +%T)"
