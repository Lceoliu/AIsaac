#!/bin/sh
# gpu57 C112 B6 (2026-11-17, the coordinator): the start-up proxy at random initialisation (seeds 1000, 1001, 1002) at
# the heads input zr (plus P token / z): V, step change, player x oof R2, door 8-sector and enemy 4-quadrant accuracy,
# for block settings unset / EGO=res / EGO=head / ENTITY_SCALE_FIX / EGO=head+FIX and the old stack. CPU only
# (jobcpu.sh), one container at a time. Script maa1e98a-rw4 c112_b2/run_b6.py (imports run_b5). Lines in c112/STATUS_b6.
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; R=/bench/bm1/trees/maa1e98a-rw4/bridge/python
say() { echo "$(date '+%F %T') $*" >> $P/STATUS_b6; echo "$(date '+%F %T') c112_b6: $*" >> $BM/reports/bm1.log; }
cd $BM; mkdir -p $P/b6_out
run() {   # name envs args
  while docker ps --format '{{.Names}}' | grep -q '^aisaac-bench-bm1-a1-cpu-'; do sleep 30; done
  say "$1 started ($2)"
  L=$(CPUSET=0-7,16-23 MEMS=0 MEM=24g ENVS="$2" sh $BM/jobcpu.sh c112-b6-$1 "cd $R && nice -n 10 \$BM1_PY c112_b2/run_b6.py --out /bench/train/c112/b6_out/$1 $3; echo b6 rc \$?" | tail -1)
  say "$1 done: $(tail -1 $L) ($L)"
}
run unset "" "--stack block --payloads c112_b2/payloads"
run ego_res "-e ISAAC_BLOCK_EGO=res" "--stack block --payloads c112_b2/payloads"
run ego_head "-e ISAAC_BLOCK_EGO=head" "--stack block --payloads c112_b2/payloads"
run scalefix "-e ISAAC_ENTITY_SCALE_FIX=1" "--stack block --payloads c112_b2/payloads"
run ego_head_scalefix "-e ISAAC_BLOCK_EGO=head -e ISAAC_ENTITY_SCALE_FIX=1" "--stack block --payloads c112_b2/payloads"
run old "" "--stack old"
say "B6 over"
