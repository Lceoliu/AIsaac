#!/bin/sh
# gpu57 C111 (2026-11-16, the coordinator): one arm's smoke and checks, one after the other; the 400-hour run is started
# by hand after a person reads them. <run dir> /root/AIsaac/train/c111/<name> holds train_args and guard.pre (KEY=VALUE:
# BASE, REQUIRE_DRIVER, ENVS, ...), copied to the smoke c111smoke/<name>s.
#   1. smoke (HOURS 8, CKPT_EVERY 5, c108_guard.sh: shared lock);
#   2. c110_gate.py [--driver on] -> c111/gate_<name>.txt; failing: stop;
#   3. D1 lr-0 (d1 tree <d1 tree>, resumed from the smoke's last.pt, D1_ARGS = the arm's train_args, its ENVS' K=V as
#      the D1 env) -> c108/d1/<name>-smoke/.
# Status lines in c111/STATUS_<name> (one line per step; "chain over" / "FAILED" at the end) and bm1.log.
#   usage: setsid nohup sh c111_smoke.sh <name> <tree> <d1 tree> <lane file> <gpus a,b> <cpuset> <mems> <on|any> > /root/AIsaac/train/c111/chain_<name>.out 2>&1 < /dev/null &
N=$1; T=$2; TD=$3; LANE=$4; GG=$5; CS=$6; MS=$7; DRV=$8
P=/root/AIsaac/train/c111; S=/root/AIsaac/train/c111smoke; BM=/root/AIsaac/bm1; SN=${N}s
say() { echo "$(date '+%F %T') $*" | tee -a $P/STATUS_$N; echo "$(date '+%F %T') c111_smoke $N: $*" >> $BM/reports/bm1.log; }
cd $BM
mkdir -p $S/$SN && cp $P/$N/train_args $S/$SN/train_args && { [ -f $P/$N/guard.pre ] && cp $P/$N/guard.pre $S/$SN/guard.pre; true; }
say "smoke $SN on $LANE, tree $T"
HOURS=8 CKPT_EVERY=5 MEM_GB=96 sh $BM/c108_guard.sh c111smoke $SN $T $LANE
say "smoke ended: $(tail -2 $S/$SN/guard.log | head -1 | cut -c1-160)"
if ! python3 $BM/c110_gate.py $S/$SN $SN --driver $DRV > $P/gate_$N.txt 2>&1; then
  say "smoke gate FAILED ($P/gate_$N.txt): stopped (a person looks)"; exit 1
fi
cp $S/$SN/$SN-s0/checkpoints/last.pt /root/AIsaac/train/c108/d1/inputs/$N-smoke-last.pt
XE=$(sed -n 's/^ENVS="\(.*\)"$/\1/p' $P/$N/guard.pre 2>/dev/null | sed 's/-e //g' | tr ' ' ',')
say "gate passed; D1 lr-0 ($N-smoke, tree $TD, env '$XE') on GPUs $GG"
D1_CODE=$TD D1_ARGS=$P/$N/train_args D1_GPUS=$GG D1_CPUSET=$CS D1_MEMS=$MS \
  sh $BM/c108/d1_run.sh $N-smoke:new:$N-smoke-last.pt::$XE > $P/d1_$N.out 2>&1
say "D1 done: $(tail -2 $P/d1_$N.out | head -1 | cut -c1-160); chain over (the run: by hand)"
