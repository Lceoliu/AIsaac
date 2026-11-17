#!/bin/sh
# gpu57 C112 (2026-11-16, the coordinator): one reward-ladder arm. Run dir /root/AIsaac/train/c112/<name> holds
# train_args and guard.pre (ENVS with ISAAC_DIAG_REWARD), copied to the smoke c112smoke/<name>s.
#   1. smoke (HOURS 4, about 3 updates; c108_guard.sh: shared lock); its guard must end with its hours reached;
#   2. c112_check.py (mode: c112/LOSS_CHECK, default strict) -> c112/check_<name>.txt; failing: stop (a person looks);
#   3. the arm (HOURS $ARM_HOURS, default 60; CKPT_EVERY 100), one STATUS line per 10 updates (c112_win.py), then the summary of the last
#      10 updates and "arm over".
# c112/HALT (the ladder's stop) is honoured before the smoke and before the arm. Lines in c112/STATUS_<name> and bm1.log.
#   usage: sh c112_arm.sh <name> <tree> <lane file> > /root/AIsaac/train/c112/arm_<name>.out 2>&1
N=$1; T=$2; LANE=$3
P=/root/AIsaac/train/c112; S=/root/AIsaac/train/c112smoke; BM=/root/AIsaac/bm1; SN=${N}s; S0=$P/$N/$N-s0
say() { echo "$(date '+%F %T') $*" >> $P/STATUS_$N; echo "$(date '+%F %T') c112_arm $N: $*" >> $BM/reports/bm1.log; }
cd $BM
[ -e $P/HALT ] && { say "HALT before the smoke: not started"; exit 1; }
mkdir -p $S/$SN && cp $P/$N/train_args $P/$N/guard.pre $S/$SN/
say "smoke $SN on $LANE, tree $T"
HOURS=4 CKPT_EVERY=5 MEM_GB=96 sh $BM/c108_guard.sh c112smoke $SN $T $LANE
say "smoke ended: $(tail -2 $S/$SN/guard.log | head -1 | cut -c1-160)"
grep -q "game hours reached" $S/$SN/guard.log || { say "FAILED: the smoke fell short (a person looks)"; exit 1; }
M=$(cat $P/LOSS_CHECK 2>/dev/null || echo strict)
if ! python3 $BM/c112_check.py $S/$SN/$SN-s0 $M > $P/check_$N.txt 2>&1; then
  say "FAILED: smoke check ($M): $(tail -1 $P/check_$N.txt | cut -c1-300) (a person looks)"; exit 1
fi
say "smoke check ok ($M; $P/check_$N.txt)"
if [ -n "${D1_TREE:-}" ]; then   # (rw4 EGO arms, the coordinator: actor / learner logp at frozen weights, max mean |dlogp| < 0.001)
  case $LANE in *-a.env) GG=4,5; CS=0-7,16-23; MS=0 ;; *) GG=6,7; CS=8-15,24-31; MS=1 ;; esac
  cp $S/$SN/$SN-s0/checkpoints/last.pt /root/AIsaac/train/c108/d1/inputs/$N-smoke-last.pt
  XE=$(sed -n 's/^ENVS="\(.*\)"$/\1/p' $P/$N/guard.pre 2>/dev/null | sed 's/-e //g' | tr ' ' ',')
  say "D1 lr-0 ($N-smoke, tree $D1_TREE, env '$XE') on GPUs $GG"
  D1_CODE=$D1_TREE D1_ARGS=$P/$N/train_args D1_GPUS=$GG D1_CPUSET=$CS D1_MEMS=$MS \
    sh $BM/c108/d1_run.sh $N-smoke:new:$N-smoke-last.pt::$XE > $P/d1_$N.out 2>&1
  D1=$(python3 -c "
import json
L=[json.loads(l) for l in open('/root/AIsaac/train/c108/d1/$N-smoke/d1_diag.jsonl')]
m=max(r['all']['mean_abs'] for r in L)
print(len(L), '%.6f' % m, 'ok' if L and m < 0.001 else 'bad')" 2>&1 | tail -1)
  case "$D1" in *ok) say "D1 ok (lines, max mean |dlogp|: $D1)" ;; *) say "FAILED: D1 ($D1; $P/d1_$N.out) (a person looks)"; exit 1 ;; esac
fi
[ -e $P/HALT ] && { say "HALT before the arm: not started"; exit 1; }
HOURS=${ARM_HOURS:-60} CKPT_EVERY=100 MEM_GB=96 setsid nohup sh $BM/c108_guard.sh c112 $N $T $LANE > $P/$N.launch.out 2>&1 < /dev/null &
sleep 30
say "arm started ($(tail -1 $P/$N/guard.log 2>/dev/null | cut -c1-80))"
last=0
nrow() { echo $(( $(wc -l < $S0/progress.csv 2>/dev/null || echo 1) - 1 )); }
flush() {
  n=$(nrow)
  while [ $((last + 10)) -le $n ]; do
    say "u$((last + 1))-$((last + 10)): $(python3 $BM/c112_win.py $S0 $((last + 1)) $((last + 10)) 2>&1 | tail -1)"
    last=$((last + 10))
  done
}
while pgrep -f "c108_guard.sh c112 $N " > /dev/null; do flush; sleep 60; done
flush; n=$(nrow)
say "ended: $(tail -2 $P/$N/guard.log | head -1 | cut -c1-160)"
say "SUMMARY last 10 (u$((n - 9))-u$n): $(python3 $BM/c112_win.py $S0 $((n - 9)) $n 2>&1 | tail -1)"
say "arm over"
