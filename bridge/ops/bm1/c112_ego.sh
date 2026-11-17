#!/bin/sh
# gpu57 C112 EGO arms (2026-11-16, the coordinator): tree maa1e98a-rw4 (ISAAC_BLOCK_EGO), L2 block, 60 gh, seed 1000.
#   1. c112_rw4_tests.sh on lane a (block CPU tests x 4 switch settings, default-off identity, EGO forwards): anything
#      but "RW4 TESTS OK": stop (HALT).
#   2. l2ego-res on lane a (smoke, coef checks, D1 in maa1e98a-rw4-d1 with max mean |dlogp| < 0.001, the arm);
#   3. l2ego-query on whichever lane frees first (lane b after l2long, or lane a after l2ego-res).
# Every launch: the lane's GPUs with no compute process, else stop (HALT). A "PASS" line in c112/LADDER at once.
#   usage: setsid nohup sh c112_ego.sh > /root/AIsaac/train/c112/ego.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw4; TD=maa1e98a-rw4-d1
LA=env/gpu57-c110-a.env; LB=env/gpu57-c110-b.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_ego: $*" >> $BM/reports/bm1.log; }
val() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed -n "s/.* $2 \([0-9.]*\).*/\1/p"; }
sum() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed 's/.*SUMMARY //'; }
py() { python3 -c "print(1 if ($1) else 0)"; }
busy() {   # gpu list "4 5"
  for g in $1; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u" | cut -d, -f2 | tr -d ' ' | \
      tr '\n' ' ' | sed "s/^\(.\)/GPU $g: \1/"
  done
}
stop() { touch $P/HALT; say "ladder stopped: $*"; exit 1; }
run_arm() {   # name lane gpus
  b=$(busy "$3"); [ -n "$b" ] && { touch $P/HALT; say "ladder stopped: GPUs $3 busy before $1 ($b)"; return 1; }
  say "EGO: $1 on $2 (GPUs $3 free), tree $T, 60 game hours, D1 in $TD"
  D1_TREE=$TD ARM_HOURS=60 sh $BM/c112_arm.sh $1 $T $2 > $P/arm_$1.out 2>&1
  if grep -qE "FAILED|HALT" $P/STATUS_$1; then touch $P/HALT; say "ladder stopped: $1 $(grep -E 'FAILED|HALT' $P/STATUS_$1 | tail -1 | cut -c21-200)"; return 1; fi
  h=$(val $1 diag_hit); l=$(val $1 diag_left); r=$(val $1 diag_right)
  if [ "$(py "${h:-0} > 0.62 and ${l:-0} >= 0.15 and ${r:-0} >= 0.15")" = 1 ]; then v=PASS; else v="no pass"; fi
  say "EGO: $1 $v: $(sum $1)"
}
cd $BM
[ -e $P/HALT ] && { say "EGO: HALT present, not started"; exit 1; }
b=$(busy "4 5"); [ -n "$b" ] && stop "GPUs 4/5 busy before the rw4 tests ($b)"
say "EGO: rw4 tests (lane a, CPU)"
sh $BM/c112_rw4_tests.sh > $P/rw4_tests.out 2>&1
r=$(tail -1 $P/rw4_tests/RESULT 2>/dev/null)
[ "$r" = "RW4 TESTS OK" ] || stop "rw4 tests: ${r:-no RESULT} ($P/rw4_tests/RESULT)"
say "EGO: rw4 tests ok ($(grep -c . $P/rw4_tests/RESULT) lines in rw4_tests/RESULT)"
( run_arm l2ego-res $LA "4 5" ) &
A=$!
sleep 120
while :; do
  if ! pgrep -f "c112_arm.sh l2long " > /dev/null && ! pgrep -f "^sh c112_l2x3.sh" > /dev/null; then lane=$LB; g="6 7"; break; fi
  if ! kill -0 $A 2>/dev/null; then lane=$LA; g="4 5"; break; fi
  sleep 60
done
[ -e $P/HALT ] && { say "EGO: HALT present: l2ego-query not started"; wait; exit 1; }
run_arm l2ego-query $lane "$g"
wait
say "EGO over: last-10 diag_hit l2ego-res $(val l2ego-res diag_hit), l2ego-query $(val l2ego-query diag_hit) (l2blk 0.431, l2old 0.726)"
