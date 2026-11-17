#!/bin/sh
# gpu57 C112 EGO arms, second driver (2026-11-16, the coordinator: rw4 released -- the one pytest error, the teardown
# thread-leak check of test_packed_learner_spatial_matches_padded[cpu], occurs on unmodified maa1e98a too; l2long
# stopped; GPU 5 taken by another tenant). Waits for l2long's arm and driver to end, then l2ego-res on lane b (GPUs 6,7);
# l2ego-query starts on lane a as soon as GPUs 4 and 5 both show no compute process, else on lane b after l2ego-res.
# Each arm: smoke, coef checks, D1 in maa1e98a-rw4-d1 (max mean |dlogp| < 0.001), 60 gh. Every launch checks its GPUs.
#   usage: setsid nohup sh c112_ego2.sh > /root/AIsaac/train/c112/ego2.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw4; TD=maa1e98a-rw4-d1
LA=env/gpu57-c110-a.env; LB=env/gpu57-c110-b.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_ego2: $*" >> $BM/reports/bm1.log; }
val() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed -n "s/.* $2 \([0-9.]*\).*/\1/p"; }
sum() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed 's/.*SUMMARY //'; }
py() { python3 -c "print(1 if ($1) else 0)"; }
busy() {
  for g in $1; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u" | cut -d, -f2 | tr -d ' ' | \
      tr '\n' ' ' | sed "s/^\(.\)/GPU $g: \1/"
  done
}
run_arm() {   # name lane gpus
  b=$(busy "$3"); [ -n "$b" ] && { touch $P/HALT; say "ladder stopped: GPUs $3 busy before $1 ($b)"; return 1; }
  say "EGO: $1 on $2 (GPUs $3 free), tree $T, 60 game hours, D1 in $TD"
  D1_TREE=$TD ARM_HOURS=60 sh $BM/c112_arm.sh $1 $T $2 > $P/arm_$1.out 2>&1
  if grep -qE "FAILED|HALT" $P/STATUS_$1; then touch $P/HALT; say "ladder stopped: $1 $(grep -E 'FAILED|HALT' $P/STATUS_$1 | tail -1 | cut -c21-200)"; return 1; fi
  h=$(val $1 diag_hit); l=$(val $1 diag_left); r=$(val $1 diag_right)
  if [ "$(py "${h:-0} > 0.62 and ${l:-0} >= 0.15 and ${r:-0} >= 0.15")" = 1 ]; then v=PASS; else v="no pass"; fi
  say "EGO: $1 $v: $(sum $1)"
}
claim() { mkdir $P/.egoq_claimed 2>/dev/null; }   # l2ego-query starts once
cd $BM
say "EGO2: rw4 released by the coordinator (the teardown error is pre-existing); waiting for l2long to end"
while pgrep -f "c112_arm.sh l2long " > /dev/null || pgrep -f "^sh c112_l2x3.sh" > /dev/null; do sleep 30; done
[ -e $P/HALT ] && mv $P/HALT $P/HALT.1629
( while ! [ -d $P/.egoq_claimed ]; do
    if [ -z "$(busy '4 5')" ] && claim; then run_arm l2ego-query $LA "4 5"; break; fi
    sleep 60
  done ) &
W=$!
run_arm l2ego-res $LB "6 7"
if [ ! -e $P/HALT ] && claim; then kill $W 2>/dev/null; run_arm l2ego-query $LB "6 7"; fi
wait
say "EGO over: last-10 diag_hit l2ego-res $(val l2ego-res diag_hit), l2ego-query $(val l2ego-query diag_hit) (l2blk 0.431, l2old 0.726)"
