#!/bin/sh
# gpu57 C112 L2 bench, third driver (2026-11-16, the coordinator: the other tenant left at 15:50, lane b usable again
# with the same per-launch check). Replaces c112_l2x2.sh (stopped by hand during its gap job): finishes the running gap
# job's bookkeeping in the background (GAP_JOB -> GAP_JOB.done-<time>, a LADDER line), and runs A3 l2long (seed 1001,
# 182 gh) on lane b now. GPUs 6/7 with any compute process: c112/HALT, stop. Lines in c112/LADDER and bm1.log.
#   usage: setsid nohup sh c112_l2x3.sh > /root/AIsaac/train/c112/l2x3.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw; LB=env/gpu57-c110-b.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_l2x3: $*" >> $BM/reports/bm1.log; }
val() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed -n "s/.* $2 \([0-9.]*\).*/\1/p"; }
sum() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed 's/.*SUMMARY //'; }
py() { python3 -c "print(1 if ($1) else 0)"; }
busy() {
  for g in 6 7; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u" | cut -d, -f2 | tr -d ' ' | \
      tr '\n' ' ' | sed "s/^\(.\)/GPU $g: \1/"
  done
}
cd $BM
( while pgrep -f "^sh $P/GAP_JOB" > /dev/null; do sleep 30; done
  s=$(date +%m%d-%H%M%S); [ -f $P/GAP_JOB ] && mv $P/GAP_JOB $P/GAP_JOB.done-$s
  say "gap job over ($(ls -t $P/gap_*.out 2>/dev/null | head -1); b_out: $(ls $P/b_out 2>/dev/null | tr '\n' ' '))" ) &
b=$(busy); [ -n "$b" ] && { touch $P/HALT; say "ladder stopped: GPUs 6/7 busy before l2long ($b): not started"; wait; exit 1; }
say "L2 bench: l2long on lane b (GPUs 6,7 free), tree $T, 182 game hours"
ARM_HOURS=182 sh $BM/c112_arm.sh l2long $T $LB > $P/arm_l2long.out 2>&1
if grep -qE "FAILED|HALT" $P/STATUS_l2long; then touch $P/HALT; say "ladder stopped: l2long $(grep -E 'FAILED|HALT' $P/STATUS_l2long | tail -1 | cut -c21-200)"; wait; exit 1; fi
h=$(val l2long diag_hit); l=$(val l2long diag_left); r=$(val l2long diag_right)
if [ "$(py "${h:-0} > 0.62 and ${l:-0} >= 0.15 and ${r:-0} >= 0.15")" = 1 ]; then v=PASS; else v="no pass"; fi
say "L2 bench: l2long $v: $(sum l2long)"
wait
say "L2 bench over: last-10 diag_hit l2hd $(val l2hd diag_hit), l2hs $(val l2hs diag_hit), l2long $(val l2long diag_hit) (l2blk 0.431, l2old 0.726)"
