#!/bin/sh
# gpu57 C112 L2 test bench, second driver (2026-11-16, the coordinator: B2 runs on gpu57 in a gap of lane a). Takes over
# from c112_l2x.sh (stopped by hand while its first arm l2hd runs): waits for the running arm l2hd's c112_arm.sh to end,
# writes its verdict, then runs l2hs (60 gh) and l2long (182 gh) as c112_l2x.sh would. Before each of those arms, and once
# more at the end: when c112/GAP_JOB exists (a shell script for a short container job on lane a, written by hand), it
# runs first (sh GAP_JOB, output c112/gap_<time>.out) and is renamed GAP_JOB.done-<time>. Pass: last-10 diag_hit > 0.62
# with diag_left, diag_right >= 0.15. GPUs 4/5 with any compute process before an arm: c112/HALT, stop. Lines in
# c112/LADDER and bm1.log.
#   usage: setsid nohup sh c112_l2x2.sh > /root/AIsaac/train/c112/l2x2.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw; LA=env/gpu57-c110-a.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_l2x2: $*" >> $BM/reports/bm1.log; }
val() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed -n "s/.* $2 \([0-9.]*\).*/\1/p"; }
sum() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed 's/.*SUMMARY //'; }
py() { python3 -c "print(1 if ($1) else 0)"; }
busy() {
  for g in 4 5; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u" | cut -d, -f2 | tr -d ' ' | \
      tr '\n' ' ' | sed "s/^\(.\)/GPU $g: \1/"
  done
}
stop() { touch $P/HALT; say "ladder stopped: $*"; exit 1; }
verdict() {
  grep -qE "FAILED|HALT" $P/STATUS_$1 && stop "$1 $(grep -E 'FAILED|HALT' $P/STATUS_$1 | tail -1 | cut -c21-200)"
  h=$(val $1 diag_hit); l=$(val $1 diag_left); r=$(val $1 diag_right)
  if [ "$(py "${h:-0} > 0.62 and ${l:-0} >= 0.15 and ${r:-0} >= 0.15")" = 1 ]; then v=PASS; else v="no pass"; fi
  say "L2 bench: $1 $v: $(sum $1)"
}
gap() {
  [ -f $P/GAP_JOB ] || return 0
  b=$(busy); [ -n "$b" ] && stop "GPUs 4/5 busy before the gap job ($b)"
  s=$(date +%m%d-%H%M%S)
  say "gap job on lane a: $(head -c 120 $P/GAP_JOB | tr '\n' ' ')"
  sh $P/GAP_JOB > $P/gap_$s.out 2>&1; rc=$?
  mv $P/GAP_JOB $P/GAP_JOB.done-$s
  say "gap job over rc=$rc ($P/gap_$s.out)"
}
cd $BM
say "L2 bench: driver 2 takes over (waiting for l2hd)"
while pgrep -f "c112_arm.sh l2hd " > /dev/null; do sleep 30; done
verdict l2hd
for a in "l2hs 60" "l2long 182"; do
  set -- $a
  gap
  [ -e $P/HALT ] && stop "HALT present before $1"
  b=$(busy); [ -n "$b" ] && stop "GPUs 4/5 busy before $1 ($b): no wait, no other GPUs"
  say "L2 bench: $1 on lane a (GPUs 4,5 free), tree $T, $2 game hours"
  ARM_HOURS=$2 sh $BM/c112_arm.sh $1 $T $LA > $P/arm_$1.out 2>&1
  verdict $1
done
gap
say "L2 bench over: last-10 diag_hit l2hd $(val l2hd diag_hit), l2hs $(val l2hs diag_hit), l2long $(val l2long diag_hit) (l2blk 0.431, l2old 0.726)"
