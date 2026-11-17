#!/bin/sh
# gpu57 C112 L2 test bench (2026-11-16, the coordinator): L2 block (side, R0 0.05) variants on lane a only, one after the
# other, tree maa1e98a-rw, the rest as l2blk: l2hd (--block-history-mode detach, 60 gh), l2hs (--block-history-scale
# none, 60 gh), l2long (seed 1001, 182 gh = about 150 updates). Pass: last-10 diag_hit > 0.62 with diag_left, diag_right
# >= 0.15 ("PASS" line in c112/LADDER at once). Before each arm GPUs 4 and 5 must show no compute process: otherwise
# c112/HALT, stop. An arm FAILED / HALT: c112/HALT, stop. Lines in c112/LADDER and bm1.log.
#   usage: setsid nohup sh c112_l2x.sh > /root/AIsaac/train/c112/l2x.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw; LA=env/gpu57-c110-a.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_l2x: $*" >> $BM/reports/bm1.log; }
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
cd $BM
[ -e $P/HALT ] && { say "L2 bench: HALT present, not started"; exit 1; }
for a in "l2hd 60" "l2hs 60" "l2long 182"; do
  set -- $a
  b=$(busy); [ -n "$b" ] && stop "GPUs 4/5 busy before $1 ($b): no wait, no other GPUs"
  say "L2 bench: $1 on lane a (GPUs 4,5 free), tree $T, $2 game hours"
  ARM_HOURS=$2 sh $BM/c112_arm.sh $1 $T $LA > $P/arm_$1.out 2>&1
  grep -qE "FAILED|HALT" $P/STATUS_$1 && stop "$1 $(grep -E 'FAILED|HALT' $P/STATUS_$1 | tail -1 | cut -c21-200)"
  h=$(val $1 diag_hit); l=$(val $1 diag_left); r=$(val $1 diag_right)
  if [ "$(py "${h:-0} > 0.62 and ${l:-0} >= 0.15 and ${r:-0} >= 0.15")" = 1 ]; then v=PASS; else v="no pass"; fi
  say "L2 bench: $1 $v: $(sum $1)"
done
say "L2 bench over: last-10 diag_hit l2hd $(val l2hd diag_hit), l2hs $(val l2hs diag_hit), l2long $(val l2long diag_hit) (l2blk 0.431, l2old 0.726)"
