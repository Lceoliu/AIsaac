#!/bin/sh
# gpu57 C112 L3 / L4 on lane a only (2026-11-16, the coordinator): block first, the old stack only as a ruler when block
# misses. Tree maa1e98a-rw2 (rw + the door / room modes). Waits for c112_l2a.sh to end, then:
#   L2: block passes when its last-10 diag_hit > 0.62 with diag_left, diag_right >= 0.15 and (the old stack passing L2)
#       (block - 0.140) >= (old - 0.133) / 2; otherwise stop ("stop before L3").
#   L3 block (60 gh; ISAAC_DIAG_REWARD=door): last-10 diag_hit > 0.40 -> L4; otherwise L3 old as the ruler, then stop.
#   L4 block (182 gh, about 150 updates; room, ISAAC_DIAG_R0=1): passes when the last-10 mean diag_per_end >= 1.0 and >=
#       2 x the first 10 updates' mean; otherwise L4 old (182 gh) as the ruler. End.
# Before each arm GPUs 4 and 5 must show no compute process (none of ours runs then): otherwise c112/HALT, stop. An arm
# FAILED / HALT: c112/HALT, stop. Lines in c112/LADDER and bm1.log.
#   usage: setsid nohup sh c112_l34.sh > /root/AIsaac/train/c112/l34.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112; BM=/root/AIsaac/bm1; T=maa1e98a-rw2; LA=env/gpu57-c110-a.env
say() { echo "$(date '+%F %T') $*" >> $P/LADDER; echo "$(date '+%F %T') c112_l34: $*" >> $BM/reports/bm1.log; }
val() { grep SUMMARY $P/STATUS_$1 2>/dev/null | tail -1 | sed -n "s/.* $2 \([0-9.]*\).*/\1/p"; }
first10() { grep " u1-10: " $P/STATUS_$1 2>/dev/null | tail -1 | sed -n "s/.* $2 \([0-9.]*\).*/\1/p"; }
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
arm() {   # name, game hours
  b=$(busy); [ -n "$b" ] && stop "GPUs 4/5 busy before $1 ($b): no wait, no other GPUs"
  say "$1 on lane a (GPUs 4,5 free), tree $T, $2 game hours"
  ARM_HOURS=$2 sh $BM/c112_arm.sh $1 $T $LA > $P/arm_$1.out 2>&1
  grep -qE "FAILED|HALT" $P/STATUS_$1 && stop "$1 $(grep -E 'FAILED|HALT' $P/STATUS_$1 | tail -1 | cut -c21-200)"
  say "$1 over: $(sum $1)"
}
cd $BM
say "L3 / L4 waiting for L2 (c112_l2a.sh)"
while pgrep -f "^sh c112_l2a.sh" > /dev/null; do sleep 60; done
tail -1 $P/LADDER | grep -q "ladder over (L2 on lane a)" || { say "L2 did not end normally: no L3 ($(tail -1 $P/LADDER | cut -c21-120))"; exit 1; }
[ -e $P/HALT ] && { say "HALT present: no L3"; exit 1; }
h=$(val l2blk diag_hit); l=$(val l2blk diag_left); r=$(val l2blk diag_right)
ho=$(val l2old diag_hit); lo=$(val l2old diag_left); ro=$(val l2old diag_right)
oldok=$(py "${ho:-0} > 0.62 and ${lo:-0} >= 0.15 and ${ro:-0} >= 0.15")
blkok=$(py "${h:-0} > 0.62 and ${l:-0} >= 0.15 and ${r:-0} >= 0.15 and ($oldok == 0 or ${h:-0} - 0.140 >= (${ho:-0} - 0.133) / 2)")
if [ "$blkok" != 1 ]; then
  say "L2: block did not pass (block hit $h left $l right $r; old hit $ho left $lo right $ro, old passing $oldok): stop before L3 (the coordinator)"; exit 0
fi
say "L2: block passed (block hit $h left $l right $r; old hit $ho, old passing $oldok)"
arm l3blk 60
h=$(val l3blk diag_hit)
if [ "$(py "${h:-0} > 0.40")" != 1 ]; then
  say "L3: block missed (last-10 diag_hit ${h:-?} <= 0.40): the old stack as the ruler"
  arm l3old 60
  say "L3 over: block ${h:-?}, old $(val l3old diag_hit) (baseline 0.125): stop, no L4 (the coordinator)"; exit 0
fi
say "L3: block passed (last-10 diag_hit $h > 0.40): no old arm; L4"
arm l4blk 182
e=$(val l4blk diag_per_end); e0=$(first10 l4blk diag_per_end)
if [ "$(py "${e:-0} >= 1.0 and ${e:-0} >= 2 * ${e0:-1e9}")" = 1 ]; then
  say "ladder over: L4 block passed (diag_per_end last-10 $e, first-10 $e0): no old arm"; exit 0
fi
say "L4: block missed (diag_per_end last-10 ${e:-?}, first-10 ${e0:-?}): the old stack as the ruler"
arm l4old 182
say "ladder over: L4 block ${e:-?} (first-10 ${e0:-?}), old $(val l4old diag_per_end) (first-10 $(first10 l4old diag_per_end))"
