#!/bin/sh
# gpu57 C110 (2026-11-15, the coordinator): the fp32 arm on whichever lane frees first. Waits for the first of
# /root/AIsaac/train/c110/AFTER_DONE_c110-r / _c110-q (that arm's training and evaluation over: its two GPUs free), then
# on that arm's lane, one step after the other:
#   1. the smoke c110smoke/c110s-f (HOURS 8, CKPT_EVERY 5, c108_guard.sh: shared lock), train_args = c110/fp32.train_args;
#   2. its gate (c110_gate.py: hours reached, error counters 0, no Traceback, B52 driver on, learner reserved < 23 GB,
#      config amp off / actor_bf16 0 / micro 2048 / minibatch 4096) -> c110/gate_c110s-f.txt; failing: stop here;
#   3. the D1 lr-0 check in fp32 (c108/d1_run.sh, tag f-bm-last: ab2-bm's last.pt, --lr 0, 5 updates, + --amp off
#      --actor-bf16 0 --micro 2048) on the same lane -> c108/d1/f-bm-last/;
#   4. the run c110/c110-f (HOURS 400, CKPT_EVERY 100, the same train_args) and its c110_after.sh.
# Status lines in c110/FP32_STATUS and bm1.log.
#   usage: [F32_TREE=<tree, default maa1e98a>] setsid nohup sh c110_fp32.sh [a|b: that lane at once] > /root/AIsaac/train/c110/fp32_chain.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c110; S=/root/AIsaac/train/c110smoke; BM=/root/AIsaac/bm1
say() { echo "$(date '+%F %T') $*" | tee -a $P/FP32_STATUS; echo "$(date '+%F %T') c110_fp32: $*" >> $BM/reports/bm1.log; }
[ -f $P/fp32.train_args ] || { say "no $P/fp32.train_args: stopped"; exit 2; }
first=""
# (2026-11-15, the coordinator: R / Q off -- the fp32 arm at once on lane b: c110_fp32.sh b; no argument: wait as above)
case "${1:-}" in a) first=c110-r ;; b) first=c110-q ;; esac
while [ -z "$first" ]; do
  for n in c110-r c110-q; do [ -f $P/AFTER_DONE_$n ] && { first=$n; break; }; done
  [ -n "$first" ] || sleep 60
done
case $first in
  c110-r) LANE=env/gpu57-c110-a.env; G=4; GG=4,5; CS=0-7,16-23; MS=0 ;;
  c110-q) LANE=env/gpu57-c110-b.env; G=6; GG=6,7; CS=8-15,24-31; MS=1 ;;
esac
if [ -n "${1:-}" ]; then why_="lane $1 given"; else why_="$first done ($(cat $P/AFTER_DONE_$first))"; fi
T=${F32_TREE:-maa1e98a}   # (2026-11-15: maa1e98a-f32, the bool casts of warm_spatial's bf16 -- --amp off crashed maa1e98a)
say "$why_: fp32 on $LANE (GPUs $GG), tree $T"
cd $BM
mkdir -p $S/c110s-f && cp $P/fp32.train_args $S/c110s-f/train_args
HOURS=8 CKPT_EVERY=5 MEM_GB=96 sh $BM/c108_guard.sh c110smoke c110s-f $T $LANE
say "smoke ended: $(tail -2 $S/c110s-f/guard.log | head -1 | cut -c1-160)"
if ! python3 $BM/c110_gate.py $S/c110s-f c110s-f amp=off actor_bf16=0 micro=2048 minibatch=4096 --driver on > $P/gate_c110s-f.txt 2>&1; then
  say "smoke gate FAILED ($P/gate_c110s-f.txt): stopped, no D1, no run (a person looks)"; exit 1
fi
say "smoke gate passed; D1 lr-0 fp32 check (f-bm-last) on GPUs $GG"
D1_GPUS=$GG D1_CPUSET=$CS D1_MEMS=$MS sh $BM/c108/d1_run.sh f-bm-last:new:bm-last.pt:--amp,off,--actor-bf16,0,--micro,2048 > $P/d1_fp32.out 2>&1
say "D1 done: $(tail -2 $P/d1_fp32.out | head -1 | cut -c1-160)"
mkdir -p $P/c110-f && cp $P/fp32.train_args $P/c110-f/train_args
HOURS=400 CKPT_EVERY=100 MEM_GB=96 setsid nohup sh $BM/c108_guard.sh c110 c110-f $T $LANE > $P/c110-f.launch.out 2>&1 < /dev/null &
sleep 20
setsid nohup sh $BM/c110_after.sh c110-f $G $CS $MS > $P/after_c110-f.out 2>&1 < /dev/null &
say "c110-f started on $LANE (guard.log: $(tail -1 $P/c110-f/guard.log 2>/dev/null | cut -c1-120))"
