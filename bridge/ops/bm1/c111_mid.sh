#!/bin/sh
# gpu57 C111 (2026-11-16, the coordinator; the user: "早点检查看趋势"): an arm's early readings, one line per 50-update
# window at u50 / u100 / u150 / u200 (c111_win50.py: rooms>=1 of the window's episodes, mean ent_move) -> c111/MID_<name>.
# With "earlystop <eval gpu> <cpuset> <mems>" (REF): at u150, when u51-100 and u101-150 both have rooms>=1 <= 0.22, the
# run is stopped (its STOP file; directory and checkpoints kept) and its last.pt evaluated as C110's (128 seeds,
# ev_ab2.sh PARENT=c111 -> c111/eval/<name>-c128-last). They change no other verdict.
#   usage: setsid nohup sh c111_mid.sh <name> [earlystop <gpu> <cpuset> <mems>] > /root/AIsaac/train/c111/mid_<name>.out 2>&1 < /dev/null &
N=$1; ES=${2:-}; G=${3:-}; CS=${4:-}; MS=${5:-}
P=/root/AIsaac/train/c111; BM=/root/AIsaac/bm1; S0=$P/$N/$N-s0
say() { echo "$(date '+%F %T') $*" >> $P/MID_$N; echo "$(date '+%F %T') c111_mid $N: $*" >> $BM/reports/bm1.log; }
r() { python3 $BM/c111_win50.py $S0 $1 $2 2>&1 | tail -1; }
for u in 50 100 150 200; do
  while [ "$(( $(wc -l < $S0/progress.csv 2>/dev/null || echo 1) - 1 ))" -lt $u ]; do
    pgrep -f "c108_guard.sh c111 $N " > /dev/null || { say "u$u: the run ended first"; exit 0; }
    sleep 60
  done
  sleep 30   # (the window's last episodes written)
  say "u$u: $(r $((u-49)) $u)"
  if [ $u = 150 ] && [ "$ES" = earlystop ]; then
    a=$(r 51 100 | sed -n 's/.*rooms>=1 \([0-9.]*\).*/\1/p'); b=$(r 101 150 | sed -n 's/.*rooms>=1 \([0-9.]*\).*/\1/p')
    if [ "$(python3 -c "print(1 if $a <= 0.22 and $b <= 0.22 else 0)")" = 1 ]; then
      say "early stop: u51-100 $a and u101-150 $b both <= 0.22: STOP"
      touch $P/$N/STOP
      while pgrep -f "c108_guard.sh c111 $N " > /dev/null; do sleep 30; done
      PARENT=c111 sh $BM/ev_ab2.sh $N new $G $CS $MS 14 $N-c128-last,last.pt,2147490000:128 > $P/eval_$N.out 2>&1
      say "early stop: run ended, evaluation $(ls $P/eval/$N-c128-last/summary.json 2>/dev/null | wc -l)/1 summary.json; c111_mid over"
      exit 0
    fi
    say "u150: no early stop (u51-100 $a, u101-150 $b)"
  fi
done
say "u200 reached: c111_mid over (the coordinator decides the rest)"
