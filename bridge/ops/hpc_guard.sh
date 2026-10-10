#!/bin/bash
# Resource guard for the HPC node (sandbox /101063/AIsaac/ws1; 160-core cgroup quota). Every 60 s:
#  - game processes whose launching worker (ABP_WATCH_PID) is gone and that are older than 2 min are ended;
#  - when no trainer / evaluation runs for 3 checks in a row, every game process and worker helper left is ended
#    (a finished or killed run must not leave 2,000 clones spinning: the node's ssh becomes unusable);
#  - logged: trainers whose progress.csv has not moved for 15 min, load above the quota, a missing Xvfb :93.
# usage: nohup bash hpc_guard.sh > /dev/null 2>&1 &     log: /101063/AIsaac/ws1/runs/guard.log
#        hpc_guard.sh once (report)     hpc_guard.sh clean (end everything of ours now: only when nothing should run)
LOG=/101063/AIsaac/ws1/runs/guard.log
QUOTA=160
idle_checks=0
trainers() { pgrep -fc "[t]rain_tok.py|[e]val_tok.py"; }
end_all() {
  n=$(pgrep -c isaac.x64)
  pkill -KILL -f "[f]rom multiprocessing" 2>/dev/null
  pkill -KILL isaac.x64 2>/dev/null
  echo "$(date +%F_%T) $1: ended $n game processes and the worker helpers" >> $LOG
}
check() {
  now=$(date +%F_%T)
  for p in $(pgrep isaac.x64); do
    w=$( (tr '\000' '\012' < /proc/$p/environ) 2>/dev/null | sed -n 's/^ABP_WATCH_PID=//p')
    if [ -n "$w" ] && ! kill -0 "$w" 2>/dev/null && [ "$(ps -o etimes= -p $p 2>/dev/null | tr -d ' ')" -gt 120 ] 2>/dev/null; then
      echo "$now orphan game process $p (worker $w gone) ended" >> $LOG
      kill -KILL $p 2>/dev/null
    fi
  done
  if [ "$(trainers)" = 0 ]; then
    idle_checks=$((idle_checks + 1))
    if [ $idle_checks -ge 3 ] && [ "$(pgrep -c isaac.x64)" -gt 0 ]; then end_all "no trainer for 3 min"; fi
  else
    idle_checks=0
  fi
  for pid in $(pgrep -f "[t]rain_tok.py"); do
    out=$( (tr '\000' '\012' < /proc/$pid/cmdline) 2>/dev/null | grep -A1 "^--out$" | tail -n 1)
    [ -f "$out/progress.csv" ] || continue
    age=$(( $(date +%s) - $(stat -c %Y "$out/progress.csv") ))
    [ $age -gt 900 ] && echo "$now trainer $pid ($out) has written nothing for $age s" >> $LOG
  done
  load=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
  [ "$load" -gt $QUOTA ] && echo "$now load $load above the $QUOTA-core quota" >> $LOG
  pgrep -f "Xvfb :93" > /dev/null || echo "$now Xvfb :93 is not running" >> $LOG
}
case "$1" in
  once)
    check
    echo "load $(cut -d' ' -f1-3 /proc/loadavg) | game processes $(pgrep -c isaac.x64) | defunct $(ps -eo stat,comm | awk '$1 ~ /Z/ && $2 ~ /isaac/' | wc -l) | trainers $(pgrep -fc '[t]rain_tok.py') | evals $(pgrep -fc '[e]val_tok.py') | CUDA python $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null | grep -c python) | mem avail $(free -g | awk 'NR==2{print $7}') G"
    tail -n 5 $LOG 2>/dev/null
    exit 0 ;;
  clean)
    [ "$(trainers)" = 0 ] || { echo "a trainer or evaluation is running; not cleaning"; exit 1; }
    end_all "clean"; exit 0 ;;
esac
while true; do check; sleep 60; done
