#!/bin/sh
# gpu57 C112 host-side watcher (2026-11-16, the coordinator: the local poll died with its session and two events went
# unreported for 55 minutes). Read-only, nice 19. Every 60 s: a digest of every c112 status file's size and mtime
# (STATUS_*, LADDER, HALT*, rw4_tests/RESULT, GAP_JOB*); a change appends one line to c112/WATCH_EVENTS naming the files
# that changed. Every 30 min (and whenever the pair state changes, checked every 5 min): one line in c112/GPU_WATCH with
# GPUs 4-7's compute processes (pids, users) and which pairs (4+5, 6+7) are entirely free; a pair turning free also
# appends "PAIR FREE" to WATCH_EVENTS. Never starts anything.
#   usage: setsid nohup nice -n 19 sh c112_watch.sh > /root/AIsaac/train/c112/watch.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112
ev() { echo "$(date '+%F %T') $*" >> $P/WATCH_EVENTS; }
snap() { cd $P && ls -l --time-style=+%s STATUS_* LADDER HALT* rw4_tests/RESULT GAP_JOB* 2>/dev/null | awk '{print $5, $6, $7}'; }
gpus() {
  out=""; free=""
  for g in 4 5 6 7; do
    u=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
    ps_=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$u" | cut -d, -f2 | tr -d ' ')
    us=""; for p in $ps_; do us="$us $p:$(ps -o user= -p $p 2>/dev/null)"; done
    out="$out GPU$g[${us# }]"; [ -z "$ps_" ] && free="$free $g"
  done
  pairs=""; case " $free " in *" 4 "*" 5 "*) pairs="$pairs 4+5";; esac; case " $free " in *" 6 "*" 7 "*) pairs="$pairs 6+7";; esac
  echo "$out | free pairs:${pairs:- none}"
}
prev=$(snap | md5sum); prevp=""; n=0
ev "watch started (pid $$)"
while :; do
  cur=$(snap | md5sum)
  if [ "$cur" != "$prev" ]; then
    ev "changed: $(cd $P && find STATUS_* LADDER HALT* rw4_tests/RESULT GAP_JOB* -newermt "-70 seconds" 2>/dev/null | tr '\n' ' ')"
    prev=$cur
  fi
  if [ $((n % 5)) = 0 ]; then
    g=$(gpus); p=${g##*free pairs:}
    if [ $((n % 30)) = 0 ] || [ "$p" != "$prevp" ]; then echo "$(date '+%F %T')$g" >> $P/GPU_WATCH; fi
    if [ "$p" != "$prevp" ] && [ "$p" != " none" ]; then ev "PAIR FREE:$p ($g)"; fi
    prevp=$p
  fi
  n=$((n + 1)); sleep 60
done
