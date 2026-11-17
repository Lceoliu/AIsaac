#!/bin/sh
# gpu57 C112 host-side watcher, files only (2026-11-17: GPU fault since 2026-11-16 21:14 -- Xid 79, PCI 0000:08:00.0
# (minor 3) fell off the bus, nvidia-smi hangs; this version never calls nvidia-smi). Read-only, nice 19. Every 60 s:
# a digest of c112's status files (STATUS_*, LADDER, HALT*, rw4_tests*/RESULT, GAP_JOB*) and bm1/STOP_QUEUE; a change
# appends one line to c112/WATCH_EVENTS naming the files that changed.
#   usage: setsid nohup nice -n 19 sh c112_watch2.sh > /root/AIsaac/train/c112/watch2.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c112
ev() { echo "$(date '+%F %T') $*" >> $P/WATCH_EVENTS; }
snap() { cd $P && ls -l --time-style=+%s STATUS_* LADDER HALT* rw4_tests*/RESULT GAP_JOB* /root/AIsaac/bm1/STOP_QUEUE 2>/dev/null | awk '{print $5, $6, $7}'; }
prev=$(snap | md5sum)
ev "watch2 started (pid $$; files only, no nvidia-smi)"
while :; do
  cur=$(snap | md5sum)
  if [ "$cur" != "$prev" ]; then
    ev "changed: $(cd $P && find STATUS_* LADDER HALT* rw4_tests*/RESULT GAP_JOB* /root/AIsaac/bm1/STOP_QUEUE -newermt "-70 seconds" 2>/dev/null | tr '\n' ' ')"
    prev=$cur
  fi
  sleep 60
done
