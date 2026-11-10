#!/bin/sh
# The machine's cores and the trainer's threads by name (the python process running train_tok.py, not its xvfb-run /
# timeout wrappers), the game processes. usage: threads_snapshot.sh > file
echo "cores $(nproc)"
for p in $(pgrep -f 'train_tok.py'); do
  c=$(cat /proc/$p/comm 2>/dev/null)
  case "$c" in python*) ;; *) continue ;; esac
  echo "trainer pid $p ($c) threads $(ls /proc/$p/task 2>/dev/null | wc -l)"
  for t in /proc/$p/task/*; do cat $t/comm 2>/dev/null; done | sed 's/[0-9]*$//' | sort | uniq -c | sort -rn | head -25
done
echo "processes: $(pgrep -c isaac) isaac, $(ps -e --no-headers | wc -l) total; load $(cut -d' ' -f1-3 /proc/loadavg)"
