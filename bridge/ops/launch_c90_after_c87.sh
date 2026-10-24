#!/bin/bash
# On HPC128: wait until C87's trainer has exited (finished line or no train_tok process), then start C90 + its watcher.
L=/101063/AIsaac/ws4/runs/c87-memory.log
while true; do
  if grep -q "^finished" $L || [ "$(pgrep -fc '[t]rain_tok.py')" = 0 ]; then break; fi
  sleep 120
done
date +%T; grep "^finished" $L | tail -n 1
sleep 90
echo "game processes before C90: $(pgrep -c isaac.x64)"
cd /101063/AIsaac/ws4 && bash hpc_c90.sh run
setsid nohup bash /101063/AIsaac/ws4/hpc_c90.sh watch > /101063/AIsaac/ws4/runs/c90-watch.out 2>&1 < /dev/null &
sleep 2
echo "watchers $(pgrep -fc 'hpc_c90.sh watch') trainers $(pgrep -fc '[t]rain_tok.py')"
bash /101063/AIsaac/ws4/hpc_guard.sh once | head -n 1
