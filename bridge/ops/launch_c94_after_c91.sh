#!/bin/bash
# On HPC160: wait until C91's trainer has exited (finished line or no train_tok process), then start C94 (ws7) from
# C93's latest checkpoint on the shared storage + its watcher.
L=/101063/AIsaac/ws5/runs/c91-memory-cont2.log
while true; do
  if grep -q "^finished" $L || [ "$(pgrep -fc '[t]rain_tok.py')" = 0 ]; then break; fi
  sleep 120
done
date +%T; grep "^finished" $L | tail -n 1
sleep 90
echo "game processes before C94: $(pgrep -c isaac.x64)"
CK=$(ls -t /101063/AIsaac/ws4/runs/c93-items-cont/checkpoints/update-*.pt | head -n 1)
echo "resume $CK"
cd /101063/AIsaac/ws7 && bash hpc_c94.sh run "$CK"
setsid nohup bash /101063/AIsaac/ws7/hpc_c94.sh watch > /101063/AIsaac/ws7/runs/c94-watch.out 2>&1 < /dev/null &
sleep 2
echo "watchers $(pgrep -fc 'hpc_c94.sh watch') trainers $(pgrep -fc '[t]rain_tok.py')"
bash /101063/AIsaac/ws1/hpc_guard.sh once | head -n 1
