#!/bin/bash
# On HPC160: wait until C94's trainer has exited (finished line or no hc94 trainer), then start C95 (ws8, B24 fused)
# from C94's final checkpoint (last.pt, else the latest update-*.pt) + its watcher.
L=/101063/AIsaac/ws7/runs/c94-pickup.log
while true; do
  if grep -q "^finished" $L || [ "$(pgrep -fc 'name [h]c94 ')" = 0 ]; then break; fi
  sleep 120
done
date +%T; grep "^finished" $L | tail -n 1
sleep 90
echo "game processes before C95: $(pgrep -c isaac.x64)"
CK=/101063/AIsaac/ws7/runs/c94-pickup/checkpoints/last.pt
[ -f $CK ] || CK=$(ls -t /101063/AIsaac/ws7/runs/c94-pickup/checkpoints/update-*.pt | head -n 1)
echo "resume $CK"
cd /101063/AIsaac/ws8 && bash hpc_c95.sh run "$CK"
setsid nohup bash /101063/AIsaac/ws8/hpc_c95.sh watch > /101063/AIsaac/ws8/runs/c95-watch.out 2>&1 < /dev/null &
sleep 2
echo "watchers $(pgrep -fc '^bash /101063/AIsaac/ws8/hpc_c95.sh watch') trainers $(pgrep -fc '[t]rain_tok.py')"
bash /101063/AIsaac/ws1/hpc_guard.sh once | head -n 1
