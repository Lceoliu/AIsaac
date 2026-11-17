#!/bin/bash
# On HPC160: wait until C95's trainer has finished (finished line or no hc95 trainer), then start C96 (ws9, B30 temporal
# attention + B29 fast actor) from C95's final checkpoint (last.pt, else the latest update-*.pt) + its watcher.
# C95 is expected to finish ~2026-10-15 04:20 local (20:20 UTC on the node's clock... the node runs UTC: 20:20 10-14 UTC).
L=/101063/AIsaac/ws8/runs/c95-fused.log
while true; do
  if grep -q "^finished" $L || [ "$(pgrep -fc 'name [h]c95 ')" = 0 ]; then break; fi
  sleep 120
done
date +%T; grep "^finished" $L | tail -n 1
sleep 90
echo "game processes before C96: $(pgrep -c isaac.x64)"
touch /101063/AIsaac/ws8/runs/c95-fused/stop-evals   # (the C95 watcher finishes the last eval, then exits)
CK=/101063/AIsaac/ws8/runs/c95-fused/checkpoints/last.pt
[ -f $CK ] || CK=$(ls -t /101063/AIsaac/ws8/runs/c95-fused/checkpoints/update-*.pt | head -n 1)
echo "resume $CK"
cd /101063/AIsaac/ws9 && bash hpc_c96.sh run "$CK"
setsid nohup bash /101063/AIsaac/ws9/hpc_c96.sh watch > /101063/AIsaac/ws9/runs/c96-watch.out 2>&1 < /dev/null &
sleep 2
echo "watchers $(pgrep -fc '^bash /101063/AIsaac/ws9/hpc_c96.sh watch') trainers $(pgrep -fc '[t]rain_tok.py')"
bash /101063/AIsaac/ws1/hpc_guard.sh once | head -n 1
