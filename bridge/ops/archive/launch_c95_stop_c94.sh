#!/bin/bash
# On HPC160 (user 10-14 17:3x): wait for C94's checkpoint update-058800.pt (~62,600 game hours = C93's end point), stop
# C94's trainer, clean its game processes, then start C95 (ws8, B24 fused) from that checkpoint + its watcher.
# Replaces launch_c95_after_c94.sh (which waited for C94's natural end).
CK=/101063/AIsaac/ws7/runs/c94-pickup/checkpoints/update-058800.pt
L=/101063/AIsaac/ws7/runs/c94-pickup.log
while [ ! -f $CK ]; do
  if [ "$(pgrep -fc 'name [h]c94 ')" = 0 ]; then echo "C94 trainer gone before u58800"; break; fi
  sleep 60
done
sleep 30   # (the checkpoint is written in one go, but give it a moment)
date +%T; ls -la $CK; tail -n 1 $L | cut -c1-120
P=$(pgrep -f 'name [h]c94 ' | head -n 1)
if [ -n "$P" ]; then
  KIDS=$(pgrep -P $P | tr '
' ' ')   # (its worker helpers; their game processes = the guard's orphan rule)
  echo "stopping C94 trainer $P (helpers: $KIDS)"; kill -TERM $P; sleep 20; kill -0 $P 2>/dev/null && { kill -KILL $P; sleep 5; }
  for k in $KIDS; do kill -KILL $k 2>/dev/null; done
fi
sleep 5
bash /101063/AIsaac/ws1/hpc_guard.sh once | head -n 1
sleep 120
bash /101063/AIsaac/ws1/hpc_guard.sh once | head -n 1
echo "game processes before C95: $(pgrep -c isaac.x64) (the ws7 eval of u58800 may be running: 8 workers)"
touch /101063/AIsaac/ws7/runs/c94-pickup/stop-evals   # (the watcher finishes u58800's eval, then exits)
echo "resume $CK"
cd /101063/AIsaac/ws8 && bash hpc_c95.sh run "$CK"
setsid nohup bash /101063/AIsaac/ws8/hpc_c95.sh watch > /101063/AIsaac/ws8/runs/c95-watch.out 2>&1 < /dev/null &
sleep 2
echo "watchers $(pgrep -fc '^bash /101063/AIsaac/ws8/hpc_c95.sh watch') trainers $(pgrep -fc '[t]rain_tok.py')"
bash /101063/AIsaac/ws1/hpc_guard.sh once | head -n 1
