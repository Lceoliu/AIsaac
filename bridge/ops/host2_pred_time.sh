#!/bin/bash
# The update's cost of the switches, one trainer at a time on host 2's GPU: arms in turn (ROUNDS times), minibatch and micro
# as in production (4096), ISAAC_RL_PROFILE=1 (u_* = real GPU seconds per part). A rough early reading; the gate is the 3-GPU
# layout (M4). usage: host2_pred_time.sh <label>
cd /home/rhythmo/isaac-abplus/pred || exit 1
label=${1:-t}
export ISAAC_RL_PROFILE=1 ISAAC_RL_LOG_GRAD_NORM=1 SMOKE_TIMEOUT=1200
for round in $(seq 1 ${ROUNDS:-2}); do
  for arm in ${ARMS:-a0 a1 a2 a3}; do
    case $arm in
      a0) sw="" ;;
      a1) sw="--aux-predict 1" ;;
      a2) sw="--priv-critic 1" ;;
      a3) sw="--aux-predict 1 --priv-critic 1" ;;
    esac
    HOURS=${HOURS:-6} WORKERS=8 PORT=49700 ops/host2_pred_smoke.sh "$label-$arm-$round" --minibatch 4096 --micro 4096 ${EXTRA:-} $sw | head -n 1
  done
done
cat > /tmp/pred_cmp.py <<EOF
import csv, glob, statistics as s
keys = ("update_s", "u_ppo", "u_batch", "u_recompute", "u_opt", "u_aux", "u_priv", "collect_s", "x_real_time")
for arm in ("a0", "a1", "a2", "a3"):
    rows = []
    for f in sorted(glob.glob("runs/$label-%s-*/progress.csv" % arm)):
        rows += list(csv.DictReader(open(f)))[3:]   # (the first updates: graphs, plans)
    out = {}
    for k in keys:
        v = [float(r[k]) for r in rows if r.get(k, "") != ""]
        if v:
            out[k] = round(s.median(v), 3)
    print(arm, len(rows), out)
EOF
../.venv/bin/python /tmp/pred_cmp.py
