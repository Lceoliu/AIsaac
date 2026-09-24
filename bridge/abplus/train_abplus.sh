#!/usr/bin/env bash
# AB+ room-mixture PPO run on the remote host (rl/bridge/abplus/README.md, "训练管线").
#
# usage: train_abplus.sh <run-name> [--wait-for-sim] [extra train_abplus.py arguments]
#   --wait-for-sim  start only after the simulator training (train_sim.py --train) has exited,
#                   so the learner gets the GPU to itself.
# Environment: ABP_HOME (~/isaac-abplus), ABP_PYTHON (the isaac-rl venv, read-only use),
#   WARM_START (default: latest checkpoint of the simulator run SIM_RUN).
# The simulator's run directory is only read (warm start); outputs go to $ABP_HOME/train/<run-name>.
set -euo pipefail
RUN=${1:?run name}
shift
WAIT=0
if [[ "${1:-}" == "--wait-for-sim" ]]; then
  WAIT=1
  shift
fi
ABP_HOME=${ABP_HOME:-$HOME/isaac-abplus}
PY=${ABP_PYTHON:-/home/eolc/isaac-rl/.venv/bin/python}
SIM_RUN=${SIM_RUN:-/home/eolc/isaac-rl/runs/monstro-128k-seg1024-20260923}
if (( WAIT )); then
  echo "$(date -Is) waiting for the simulator training to exit"
  while pgrep -f "train_sim.py --train" > /dev/null; do sleep 60; done
  echo "$(date -Is) simulator training has exited"
fi
WARM=${WARM_START:-$SIM_RUN/checkpoints/latest.json}
mkdir -p "$ABP_HOME/train"
cd "$ABP_HOME/bridge/python"
exec env PYTHONDONTWRITEBYTECODE=1 "$PY" -u train_abplus.py --train --out "$ABP_HOME/train/$RUN" \
  --warm-start "$WARM" --envs 16 --chunks 1 --n-steps 1024 --batch-size 1024 --micro-batch 256 \
  --segment-length 32 --n-epochs 2 --ent-coef 0.01 --nice 0 "$@"
