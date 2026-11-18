#!/bin/bash
# C103 "bomb walls" (B40, user 10-18 "先准备(b)"): the C99 recipe (COMMON / FAST below = hpc_c99.sh's) continued from
# C99's segment2-final.pt (58,506 h) for 10,000 game hours with the counterfactual teacher's wall trigger (--cf-wall 1:
# a cleared room, a bomb held, a wall without a door whose minimap neighbour is unknown -> a branch that walks to the
# wall's middle and places one bomb, then the policy; no reward term, the realised return decides) and A30's deferred
# door targets (--cf-defer-target 1: the target branch of a locked / shop / treasure / cleared door trigger and the
# bomb_at branch play the policy until the room is cleared, then walk; distilled from there). Control = C99 segment 2
# (and C102) at equal game hours from the same checkpoint; set DEFER=0 to keep B34's door targets (wall trigger only).
# Also B40 (user): --stall-progress pickups (a gain in coins / keys / bombs / health resets the floor's 60-s stall timer
# too, so collecting and bombing are not cut off; the stall share is a policy input; STALL_PROGRESS=events = as C99).
# The watcher evaluates with EVAL_STALL (default events = C99's rule, comparable curves; pickups = the training rule).
# B38 4-GPU layout (two actor processes on the 1st and 3rd visible GPUs, the learner process on the 2nd, --learner-fast
# 2, overlap + decoupled): run with CUDA_VISIBLE_DEVICES=0,1,2,3 (actors 0 and 2, learner 1, watcher 3).
# Minibatch: MB (default 8192, user 10-18) for --minibatch and --micro. Budget 20,000 h (user 10-18).
# Sandbox ws16 (worktree of the b40-bomb-trigger commit; mk_ws16.sh), own --out / name / ports (train 49250, watch
# 49480, smoke 49660, probe 49695).
# usage: hpc_c103.sh smoke [extra] | probe | run [ck] | resume <abs ck> | watch
#   smoke: 8 workers, ~60 updates (18.3 game h at rollout 8192), from segment2-final.pt, 2-GPU process layout
#          (SMOKE_LAYOUT; CUDA_VISIBLE_DEVICES=0,1); prints the cf_wall_* / cf_defer_* columns
#   probe: abplus_probe_wall.py (seeds with a secret room next to the start: the bomb_at branch opens it; controls)
source /101063/AIsaac/ws16/hpc_env.sh
R=$ABP_HOME/runs
CK0=/101063/AIsaac/ws9/runs/c99-scratch/checkpoints/segment2-final.pt
OUT=$R/c103-wall
MB=${MB:-8192}   # 2026-10-18 (user): 8192 (C101 leads 4096 by 0.22 floors at equal hours)
GRP_FILE=../abplus/catalog/scaling2_groups.json
COMMON="--groups-file $GRP_FILE --mode run --items --epochs 3 --teacher --teacher-queue 12 --archive-size 12 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 --teacher-death-depths 2,4,8,16,32 --teacher-thread 1 --teacher-fork 1 \
  --teacher-share 0.1 --teacher-slots 6 --poll-wait 0.003 \
  --item-tokens 1 --global-token 1 --item-shared 1 --start-build-prob 0.5 --start-build-max 3 --start-build-mode pedestal \
  --pickup-force 1:0 --reward heal=0.5,resource=0.25 --item-slots 8 \
  --temporal 1 --temporal-width 256 --temporal-layers 4 --temporal-heads 4 --temporal-window 256 --temporal-rooms 128 --temporal-bptt 256 \
  --obs-v2 1 \
  --cf 1 --cf-points 6 --cf-branches 4 --cf-seconds 180 --cf-target-seconds 20 --cf-margin 0.3 --cf-cap 2 \
  --cf-distill-steps 8 --cf-random 1 --cf-advantage 0"
FAST="--actor-compile default --actor-fold 1 --actor-sdpa-math 1 --graph-entities-extra 16,40 --actor-row-buckets 16,32"
WALL="--cf-wall 1 --cf-wall-distill 64 --cf-defer-target ${DEFER:-1} --stall-progress ${STALL_PROGRESS:-pickups}"
LAYOUT="--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2 --actor-processes 2 --actor-devices cuda:0,cuda:2"
SMOKE_LAYOUT=${SMOKE_LAYOUT:-"--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2"}

if [ "$1" = smoke ]; then
  shift
  tag=${TAG:-smoke-c103}
  PYTHONPATH=. timeout 3600 $PY train_tok.py $COMMON $WALL $SMOKE_LAYOUT \
    --workers 8 --game-hours ${SMOKE_HOURS:-18.3} --checkpoint-every 100000 --rollout 8192 --minibatch 2048 --micro 1024 "$@" \
    --resume "$CK0" --name hsm16 --port 49660 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "devices (B38)\|learner process" $R/$tag.log | head -n 2 | cut -c1-200
  grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-220
  python3 - "$R/$tag" <<'EOF'
import csv, sys
from pathlib import Path
rows = list(csv.DictReader(open(Path(sys.argv[1]) / 'progress.csv')))
last = rows[-1]
keys = ('update', 'game_hours', 'x_real_time', 'errors', 'cf_points', 'cf_done', 'cf_improving', 'cf_errors',
        'cf_points/wall', 'cf_wall_done', 'cf_wall_bombed', 'cf_wall_opened', 'cf_wall_best', 'cf_wall_gain', 'cf_wall_p',
        'cf_defer_points', 'cf_defer_started', 'cf_defer_wait', 'cf_records', 'cf_points/treasure', 'cf_points/shop',
        'cf_points/locked', 'cf_points/cleared')
print('progress', len(rows), 'rows;', {k: last.get(k) for k in keys})
for k in ('cf_prior/wall', 'cf_wall_opened_share', 'cf_wall_bombed_share', 'cf_wall_gain_mean', 'cf_wall_best_share',
          'cf_wall_p_mean', 'cf_defer_started_share', 'cf_defer_wait_s', 'cf_improving_share', 'cf_prior'):
    vals = [float(r[k]) for r in rows if r.get(k) not in (None, '')]
    print(k, 'n', len(vals), 'mean', round(sum(vals) / len(vals), 4) if vals else None, 'last', vals[-1] if vals else None)
EOF
  exit 0
fi
if [ "$1" = probe ]; then
  mkdir -p $R/c103-probe
  PYTHONPATH=. timeout 3600 $PY abplus_probe_wall.py --groups-file $GRP_FILE --seeds ${PROBE_SEEDS:-2147490000:80} \
    --want 4 --controls 2 --name hb40p --port 49695 --out $R/c103-probe > $R/c103-probe.log 2>&1 < /dev/null
  grep "^SUMMARY" $R/c103-probe.log; echo "tracebacks $(grep -c Traceback $R/c103-probe.log)"
  exit 0
fi
if [ "$1" = run ]; then
  CK=${2:-$CK0}
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST $WALL $LAYOUT \
    --workers 64 --game-hours 20000 --checkpoint-every 1000 --rollout 32768 --minibatch $MB --micro $MB \
    --resume "$CK" --name hc103 --port 49250 --out $OUT > $OUT.log 2>&1 < /dev/null &
  sleep 300
  date +%T; grep -i "devices (B38)" $OUT.log | tail -n 1 | cut -c1-160; tail -n 3 $OUT.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $OUT.log)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = resume ]; then   # (a restart: same flags, --resume the given ABSOLUTE checkpoint path into the same --out)
  CK=$2
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST $WALL $LAYOUT \
    --workers 64 --game-hours 20000 --checkpoint-every 1000 --rollout 32768 --minibatch $MB --micro $MB \
    --resume "$CK" --name hc103 --port 49250 --out $OUT >> $OUT.log 2>&1 < /dev/null &
  sleep 300
  date +%T; tail -n 3 $OUT.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $OUT.log)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then   # (each checkpoint on the 128 seeds of C99 / C101 / C102, 8 workers)
  export OMP_NUM_THREADS=4
  mkdir -p $OUT/evals
  while true; do
    for ck in $(ls $OUT/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$OUT/evals/$tag" ] && continue
      PYTHONPATH=. timeout 7200 $PY eval_tok.py --checkpoint "$ck" --groups-file $GRP_FILE --mode run --items \
        --workers 8 --seeds 2147490000:128 --stall-progress ${EVAL_STALL:-events} --name hc103e --port 49480 \
        --out "$OUT/evals/$tag" \
        > "$OUT/evals/$tag.log" 2>&1 < /dev/null
      [ -f "$OUT/evals/$tag/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$OUT/evals/$tag/summary.json" "$tag" >> "$OUT/evals.jsonl"
    done
    [ -f "$OUT/stop-evals" ] && break
    sleep 120
  done
fi
