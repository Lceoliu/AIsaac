#!/bin/bash
# C102 "no-active starts" (B39, user 10-18): the C99 recipe (COMMON / FAST below = hpc_c99.sh's) continued from C99's
# segment2-final.pt (58,506 h) for 10,000 game hours with --no-active-prob 0.5: half the floor-start training episodes
# begin without Isaac's active item (The D6; whatever a character starts with), removed in the episode's fork before
# its first record (tok_pickup.NO_ACTIVE_LUA). A29: with the D6 removed at the AB+ start and nothing else changed, the
# C99 policy fell from 1.29 to 0.25 floors on 128 seeds (it never saw a start without an active item).
# Evaluation on two tracks per checkpoint: normal (128 seeds -> evals/<tag>, evals.jsonl) and no-D6 (eval_tok
# --no-active 1, the same 128 seeds -> evals-noactive/<tag>, evals-noactive.jsonl), run side by side by the watcher.
# B38 4-GPU layout (two actor processes on the 1st and 3rd visible GPUs, the learner process on the 2nd, --learner-fast
# 2, overlap + decoupled): run with CUDA_VISIBLE_DEVICES=0,1,2,3 (actors 0 and 2, learner 1, watcher 3).
# Minibatch: MB (default 4096 = C99; set MB=8192 if C101 says so) for --minibatch and --micro.
# Sandbox ws15 (worktree of the b39-no-active commit; mk_ws15.sh), own --out / name / ports (train 49200, watch
# 49450 / 49470, baseline 49500 / 49530, smoke 49650, probe 49690).
# usage: hpc_c102.sh smoke [extra] | probe | baseline | run [ck] | resume <abs ck> | watch
#   smoke: 8 workers, 1.5 h, from segment2-final.pt, 2-GPU process layout (SMOKE_LAYOUT; CUDA_VISIBLE_DEVICES=0,1)
#   probe: abplus_probe_no_active.py (first ROW with / without the active item, the starting room's restore point)
#   baseline: segment2-final.pt on both tracks (128 seeds, 16 workers each, side by side) -> $R/c102-baseline
source /101063/AIsaac/ws15/hpc_env.sh
R=$ABP_HOME/runs
CK0=/101063/AIsaac/ws9/runs/c99-scratch/checkpoints/segment2-final.pt
OUT=$R/c102-noactive
MB=${MB:-4096}
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
NOACT="--no-active-prob ${NOACT_P:-0.5}"
LAYOUT="--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2 --actor-processes 2 --actor-devices cuda:0,cuda:2"
# B39's starting-room restore point (tok_floor.START_BASE_LEGACY): 0 (default) = every changed start (no-active,
# pedestal build) restores from the episode itself; 1 = pedestal-build episodes restore from the unchanged start as in
# C99 (then C102 differs from C99 by the no-active starts only)
export ISAAC_RL_START_BASE_LEGACY=${START_BASE_LEGACY:-0}
SMOKE_LAYOUT=${SMOKE_LAYOUT:-"--learner-device cuda:1 --learner-process 1 --overlap --decoupled 1 --learner-fast 2"}

# eval_track <checkpoint> <out dir> <name> <port> <workers> [extra eval_tok args]: one 128-seed run-mode evaluation
eval_track() {
  local ck=$1 dir=$2 name=$3 port=$4 workers=$5
  shift 5
  PYTHONPATH=. timeout 7200 $PY eval_tok.py --checkpoint "$ck" --groups-file $GRP_FILE --mode run --items \
    --workers $workers --seeds 2147490000:128 --name $name --port $port --out "$dir" "$@" > "$dir.log" 2>&1 < /dev/null
}
# record <summary dir> <tag> <jsonl>: the summary plus its tag appended to the jsonl (nothing for a failed eval)
record() {
  [ -f "$1/summary.json" ] && python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); s['tag']=sys.argv[2]; print(json.dumps(s))" "$1/summary.json" "$2" >> "$3"
}
brief() {   # floors cleared / stage-4+ / no-active starts of a summary
  python3 -c "
import json,sys
s=json.load(open(sys.argv[1])); g=s['groups']['run']
st=g.get('stage_reached',{}); deep=sum(v for k,v in st.items() if int(k)>=4)
print(sys.argv[1].split('/')[-1], 'floors', round(g['floors_cleared'],3), 'stage4+', deep, 'death', round(g['death'],3),
      'timeout', round(g['timeout'],3), 'no_active', s.get('no_active', False), 'starts', s.get('no_active_starts'),
      'removed', s.get('no_active_removed'), 'errors', s.get('errors'), 'seconds', s.get('seconds'))" "$1/summary.json"
}

if [ "$1" = smoke ]; then
  shift
  tag=${TAG:-smoke-c102}
  PYTHONPATH=. timeout 1800 $PY train_tok.py $COMMON $NOACT $SMOKE_LAYOUT \
    --workers 8 --game-hours 1.5 --checkpoint-every 100000 --rollout 8192 --minibatch 2048 --micro 1024 "$@" \
    --resume "$CK0" --name hsm15 --port 49650 --out $R/$tag > $R/$tag.log 2>&1 < /dev/null
  echo "tracebacks $(grep -c Traceback $R/$tag.log)"; grep -i "devices (B38)\|learner process" $R/$tag.log | head -n 2 | cut -c1-200
  grep "^finished\|^u[0-9]* " $R/$tag.log | tail -n 3 | cut -c1-220
  python3 - "$R/$tag" <<'EOF'
import csv, json, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = list(csv.DictReader(open(d / 'progress.csv')))
last = rows[-1]
keys = ('update', 'game_hours', 'errors', 'no_actives', 'no_active_removed', 'start_pedestals', 'cf_points',
        'death_searches', 'noact/episodes', 'noact/exits', 'act/episodes', 'act/exits')
print('progress', len(rows), 'rows;', {k: last.get(k) for k in keys})
eps = [json.loads(l) for l in open(d / 'episodes.jsonl')] if (d / 'episodes.jsonl').exists() else []
a0 = [e.get('active0') for e in eps]
print('episodes', len(eps), 'active0=0', a0.count(0), 'active0=1', a0.count(1), 'none', a0.count(None),
      'both no-active and pedestal build', sum(e.get('active0') == 0 and bool(e.get('build_pedestals')) for e in eps))
EOF
  exit 0
fi
if [ "$1" = probe ]; then
  mkdir -p $R/c102-probe
  PYTHONPATH=. timeout 1800 $PY abplus_probe_no_active.py --groups-file $GRP_FILE --seeds 2147490000:2 --chars 0,3,5 \
    --name hb39p --port 49690 --out $R/c102-probe > $R/c102-probe.log 2>&1 < /dev/null
  grep "^SUMMARY" $R/c102-probe.log; echo "tracebacks $(grep -c Traceback $R/c102-probe.log)"
  exit 0
fi
if [ "$1" = baseline ]; then   # (C102's starting point: segment2-final.pt on both tracks, side by side)
  export OMP_NUM_THREADS=4
  B=$R/c102-baseline
  mkdir -p $B
  eval_track "$CK0" $B/seg2-final 102bn 49500 16 &
  eval_track "$CK0" $B/seg2-final-noactive 102ba 49530 16 --no-active 1 &
  wait
  record $B/seg2-final seg2-final $B/evals.jsonl
  record $B/seg2-final-noactive seg2-final $B/evals-noactive.jsonl
  for d in $B/seg2-final $B/seg2-final-noactive; do [ -f $d/summary.json ] && brief $d || echo "$d: no summary"; done
  exit 0
fi
if [ "$1" = run ]; then
  CK=${2:-$CK0}
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST $NOACT $LAYOUT \
    --workers 64 --game-hours 10000 --checkpoint-every 1000 --rollout 32768 --minibatch $MB --micro $MB \
    --resume "$CK" --name hc102 --port 49200 --out $OUT > $OUT.log 2>&1 < /dev/null &
  sleep 300
  date +%T; grep -i "devices (B38)" $OUT.log | tail -n 1 | cut -c1-160; tail -n 3 $OUT.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $OUT.log)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = resume ]; then   # (a restart: same flags, --resume the given ABSOLUTE checkpoint path into the same --out)
  CK=$2
  PYTHONPATH=. nohup $PY train_tok.py $COMMON $FAST $NOACT $LAYOUT \
    --workers 64 --game-hours 10000 --checkpoint-every 1000 --rollout 32768 --minibatch $MB --micro $MB \
    --resume "$CK" --name hc102 --port 49200 --out $OUT >> $OUT.log 2>&1 < /dev/null &
  sleep 300
  date +%T; tail -n 3 $OUT.log | cut -c1-220; echo "tracebacks $(grep -c Traceback $OUT.log)"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; uptime
  exit 0
fi
if [ "$1" = watch ]; then   # (each checkpoint on both tracks, side by side; 8 workers each)
  export OMP_NUM_THREADS=4
  mkdir -p $OUT/evals $OUT/evals-noactive
  while true; do
    for ck in $(ls $OUT/checkpoints/update-*.pt 2>/dev/null | sort); do
      tag=$(basename "$ck" .pt)
      [ -e "$OUT/evals/$tag" ] && [ -e "$OUT/evals-noactive/$tag" ] && continue
      run_n=0; run_a=0
      if [ ! -e "$OUT/evals/$tag" ]; then run_n=1; eval_track "$ck" "$OUT/evals/$tag" hc102e 49450 8 & fi
      if [ ! -e "$OUT/evals-noactive/$tag" ]; then
        run_a=1; eval_track "$ck" "$OUT/evals-noactive/$tag" hc102n 49470 8 --no-active 1 &
      fi
      wait
      [ $run_n = 1 ] && record "$OUT/evals/$tag" "$tag" "$OUT/evals.jsonl"
      [ $run_a = 1 ] && record "$OUT/evals-noactive/$tag" "$tag" "$OUT/evals-noactive.jsonl"
    done
    [ -f "$OUT/stop-evals" ] && break
    sleep 120
  done
fi
