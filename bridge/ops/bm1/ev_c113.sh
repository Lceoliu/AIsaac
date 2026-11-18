#!/bin/sh
# C113 held-out evaluation (dgene, 2026-11-17): ev_ab2.sh's kind new made host-independent (bm1.env), C111's protocol:
# eval_tok.py through evtools/eval_ent.py (+ entropy.json, config.json), --mode run --items, TOK_EVAL_DEVICE=cuda,
# --workers <workers>, the run's own tree and switches (EV_TREE, EV_ENVS="-e K=V ..."). One container job (job.sh:
# shared lock) on the given GPU / CPUs / memory node; a finished one (summary.json) is skipped.
#   usage: [EV_TREE=maa1e98a-rw4] [EV_ENVS=...] ev_c113.sh <run dir under $BM1_TRAIN> <gpu> <cpuset> <mems> <workers> <tag>,<checkpoint file>,<first:count>
set -u
BM=${BM:-$(cd "$(dirname "$0")" && pwd)}; . $BM/bm1.env
RD=$1; G=$2; CS=$3; MS=$4; W=$5; spec=$6
N=$(basename $RD); ET=${EV_TREE:-maa1e98a-rw4}
HOST=$BM1_TRAIN/$(dirname $RD)/eval; IN=$BM1_TRAIN_IN/$(dirname $RD)/eval
mkdir -p $HOST
tag=$(echo "$spec" | cut -d, -f1); ck=$(echo "$spec" | cut -d, -f2); seeds=$(echo "$spec" | cut -d, -f3)
echo "$(date '+%F %T') ev_c113: $RD GPU $G CPUs $CS workers $W tree $ET env '${EV_ENVS:-}': $spec" >> $BM/reports/bm1.log
[ -f $HOST/$tag/summary.json ] && { echo "skip $tag (done)"; exit 0; }
[ -e $HOST/$tag ] && { echo "skip $tag: $HOST/$tag exists without summary.json (a failed run: look)"; exit 1; }
[ -f $BM1_TRAIN/$RD/$N-s0/checkpoints/$ck ] || { echo "no checkpoint $ck"; exit 1; }
CK=$BM1_TRAIN_IN/$RD/$N-s0/checkpoints/$ck
X="xvfb-run -a -s '-screen 0 640x480x24 -nolisten tcp -noreset'"
t0=$(date +%s)
L=$(GPU=$G CPUSET=$CS MEMS=$MS MEM=${EV_MEM:-48g} ENVS="-e TOK_EVAL_DEVICE=cuda ${EV_ENVS:-}" sh $BM/job.sh ev-$N "env | grep ^ISAAC_ | sort; cd $BM1_IN/trees/$ET/bridge/python && export LIBGL_ALWAYS_SOFTWARE=1 SDL_AUDIODRIVER=dummy ALSOFT_DRIVERS=null && $X \$BM1_PY -u $BM1_IN/evtools/eval_ent.py --checkpoint $CK --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers $W --seeds $seeds --name abev --port ${EV_PORT:-41300} --out $IN/$tag" | tail -1)
cp "$L" $HOST/$tag.log 2>/dev/null; cp "$L.rc" $HOST/$tag.log.rc 2>/dev/null
# (2026-11-18: complete only with every requested seed played and the job's rc 0 -- evtools/ev_complete.py)
echo "$(date '+%F %T') $tag $ck $seeds: $(( $(date +%s) - t0 )) s, $(python3 $BM/evtools/ev_complete.py --dir $HOST/$tag --seeds $seeds --rc-file $HOST/$tag.log.rc) ($L)"
