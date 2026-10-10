#!/bin/bash
# Throughput test on the HPC node: run mode from C67's final weights with N workers for H game hours.
# usage: hpc_bench.sh N H [extra train_tok args]
N=${1:-64}; H=${2:-20}; shift 2
source /101063/AIsaac/ws1/hpc_env.sh
R=$ABP_HOME/runs
OUT=$R/bench-w$N${TAG:-}
PYTHONPATH=. timeout 3600 $PY train_tok.py --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers $N \
  --game-hours $H --epochs 3 --checkpoint-every 100000 --teacher --teacher-queue 12 --archive-size 12 --micro 1024 \
  --lr 1e-4 --lr-final 1e-4 --episodes-per-state 8 "$@" \
  --resume $R/c67-last.pt --name hb$N --port 40000 --out $OUT > $OUT.log 2>&1 < /dev/null
grep -c Traceback $OUT.log; grep "^finished\|^u[0-9]* " $OUT.log | tail -n 4 | cut -c1-200
python3 -c "
import csv
rows=list(csv.DictReader(open('$OUT/progress.csv')))
import statistics
xs=[float(r['x_real_time']) for r in rows[5:]]; cs=[float(r['collect_s']) for r in rows[5:]]; us=[float(r['update_s']) for r in rows[5:]]
print('workers $N updates', len(rows), 'x_real_time median %.0f max %.0f | collect %.2f s update %.2f s | gpu_reserved %s GB' % (statistics.median(xs), max(xs), statistics.median(cs), statistics.median(us), rows[-1].get('gpu_reserved_gb')))
"
