#!/bin/bash
# Host 2: the four arms' evaluation queue (rl/docs/PREDICTIVE_MEMORY_DESIGN.md section 2). pred_evalq.py relays checkpoints from the 8-card host into
# evalq/<case>/; this loop evaluates them one at a time, oldest first: update-*.pt on the 128 tracking seeds, final.pt on the 512 held-out seeds, then the
# delayed-cue probe on the 128-seed dump of final.pt (a diagnostic). Eight workers (with ten, intermittent Steam pipe stalls lost episodes, 2026-11-04).
# Exactly one of these on the host (flock). A result is <checkpoint>.eval/SUMMARY.json; delete nothing, the queue only looks for what has none.
cd ~/isaac-abplus/pred/python || exit 1
PY=~/isaac-abplus/.venv/bin/python
Q=~/isaac-abplus/pred/evalq
mkdir -p "$Q"
exec 9> "$Q/.lock"
flock -n 9 || { echo 'another queue is running'; exit 3; }
evaluate() {   # checkpoint seeds outdir [extra...]
  local ck=$1 seeds=$2 out=$3
  shift 3
  $PY -X utf8 eval_tok.py --checkpoint "$ck" --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 8 --seeds "$seeds" \
    --name pq --port 34700 --out "$out/eval" "$@" > "$out/eval.log" 2>&1
  local rc=$?
  grep -h '^SUMMARY ' "$out/eval.log" | tail -n 1 | sed 's/^SUMMARY //' > "$out/summary.tmp"
  return $rc
}
while [ ! -e "$Q/STOP" ]; do
  next=''
  for ck in $(ls -tr "$Q"/*/*.pt 2>/dev/null); do
    [ -s "$ck.eval/SUMMARY.json" ] || { next=$ck; break; }
  done
  if [ -z "$next" ]; then sleep 60; continue; fi
  out=$next.eval
  mkdir -p "$out"
  echo "$(date '+%F %T') $next"
  t0=$(date +%s)
  if [ "$(basename "$next")" = final.pt ]; then
    evaluate "$next" 2147490000:512 "$out"
  else
    evaluate "$next" 2147490000:128 "$out"
  fi
  rc=$?
  echo "$(date '+%F %T') $next rc=$rc $(( $(date +%s) - t0 )) s"
  if [ $rc -eq 0 ] && [ -s "$out/summary.tmp" ]; then
    mv "$out/summary.tmp" "$out/SUMMARY.json"
    if [ "$(basename "$next")" = final.pt ]; then
      mkdir -p "$out/probe128"
      $PY -X utf8 eval_tok.py --checkpoint "$next" --groups-file ../abplus/catalog/scaling2_groups.json --mode run --items --workers 8 --seeds 2147490000:128 \
        --name pq --port 34700 --dump-memory "$out/probe128/mem" --out "$out/probe128/eval" > "$out/probe128/eval.log" 2>&1
      $PY -X utf8 abplus_delay_probe.py "$out/probe128/mem" --out "$out/probe128/probe.json" > "$out/probe128/probe.log" 2>&1
    fi
  else
    # a failed evaluation stays visible and does not block the others: it is set aside once, for a look by hand
    mv "$next" "$next.failed"
  fi
done
