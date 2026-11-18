#!/bin/bash
# Inventory of what on the 3080 (host 1) is worth migrating before the machine goes away (user 2026-10-13).
# Tier 1 = final checkpoints + run records (progress/evals/episodes/logs/summaries), ev128 results, gx data.
# Tier 2 = every checkpoint of the key experiment lines. Prints sizes; writes the tier-1 file list to /tmp/host1_tier1.txt.
cd /home/eolc
echo "== top-level sizes"
du -sh isaac-abplus/experiments isaac-abplus/train isaac-abplus/offline-20261008 isaac-abplus/ovl isaac-rl/runs 2>/dev/null
echo "== isaac-abplus/experiments"; ls isaac-abplus/experiments | head -n 20
echo "== isaac-abplus/train"; ls isaac-abplus/train | head -n 20
echo "== isaac-abplus/offline-20261008"; ls isaac-abplus/offline-20261008 | head -n 10
echo "== isaac-rl/runs (sizes)"; du -sh isaac-rl/runs/* 2>/dev/null | sort -rh | head -n 25
echo "== tier 1 list"
: > /tmp/host1_tier1.txt
for r in isaac-abplus/*/runs/* isaac-rl/runs/*; do
  [ -d "$r" ] || continue
  for f in "$r"/checkpoints/last.pt "$r"/checkpoints/from-*.pt "$r"/progress.csv "$r"/evals.jsonl "$r"/episodes.jsonl "$r"/episodes.json "$r"/summary.json "$r"/config.json "$r"/args.json; do
    [ -f "$f" ] && echo "$f" >> /tmp/host1_tier1.txt
  done
  [ -d "$r/evals" ] && find "$r/evals" -type f -name "*.json" >> /tmp/host1_tier1.txt
done
find isaac-abplus/*/runs -maxdepth 1 -type f \( -name "*.log" -o -name "*.jsonl" -o -name "*.json" -o -name "*.txt" -o -name "*.sh" -o -name "*.pt" \) >> /tmp/host1_tier1.txt
find isaac-abplus/bld/runs/ev128 -type f >> /tmp/host1_tier1.txt
find isaac-abplus/gx -type f 2>/dev/null >> /tmp/host1_tier1.txt
sort -u /tmp/host1_tier1.txt -o /tmp/host1_tier1.txt
echo "tier-1 files: $(wc -l < /tmp/host1_tier1.txt)  size: $(cat /tmp/host1_tier1.txt | tr '\n' '\0' | du -ch --files0-from=- 2>/dev/null | tail -n 1 | cut -f1)"
echo "== tier 2 (all checkpoints of key lines)"
du -shc isaac-abplus/fk2/runs/c67-run isaac-abplus/ch1/runs/c74-chars isaac-abplus/dt/runs/c75-death isaac-abplus/lf/runs/c80-fastlearner isaac-abplus/tv/runs/c83-teacher2 isaac-abplus/ch2/runs/c86-chars2 isaac-abplus/mv2/runs/c89-v2mem 2>/dev/null | tail -n 1
echo "== reachability from the 3080"
timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 -p 30575 root@10.15.171.204 'echo hpc160 reachable' 2>&1 | tail -n 1
timeout 8 ssh -o BatchMode=yes -o ConnectTimeout=5 rhythmo@10.19.131.132 'echo 3090 reachable' 2>&1 | tail -n 1
