#!/bin/bash
# Base (CRN fix), combo (A1+A2+A4+A1b+CRN, no A3), then the 5-card source-every-rollout long run.
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
for i in $(seq 60); do
  $PY -X utf8 bench_v3.py status > results/chainD-wait.log 2>&1
  grep -q "memory-v3" <(sed -n '/CONTAINERS/,$p' results/chainD-wait.log) || break
  sleep 60
done
V3_REF=afd843a V3_GATES=rebuild-actor,actorprof,actor-crn $PY -X utf8 scripts/run_remote_all.py status actorgates timNew > results/chainD-base.log 2>&1; echo "base $?"
V3_REF=14a7405 V3_GATES=rebuild-actor,actor-state,actorprof,actor-crn $PY -X utf8 scripts/run_remote_all.py actorgates timNew > results/chainD-combo.log 2>&1; echo "combo $?"
V3_REF=3b492fb $PY -X utf8 scripts/run_remote_all.py status tp5u1 final > results/chainD-long.log 2>&1; echo "long $?"
