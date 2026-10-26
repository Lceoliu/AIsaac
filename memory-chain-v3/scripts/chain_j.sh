#!/bin/bash
# wait for table 5 (419fce9, launched by the stopped chain_i), collect it, then table 4's configuration with the hold accounting
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
C5=v3-419fce9-tp5u1h1024d-10250833
for i in $(seq 90); do $PY -X utf8 bench_v3.py status > results/chainJ-wait.log 2>&1; grep -q "memory-v3" <(sed -n '/CONTAINERS/,$p' results/chainJ-wait.log) || break; sleep 60; done
$PY -X utf8 bench_v3.py collect $C5 > results/chainJ-collect5.log 2>&1; echo "t5 collected $?"
V3_REF=b657026 $PY -X utf8 scripts/run_remote_all.py status tp5u1h1024 final > results/chainJ-hold.log 2>&1; echo "hold $?"
