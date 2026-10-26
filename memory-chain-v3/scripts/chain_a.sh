#!/bin/bash
# Actor A-line timing chain: wait for the running memC orchestrator, then base / A1 / A2 / old timers.
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
until grep -q "no memory-v3 container left\|step failed\|own containers still running" results/memC-orchestrator.log; do sleep 60; done
V3_REF=805868e $PY -X utf8 scripts/run_remote_all.py status timNew > results/chainA-base.log 2>&1; echo "base $?"
V3_REF=fe05db2 $PY -X utf8 scripts/run_remote_all.py actorgates timNew > results/chainA-a1.log 2>&1; echo "a1 $?"
V3_REF=3c100d5 $PY -X utf8 scripts/run_remote_all.py actorgates timNew > results/chainA-a2.log 2>&1; echo "a2 $?"
$PY -X utf8 scripts/run_remote_all.py timOld final > results/chainA-old.log 2>&1; echo "old $?"
