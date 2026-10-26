#!/bin/bash
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
until grep -q "^h1024 " C:/Users/eolc/AppData/Local/Temp/claude/D--Projects-fortune-Isaac/dbd5252c-1fb9-4fcc-9cd7-76b3f781e743/tasks/bbmvbv6oi.output; do sleep 60; done
V3_REF=419fce9 $PY -X utf8 scripts/run_remote_all.py status gates > results/chainI-gates.log 2>&1; G1=$?
V3_REF=419fce9 V3_GATES=rebuild-actor,actor-state,actor-crn $PY -X utf8 scripts/run_remote_all.py actorgates > results/chainI-actorgates.log 2>&1; G2=$?
echo "gates $G1 $G2"
if [ $G1 -eq 0 ] && [ $G2 -eq 0 ]; then V3_REF=419fce9 $PY -X utf8 scripts/run_remote_all.py tp5u1h1024d final > results/chainI-long.log 2>&1; echo "t5 $?"; else $PY -X utf8 scripts/run_remote_all.py final > results/chainI-final.log 2>&1; echo "t5 skipped"; fi
