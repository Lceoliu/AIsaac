#!/bin/bash
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
V3_REF=1727f71 $PY -X utf8 scripts/run_remote_all.py status gates > results/chainM-gates.log 2>&1; G1=$?
V3_REF=1727f71 V3_GATES=rebuild-actor,actor-state,actor-crn,layer-ckpt $PY -X utf8 scripts/run_remote_all.py actorgates > results/chainM-actorgates.log 2>&1; G2=$?
echo "gates $G1 $G2"
if [ $G1 -eq 0 ] && [ $G2 -eq 0 ]; then V3_REF=1727f71 $PY -X utf8 scripts/run_remote_all.py t7 final > results/chainM-t7b.log 2>&1; echo "t7b $?"; else $PY -X utf8 scripts/run_remote_all.py final > results/chainM-final.log 2>&1; echo "t7b skipped"; fi
