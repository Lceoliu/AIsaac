#!/bin/bash
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
until grep -q "no memory-v3 container left\|step failed\|own containers still running" results/chainK-t6.log; do sleep 60; done
V3_REF=208911e $PY -X utf8 scripts/run_remote_all.py status gates > results/chainL-gates.log 2>&1; G1=$?
V3_REF=208911e V3_GATES=rebuild-actor,actor-state,actor-crn,layer-ckpt $PY -X utf8 scripts/run_remote_all.py actorgates > results/chainL-actorgates.log 2>&1; G2=$?
echo "gates $G1 $G2"
if [ $G1 -eq 0 ] && [ $G2 -eq 0 ]; then V3_REF=208911e $PY -X utf8 scripts/run_remote_all.py t7 final > results/chainL-t7.log 2>&1; echo "t7 $?"; else $PY -X utf8 scripts/run_remote_all.py final > results/chainL-final.log 2>&1; echo "t7 skipped"; fi
