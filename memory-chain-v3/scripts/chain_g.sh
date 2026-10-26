#!/bin/bash
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
V3_REF=7d9c227 V3_GATES=layer-ckpt $PY -X utf8 scripts/run_remote_all.py status actorgates > results/chainG-gate.log 2>&1; G=$?; echo "gate $G"
V3_REF=7d9c227 $PY -X utf8 scripts/run_remote_all.py tlU1 > results/chainG-tl.log 2>&1; echo "tl $?"
if [ $G -eq 0 ]; then V3_REF=7d9c227 $PY -X utf8 scripts/run_remote_all.py ckU1 final > results/chainG-ck.log 2>&1; echo "ck $?"; else $PY -X utf8 scripts/run_remote_all.py final > results/chainG-final.log 2>&1; echo "ck skipped"; fi
