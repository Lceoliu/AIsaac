#!/bin/bash
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
V3_REF=9cc883d $PY -X utf8 scripts/run_remote_all.py status timNew > results/chainB-a3.log 2>&1; echo "a3 $?"
V3_REF=0c4b059 $PY -X utf8 scripts/run_remote_all.py timNew > results/chainB-a4.log 2>&1; echo "a4 $?"
V3_REF=2601f32 $PY -X utf8 scripts/run_remote_all.py actorgates timNew final > results/chainB-a1b.log 2>&1; echo "a1b $?"
