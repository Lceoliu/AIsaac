#!/bin/bash
cd /d/Projects/fortune/Isaac/temp/memory-chain-v3
PY=C:/Users/eolc/miniconda3/python.exe
until grep -q "^ck " C:/Users/eolc/AppData/Local/Temp/claude/D--Projects-fortune-Isaac/dbd5252c-1fb9-4fcc-9cd7-76b3f781e743/tasks/brj6uffbo.output; do sleep 60; done
V3_REF=7d9c227 $PY -X utf8 scripts/run_remote_all.py status tp5u1h1024 final > results/chainH-long1024.log 2>&1; echo "h1024 $?"
