#!/bin/bash
# Host 2: abplus_world_model.py end to end on the short logs of the timing runs: prep, wm / latent / base, verdict. A tool check: these logs
# hold ~100k decisions, nothing here is a result. usage: bash host2_wm_smoke.sh <tag> [steps]
tag=${1:?tag}
steps=${2:-1500}
cd ~/isaac-abplus/pred/python || exit 1
PY=~/isaac-abplus/.venv/bin/python
D=../runs/wm-$tag
mkdir -p "$D"
$PY -X utf8 abplus_world_model.py prep ../runs/t4-l0-1/stream ../runs/t4-l0-2/stream --out "$D/data" > "$D/prep.log" 2>&1 || exit 2
for arm in wm latent base; do
  /usr/bin/time -v $PY -X utf8 abplus_world_model.py train --data "$D/data" --arm $arm --out "$D/$arm" --steps "$steps" --eval-every "$steps" --eval-windows 1500 > "$D/$arm.log" 2>&1 || exit 3
done
$PY -X utf8 abplus_world_model.py verdict --wm "$D/wm" --base "$D/base" --latent "$D/latent" --out "$D/verdict.json" > "$D/verdict.log" 2>&1
echo done $?
