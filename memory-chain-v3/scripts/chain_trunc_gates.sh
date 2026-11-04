# truncated memory (2026-10-30, restarted 11-03): deploy + CUDA gates on 3,4,5, then the 3-card smoke on 3,1,4
set -x
cd D:/Projects/fortune/Isaac/temp/memory-chain-v3
python -X utf8 scripts/wait_then_run.py --allow 1,2 --hours 12 $1 deploy truncgates
python -X utf8 scripts/wait_then_run.py --allow 2,5 --hours 12 $1 trs final
