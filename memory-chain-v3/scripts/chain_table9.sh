# Table 9: wait up to 2 h (every 10 min) for GPU2 (others' work); 5 cards (t9t) if it frees, else 4 cards (t9t4).
cd D:/Projects/fortune/Isaac/temp/memory-chain-v3
R=1c4fc72
case=t9t4
for i in $(seq 1 12); do
  if V3_REF=$R python -X utf8 scripts/run_remote_all.py status > results/wait-t9-$R.log 2>&1; then case=t9t; break; fi
  sleep 600
done
echo "CASE $case"
V3_REF=$R python -X utf8 scripts/run_remote_all.py $case final; echo "EXIT9 $?"
