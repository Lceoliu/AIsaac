#!/bin/sh
# gpu57 C110 gate arm (2026-11-15, the coordinator): the block stack with the three use heads closed
# (ISAAC_BLOCK_GATE=closed, tree maa1e98a-gate: structured_input.action_gate all-closed; c110/gate_tree.diff),
# ab2-bm's train_args with pickup-force 1:0.984 (c110/g.train_args), lane b (GPUs 6 actor + 7 learner, node1).
# Waits for c110/AFTER_DONE_c110-f (fp32 trained and evaluated: lane b free), then one after the other:
#   1. smoke c110smoke/c110s-g (HOURS 8, CKPT_EVERY 5, c108_guard.sh; guard.pre ENVS ISAAC_BLOCK_GATE=closed, driver on);
#   2. c110_gate.py --driver on -> c110/gate_c110s-g.txt; failing: stop;
#   3. D1 lr-0 (tree maa1e98a-gate-d1, from the smoke's last.pt) with --dump-rollouts 2 -> c108/d1/g-smoke-last/;
#   4. use actions in those rollouts (c110_usecount.py) -> c110/g_check/usecount.txt.
# The 400-hour run is started by hand after a person reads them. Status in c110/G_STATUS and bm1.log.
#   usage: setsid nohup sh c110_gatearm.sh > /root/AIsaac/train/c110/g_chain.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c110; S=/root/AIsaac/train/c110smoke; BM=/root/AIsaac/bm1
say() { echo "$(date '+%F %T') $*" | tee -a $P/G_STATUS; echo "$(date '+%F %T') c110_gatearm: $*" >> $BM/reports/bm1.log; }
LANE=env/gpu57-c110-b.env; T=maa1e98a-gate
say "waiting for AFTER_DONE_c110-f"
while [ ! -f $P/AFTER_DONE_c110-f ]; do sleep 60; done
say "fp32 over ($(cat $P/AFTER_DONE_c110-f)); smoke c110s-g on $LANE, tree $T"
cd $BM
mkdir -p $S/c110s-g && cp $P/g.train_args $S/c110s-g/train_args && cp $P/g.guard.pre $S/c110s-g/guard.pre
HOURS=8 CKPT_EVERY=5 MEM_GB=96 sh $BM/c108_guard.sh c110smoke c110s-g $T $LANE
say "smoke ended: $(tail -2 $S/c110s-g/guard.log | head -1 | cut -c1-160)"
if ! python3 $BM/c110_gate.py $S/c110s-g c110s-g --driver on > $P/gate_c110s-g.txt 2>&1; then
  say "smoke gate FAILED ($P/gate_c110s-g.txt): stopped (a person looks)"; exit 1
fi
cp $S/c110s-g/c110s-g-s0/checkpoints/last.pt /root/AIsaac/train/c108/d1/inputs/g-smoke-last.pt
say "gate passed; D1 lr-0 (g-smoke-last, + --dump-rollouts 2) on GPUs 6,7"
D1_CODE=maa1e98a-gate-d1 D1_ARGS=$P/g.train_args D1_GPUS=6,7 D1_CPUSET=8-15,24-31 D1_MEMS=1 \
  sh $BM/c108/d1_run.sh g-smoke-last:new:g-smoke-last.pt:--dump-rollouts,2:ISAAC_BLOCK_GATE=closed > $P/d1_g.out 2>&1
say "D1 done: $(tail -2 $P/d1_g.out | head -1 | cut -c1-160)"
mkdir -p $P/g_check
L=$(GPU=6 CPUSET=8-15,24-31 MEMS=1 MEM=32g sh $BM/job.sh g-usecount "\$BM1_PY -u /bench/bm1/c110_usecount.py /bench/train/c108/d1/g-smoke-last/rollouts > /bench/train/c110/g_check/usecount.txt 2>&1" | tail -1)
say "use count: $(tail -1 $P/g_check/usecount.txt 2>/dev/null) ($L); chain over (the run: by hand)"
