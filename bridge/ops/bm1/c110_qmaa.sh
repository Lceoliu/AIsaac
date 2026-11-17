#!/bin/sh
# gpu57 C110 Q@maa (2026-11-15, the coordinator): the old model (--block-memory 0 --temporal 1 --obs-v2 1) on tree
# maa1e98a-nv (maa1e98a + ISAAC_NO_VISIBLE=1 in tok_obs.encode_row: no visible frame, schema 0; c110/nv_tree.diff),
# ab2-bm's train_args with pickup-force 1:0.984 (c110/qm.train_args), lane a (GPUs 4 actor + 5 learner, node0).
# The smoke and its checks, one after the other; the 400-hour run is started by hand after a person reads them:
#   1. smoke c110smoke/c110s-qm (HOURS 8, CKPT_EVERY 5, c108_guard.sh; guard.pre: xnew2.sh, REQUIRE_DRIVER=0,
#      ENVS ISAAC_NO_VISIBLE=1 + ISAAC_RL_LOG_GRAD_NORM=1);
#   2. c110_gate.py (hours, error counters, Tracebacks, learner < 23 GB) -> c110/gate_c110s-qm.txt; failing: stop;
#   3. D1 lr-0 (tree maa1e98a-nv-d1, resumed from the smoke's last.pt: also the resume check) with --dump-rollouts 2
#      -> c108/d1/qm-smoke-last/ (d1_diag.jsonl, rollouts/);
#   4. pending[t+1] == action[t]: c108/d2_actions.py table on those rollouts -> c110/qm_check/actions.json;
#   5. a 4-seed evaluation of the smoke's last.pt with ISAAC_NO_VISIBLE=1 on maa1e98a-nv -> c110smoke/eval/c110s-qm-c4/.
# Status lines in c110/QM_STATUS and bm1.log.
#   usage: setsid nohup sh c110_qmaa.sh > /root/AIsaac/train/c110/qm_chain.out 2>&1 < /dev/null &
P=/root/AIsaac/train/c110; S=/root/AIsaac/train/c110smoke; BM=/root/AIsaac/bm1
say() { echo "$(date '+%F %T') $*" | tee -a $P/QM_STATUS; echo "$(date '+%F %T') c110_qmaa: $*" >> $BM/reports/bm1.log; }
LANE=env/gpu57-c110-a.env; T=maa1e98a-nv
cd $BM
say "smoke c110s-qm on $LANE, tree $T"
HOURS=8 CKPT_EVERY=5 MEM_GB=96 sh $BM/c108_guard.sh c110smoke c110s-qm $T $LANE
say "smoke ended: $(tail -2 $S/c110s-qm/guard.log | head -1 | cut -c1-160)"
if ! python3 $BM/c110_gate.py $S/c110s-qm c110s-qm > $P/gate_c110s-qm.txt 2>&1; then
  say "smoke gate FAILED ($P/gate_c110s-qm.txt): stopped (a person looks)"; exit 1
fi
cp $S/c110s-qm/c110s-qm-s0/checkpoints/last.pt /root/AIsaac/train/c108/d1/inputs/qm-smoke-last.pt
say "gate passed; D1 lr-0 (qm-smoke-last, + --dump-rollouts 2) on GPUs 4,5"
D1_CODE=maa1e98a-nv-d1 D1_ARGS=$P/qm.train_args D1_GPUS=4,5 D1_CPUSET=0-7,16-23 D1_MEMS=0 \
  sh $BM/c108/d1_run.sh qm-smoke-last:new:qm-smoke-last.pt:--dump-rollouts,2:ISAAC_NO_VISIBLE=1 > $P/d1_qm.out 2>&1
say "D1 done: $(tail -2 $P/d1_qm.out | head -1 | cut -c1-160)"
mkdir -p $P/qm_check
L=$(GPU=4 CPUSET=0-7,16-23 MEMS=0 MEM=32g sh $BM/job.sh qm-pending "cd /bench/train/c110/qm_check && \$BM1_PY -u /bench/bm1/c108/d2_actions.py table actions.json qm-prod=/bench/train/c108/d1/qm-smoke-last/rollouts" | tail -1)
cp "$L" $P/qm_check/job.log 2>/dev/null
say "pending table: $(ls $P/qm_check/actions.json 2>/dev/null || echo none) ($L)"
PARENT=c110smoke EV_TREE=$T EV_ENVS="-e ISAAC_NO_VISIBLE=1" EV_PORT=41600 \
  sh $BM/ev_ab2.sh c110s-qm new 4 0-7,16-23 0 4 c110s-qm-c4,last.pt,2147490000:4 > $P/ev_qm_smoke.out 2>&1
say "4-seed eval: $(tail -2 $P/ev_qm_smoke.out | head -1 | cut -c1-160); chain over (the run: by hand)"
