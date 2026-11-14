#!/bin/sh
# gpu57 ab1 (2026-11-14): the two runs' first-rows checks (read-only): each rank's stdout "B52 cpp actor driver: on",
# guard.log's train_args as written, then once both have N (20) progress rows: per update x real, the losses finite, the
# heads' entropy, forced-pickup episodes, v2_loss_pi, ctr_mem_gb, broken episodes; ab1_check.py on episodes u1..N.
# Any miss: printed as MISS (nothing is stopped: the operator decides).
#   usage: check_ab1.sh [N=20]
N=${1:-20}
P=/root/AIsaac/train/ab1
rows() { [ -f $1 ] && echo $(($(wc -l < $1)-1)) || echo 0; }
t=0
while [ "$(rows $P/ab1-force/ab1-force-s0/progress.csv)" -lt $N ] || [ "$(rows $P/ab1-noforce/ab1-noforce-s0/progress.csv)" -lt $N ]; do
  sleep 20; t=$((t+20)); [ $t -ge 5400 ] && { echo "MISS: not $N rows in 5400 s"; break; }
  grep -q "ended\|STALL\|DRIVER OFF\|STOP" $P/ab1-*/guard.log && { echo "MISS: guard event"; break; }
done
for g in force noforce; do
  R=$P/ab1-$g
  echo "== ab1-$g"
  l=$(grep -a -m1 "B52 cpp actor driver:" $R/ab1-$g-s0.stdout | cut -c1-120); echo "driver: $l"; case "$l" in *"driver: on"*) ;; *) echo "MISS: driver";; esac
  echo "train_args: $(grep -m1 'train_args:' $R/guard.log | sed 's/.*train_args: //')"
  echo "tracebacks in stderr: $(grep -a -c Traceback $R/ab1-$g-s0.stderr)"
  python3 - $R/ab1-$g-s0/progress.csv $N <<'EOF'
import csv, math, sys
r = list(csv.DictReader(open(sys.argv[1])))[:int(sys.argv[2])]
bad = 0
for x in r:
    vals = {k: x.get(k, '') for k in ('pg', 'vf', 'kl', 'entropy', 'teach_loss', 'danger_loss', 'v2_loss_pi', 'aux_loss', 'grad_norm')}
    fin = all(math.isfinite(float(v)) for v in vals.values() if v not in ('', None))
    bad += not fin
    print(f"u{x['update']} gh {float(x['game_hours']):.1f} x {float(x['x_real_time']):.0f} pg {vals['pg'][:7]} vf {vals['vf'][:6]} kl {vals['kl'][:6]}"
          f" ent {float(x['ent_move']):.3f}/{float(x['ent_shoot']):.3f}/{float(x['ent_bomb']):.3f}/{float(x['ent_item']):.3f}/{float(x['ent_pill']):.3f}"
          f" teach {vals['teach_loss'][:6]} v2pi {vals['v2_loss_pi'][:6] or '-'} pf_eps {x.get('pickup_forced_eps')} mem {x.get('ctr_mem_gb', '')[:5]}"
          f" brk {x.get('broken_episodes')}{'' if fin else ' NOT FINITE'}")
xs = sorted(float(x['x_real_time']) for x in r[2:])
print(f"x real u3..u{len(r)}: median {xs[len(xs) // 2]:.0f}, min {xs[0]:.0f}, max {xs[-1]:.0f}; rows not finite {bad};"
      f" v2_loss_pi rows {sum(1 for x in r if x.get('v2_loss_pi'))}/{len(r)}")
if bad:
    print('MISS: losses not finite')
EOF
  python3 /root/AIsaac/bm1/ab1_check.py $R/ab1-$g-s0 1 $N
done
echo "CHECK_AB1 DONE"
