#!/bin/sh
# bm1: regression test of common.sh's reader-writer lock, in a scratch directory (nothing else touched):
#   1. three shared holders at once: three different lock files, each naming its own holder's pid;
#   2. two shared holders (A ends first, B later) and a writer waiting: after A ends the writer still waits; it takes
#      the lock only after B ends;
#   3. a caller's variables (f, n, p, t0, a) survive take_shared / take_excl / give;
#   4. bounded wait; 5. writer first kept;
#   6. a holder killed while its registered container still runs keeps the lock (a stand-in process for the container,
#      BM1_CONTAINER_ALIVE): a shared taker waits until the stand-in ends, then takes it.
#   usage: test_lock.sh <scratch dir (created, left for inspection)>        exit 0: pass
set -u
BMSRC=$(cd "$(dirname "$0")" && pwd)
T=$1
mkdir -p "$T"
T=$(cd "$T" && pwd)
cp "$BMSRC/common.sh" "$T/common.sh"
printf 'BM1_LOCKDIR=%s/locks\nBM1_LOCK_POLL=1\nBM1_SESSION=locktest\n' "$T" > "$T/bm1.env"
cat >> "$T/bm1.env" <<'EOF'
BM1_CONTAINER_ALIVE='kill -0 "${1#stub-}" 2>/dev/null'
EOF
fail=0
ck() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
holder() {   # holder <name> <seconds>: take_shared, record the lock file name and the time, hold, give
  BM=$T sh -c ". $T/common.sh; take_shared 'test $1'; echo \"\$LOCKFILE\" > $T/$1.file; date +%s > $T/$1.t0; sleep $2; give; date +%s > $T/$1.t1" &
}
# 1
holder h1 6; holder h2 6; holder h3 6
sleep 3
n=$(ls "$T"/locks/S.* 2>/dev/null | wc -l)
ck "three shared holders hold three lock files (found $n)" '[ "$n" = 3 ]'
ck "their lock file names differ" '[ "$(cat $T/h1.file $T/h2.file $T/h3.file | sort -u | wc -l)" = 3 ]'
ck "no literal S.* file" '[ ! -e "$T/locks/S.*" ]'
for h in h1 h2 h3; do
  ck "$h's file names its own pid" 'grep -q "^pid [0-9]" "$(cat $T/$h.file)"'
done
wait
ck "every shared file released" '[ "$(ls "$T"/locks/S.* 2>/dev/null | wc -l)" = 0 ]'
# 2
holder A 4; holder B 12
sleep 2
BM=$T sh -c ". $T/common.sh; take_excl 'test writer 计时运行'; date +%s > $T/W.t0; give" &
W=$!
sleep 5   # (A has ended by now, B holds until about t+12)
ck "A ended" '[ -f $T/A.t1 ]'
ck "the writer still waits while B holds" '[ ! -f $T/W.t0 ]'
wait $W
ck "the writer took the lock only after B ended" '[ "$(cat $T/W.t0)" -ge "$(cat $T/B.t1)" ]'
wait
# 4 (bounded wait) writer 1 holds; a shared taker starts waiting; writer 2 arrives later: X1, the shared job, X2
writer() {   # writer <name> <seconds>
  BM=$T sh -c ". $T/common.sh; take_excl 'test $1 计时运行'; date +%s > $T/$1.t0; sleep $2; give; date +%s > $T/$1.t1" &
}
writer E1 6
sleep 1
holder S4 3
sleep 2
writer E2 1
wait
ck "the shared job ran after writer 1 ended" '[ "$(cat $T/S4.t0)" -ge "$(cat $T/E1.t1)" ]'
ck "writer 2 took the lock only after the shared job ended" '[ "$(cat $T/E2.t0)" -ge "$(cat $T/S4.t1)" ]'
ck "no waiting marker left" '[ "$(ls "$T"/locks/W.* 2>/dev/null | wc -l)" = 0 ]'
# 5 (writer first is kept) a shared taker that arrives after a writer started waiting does not go before it
holder S5a 6
sleep 1
writer E3 1
sleep 2
holder S5b 1
wait
ck "the late shared job ran after the writer (writer first)" '[ "$(cat $T/S5b.t0)" -ge "$(cat $T/E3.t1)" ]'
ck "the writer waited for the early shared job" '[ "$(cat $T/E3.t0)" -ge "$(cat $T/S5a.t1)" ]'
# 6 (a dead holder whose container still runs keeps the lock)
sleep 12 &
STUB=$!
BM=$T sh -c ". $T/common.sh; take_excl 'test dead holder 计时运行'; lock_container stub-$STUB; exec sleep 600" &
H=$!
sleep 3
ck "the X file names its container" 'grep -q "^container stub-$STUB" $T/locks/X'
kill -9 $H 2>/dev/null
holder S6 1
sleep 5
ck "a shared taker waits while the killed holder's container runs" '[ ! -f $T/S6.t0 ]'
wait $STUB
date +%s > $T/stub.t1
wait
ck "the lock released after the container ended" '[ -f $T/S6.t0 ] && [ "$(cat $T/S6.t0)" -ge "$(cat $T/stub.t1)" ]'
ck "no lock file left" '[ "$(ls $T/locks/X $T/locks/S.* $T/locks/.orphan.* 2>/dev/null | wc -l)" = 0 ]'
# 3
r=$(BM=$T sh -c ". $T/common.sh; f=keep_f; n=keep_n; p=keep_p; t0=keep_t0; a=keep_a; take_shared 'test vars'; give; take_excl 'test vars 计时运行'; give; echo \$f \$n \$p \$t0 \$a")
ck "a caller's f n p t0 a survive (got: $r)" '[ "$r" = "keep_f keep_n keep_p keep_t0 keep_a" ]'
[ $fail = 0 ] && echo "test_lock: PASS" || echo "test_lock: FAIL"
exit $fail
