#!/bin/sh
# bm1: test of deploy.sh ops in a scratch directory (nothing else touched; 2026-11-18):
#   1. a first deploy writes every file of the tree's bridge/ops/bm1 with LF line ends and records it;
#   2. a host edit (the host copy differs from the one last deployed) stops the deploy: exit 3, the file listed,
#      nothing written -- also for a file the new tree does not change;
#   3. a line-end-only difference is no host edit;
#   4. OPS_ACCEPT=<file> lets the named file be overwritten;
#   5. a tree whose run.sh or train_guard.sh does not call gpu_forbid is refused: exit 4, nothing written.
#   usage: test_deploy_ops.sh <scratch dir (created, left for inspection)>        exit 0: pass
set -u
BMSRC=$(cd "$(dirname "$0")" && pwd)
T=$1
mkdir -p "$T"
T=$(cd "$T" && pwd)
cp "$BMSRC/common.sh" "$BMSRC/deploy.sh" "$T/"
printf 'BM1_SESSION=deploytest\nBM1_IN=/bench/bm1\nBM1_MODE=direct\n' > "$T/bm1.env"
fail=0
ck() { if eval "$2"; then echo "ok   $1"; else echo "FAIL $1"; fail=1; fi; }
mktree() {   # mktree <name> <run.sh guard line> <a.py text>: a tree with CRLF line ends, as git archive on Windows
  d=$T/trees/$1/bridge/ops/bm1; mkdir -p $d/sub
  printf '#!/bin/sh\r\n%s\r\necho run\r\n' "$2" > $d/run.sh
  printf '#!/bin/sh\r\ngpu_forbid "$G" "train"\r\n' > $d/train_guard.sh
  printf '%s\r\n' "$3" > $d/a.py
  printf 'x = 1\r\n' > $d/sub/b.py
}
dep() { BM=$T OPS_ACCEPT=${OPS_ACCEPT:-} sh $T/deploy.sh ops "$@" > $T/out.txt 2>&1; echo $?; }
G='gpu_forbid "$G" "$P"'
mktree t1 "$G" 'v = 1'
ck "first deploy rc 0" '[ "$(dep t1)" = 0 ]'
ck "files written" '[ -f $T/run.sh ] && [ -f $T/sub/b.py ] && [ -f $T/.ops_deployed/a.py ]'
ck "LF line ends" '[ -f $T/a.py ] && ! grep -q "$(printf "\r")" $T/run.sh $T/a.py $T/sub/b.py'
mktree t2 "$G" 'v = 2'
echo 'local = 1' >> $T/sub/b.py   # a host edit of a file the new tree leaves as it was
ck "a host edit stops the deploy (rc 3)" '[ "$(dep t2)" = 3 ]'
ck "the edited file listed" 'grep -q "sub/b.py" $T/out.txt'
ck "nothing written" 'grep -q "v = 1" $T/a.py'
printf 'x = 1\r\n' > $T/sub/b.py   # the edit undone, CRLF this time: a line-end difference only
ck "a line-end difference is no host edit (rc 0)" '[ "$(dep t2)" = 0 ]'
ck "the new version written" 'grep -q "v = 2" $T/a.py'
mktree t3 "$G" 'v = 3'
echo '# host note' >> $T/a.py
ck "OPS_ACCEPT=a.py overwrites it (rc 0)" '[ "$(OPS_ACCEPT=a.py dep t3)" = 0 ] && grep -q "v = 3" $T/a.py'
mktree t4 'echo no guard' 'v = 4'
ck "a run.sh without gpu_forbid is refused (rc 4)" '[ "$(dep t4)" = 4 ]'
ck "nothing written by the refused deploy" 'grep -q "v = 3" $T/a.py && grep -q "gpu_forbid" $T/run.sh'
[ $fail = 0 ] && echo "test_deploy_ops: PASS" || echo "test_deploy_ops: FAIL"
exit $fail
