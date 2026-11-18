#!/bin/sh
# Run real evaluation wrappers with a fake job and assert publication, rerun and counters.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
T=$(cd "$1" && pwd)
mkdir -p "$T/bin" "$T/bm/evtools" "$T/bm/reports"
cp "$HERE/ev_ab2.sh" "$HERE/ev_c113.sh" "$HERE/ab2_after.sh" "$HERE/c110_after.sh" "$T/bm/"
cp "$HERE/evtools/ev_complete.py" "$HERE/evtools/eval_status.sh" "$T/bm/evtools/"
export TEST_PYTHON=$(command -v python || command -v python3)
export BM="$T/bm" BM1_TRAIN="$T/train" PATH="$T/bin:$PATH"
cat > "$T/bin/python3" <<'EOF'
#!/bin/sh
exec "$TEST_PYTHON" -B "$@"
EOF
for n in sleep pgrep; do printf '#!/bin/sh\nexit 1\n' > "$T/bin/$n"; done
chmod +x "$T/bin/"*
printf 'BM1_TRAIN_IN=/bench/train\nBM1_IN=/bench/bm1\n' > "$BM/bm1.env"
cat > "$BM/job.sh" <<'EOF'
#!/bin/sh
mkdir -p "$WHERE"
printf '{"seeds":"100:2","groups":{"run":{"episodes":%s}},"empty":0}\n' "$PLAYED" > "$WHERE/summary.json"
L=$(mktemp --suffix=.log "$BM/reports/$1-XXXXXX")
printf 'test job\n' > "$L"
case "${JUNK_RC:-}" in
  missing) ;;
  wrong) printf '0\n' > "$L.rc" ;;
  *) printf '%s\n' "$JOB_RC" > "$L.rc" ;;
esac
printf '%s\n' "$L"
exit "$JOB_RC"
EOF
mkdir -p "$BM1_TRAIN/c/run/run-s0/checkpoints" "$BM1_TRAIN/ab2/run/run-s0/checkpoints"
: > "$BM1_TRAIN/c/run/run-s0/checkpoints/last.pt"
: > "$BM1_TRAIN/ab2/run/run-s0/checkpoints/last.pt"
ev() {
  want=$1; shift
  rc=0; sh "$@" > "$T/out" 2>&1 || rc=$?
  [ "$want" = "$rc" ] || { cat "$T/out"; echo "expected $want, got $rc"; exit 1; }
}
run_case() {
  [ "$1" = ab2 ] && {
    shift; r=$1; tag=$2; shift 2
    WHERE="$BM1_TRAIN/ab2/eval/$tag" ev "$r" "$BM/ev_ab2.sh" run new 0 0 0 1 "$tag,last.pt,100:2"
    return
  }
  shift; r=$1; tag=$2
  WHERE="$BM1_TRAIN/c/eval/$tag" ev "$r" "$BM/ev_c113.sh" c/run 0 0 0 1 "$tag,last.pt,100:2"
}
for wrapper in ab2 c113; do
  JOB_RC=0 PLAYED=2 run_case "$wrapper" 0 valid
  JOB_RC=0 PLAYED=2 run_case "$wrapper" 0 valid
  grep -q '(done)' "$T/out"
  JOB_RC=0 PLAYED=1 run_case "$wrapper" 1 short
  JOB_RC=0 PLAYED=2 run_case "$wrapper" 1 short
  ! grep -q '(done)' "$T/out"
  JOB_RC=4 PLAYED=2 run_case "$wrapper" 1 failed
  JOB_RC=0 PLAYED=2 run_case "$wrapper" 1 failed
  JOB_RC=0 PLAYED=2 JUNK_RC=missing run_case "$wrapper" 1 no_rc
  JOB_RC=0 PLAYED=2 run_case "$wrapper" 1 no_rc
  JOB_RC=4 PLAYED=2 JUNK_RC=wrong run_case "$wrapper" 1 wrong_rc
done
for n in a b c d; do
  d="$BM1_TRAIN/ab2/eval/unrelated-$n"; mkdir -p "$d"
  printf '{"seeds":"2147490000:128","groups":{"run":{"episodes":128}},"empty":0}\n' > "$d/summary.json"
  printf '0\n' > "$d.log.rc"
done
mkdir -p "$BM1_TRAIN/ab2/ab2-c99old" "$BM1_TRAIN/ab2/ab2-bm"
touch "$BM1_TRAIN/ab2/ab2-c99old/guard.log" "$BM1_TRAIN/ab2/ab2-bm/guard.log"
ev 1 "$BM/ab2_after.sh"
grep -q '0/4 complete' "$BM1_TRAIN/ab2/AFTER_DONE"
mkdir -p "$BM1_TRAIN/c110/arm" "$BM1_TRAIN/c110/eval/arm-c128-last"
: > "$BM1_TRAIN/c110/arm/guard.log"
cp "$BM1_TRAIN/ab2/eval/unrelated-a/summary.json" "$BM1_TRAIN/c110/eval/arm-c128-last/summary.json"
cp "$BM1_TRAIN/ab2/eval/unrelated-a.log.rc" "$BM1_TRAIN/c110/eval/arm-c128-last.log.rc"
ev 1 "$BM/c110_after.sh" arm 0 0 0
grep -q '0/1 complete' "$BM1_TRAIN/c110/AFTER_DONE_arm"
echo 'test_eval_ops: PASS'
