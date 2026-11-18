#!/bin/sh
# All launch commands in the deploy gate are stubs. This directory is left for inspection.
set -eu
BMSRC=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
T=$(cd "$1" && pwd)
cp "$BMSRC/common.sh" "$BMSRC/deploy.sh" "$T/"
printf 'BM1_SESSION=deploytest\nBM1_IN=/bench/bm1\nBM1_MODE=direct\nBM1_PY=python3\nBM1_TRASH=$BM/archive\n' > "$T/bm1.env"
ck() { if eval "$2"; then echo "ok: $1"; else echo "FAIL: $1"; cat "$T/out.txt"; exit 1; fi; }
mktree() {
  d=$T/trees/$1/bridge/ops/bm1
  mkdir -p "$d/sub"
  cp "$BMSRC/common.sh" "$BMSRC/run.sh" "$BMSRC/train_guard.sh" "$d/"
  printf '%s\r\n' "$2" > "$d/a.py"
  printf '%s\r\n' "${3:-x = 1}" > "$d/sub/b.py"
  printf 'one\rtwo\r\nlast\r' > "$d/sub/cr.txt"
}
dep() { BM=$T OPS_ACCEPT=${OPS_ACCEPT:-} sh "$T/deploy.sh" ops "$@" > "$T/out.txt" 2>&1; }
dep_rc() { r=0; dep "$@" || r=$?; echo "$r"; }
mktree t1 'v = 1'
printf 'existing host file\n' > "$T/a.py"
ck 'first deploy with no baseline does not overwrite a host edit' '[ "$(dep_rc t1)" = 3 ] && grep -q "existing host file" "$T/a.py" && [ ! -e "$T/.ops_deployed" ]'
mv "$T/a.py" "$T/a.py.before-first"
ck 'first deploy works and records all files' '[ "$(dep_rc t1)" = 0 ] && cmp "$T/a.py" "$T/.ops_deployed/a.py" && [ -f "$T/run.sh" ]'
printf 'one\rtwo\nlast\r' > "$T/expected-cr"
ck 'only CRLF is removed' 'cmp "$T/sub/cr.txt" "$T/expected-cr"'
mktree t2 'v = 2'
echo 'local = 1' >> "$T/sub/b.py"
ck 'host edit blocks every live write' '[ "$(dep_rc t2)" = 3 ] && grep -q sub/b.py "$T/out.txt" && grep -q "v = 1" "$T/a.py"'
printf 'x = 1\r\n' > "$T/sub/b.py"
ck 'line-end-only difference permits update' '[ "$(dep_rc t2)" = 0 ] && grep -q "v = 2" "$T/a.py"'
mktree t3 'v = 3'
echo '# host note' >> "$T/a.py"
ck 'wildcard is not an acceptance list' '[ "$(OPS_ACCEPT="*" dep_rc t3)" = 3 ]'
ck 'file-specific acceptance keeps the old host file in trash' '[ "$(OPS_ACCEPT=a.py dep_rc t3)" = 0 ] && grep -q "v = 3" "$T/a.py" && grep -R -q "host note" "$T/archive"'
for mutation in absent dead early empty; do
  mktree "$mutation" 'v = 4'
  case "$mutation" in
    absent) sed -i '/^DEVICES=.*gpu_forbid/d' "$d/run.sh" ;;
    dead) sed -i '/^DEVICES=.*gpu_forbid/{s/.*/if false; then &; fi/;}' "$d/train_guard.sh" ;;
    early) sed -i '2i exit 5' "$d/train_guard.sh" ;;
    empty) printf '\ngpu_forbid() { :; }\n' >> "$d/common.sh" ;;
  esac
  ck "$mutation guard fails the two-sided acceptance probe" '[ "$(dep_rc "$mutation")" = 4 ] && grep -q "v = 3" "$T/a.py"'
done
mktree partial 'v = 4' 'x = 4'
mkdir -p "$T/bin"
REAL_MV=$(command -v mv); export REAL_MV
cat > "$T/bin/mv" <<'EOF'
#!/bin/sh
case "${FAIL_MOVE:-}:$2" in
  file:*/.ops-stage.*/new/sub/b.py|baseline:*/.ops-stage.*/baseline) exit 9 ;;
esac
exec "$REAL_MV" "$@"
EOF
chmod +x "$T/bin/mv"
for part in file baseline; do
  ck "$part installation failure rolls back live files and baseline" \
    '[ "$(PATH="$T/bin:$PATH" FAIL_MOVE=$part dep_rc partial)" = 9 ] && grep -q "rolled back" "$T/out.txt" &&
    grep -q "v = 3" "$T/a.py" && grep -q "x = 1" "$T/sub/b.py" &&
    cmp "$T/a.py" "$T/.ops_deployed/a.py" && cmp "$T/sub/b.py" "$T/.ops_deployed/sub/b.py"'
done
echo 'test_deploy_ops: PASS'
