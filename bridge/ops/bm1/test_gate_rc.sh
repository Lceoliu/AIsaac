#!/bin/sh
# gate.sh exits with its last run's rc (2026-11-18, the coordinator), the same rc as that run's bm1.log line; RUNS < 1
# is refused (2). A stub runner stands in for the container (no GPU, no docker): it prints three native_check EXT lines
# and a pytest tail, and exits with the next rc of FAKE_RCS (one per run).
#   usage: sh test_gate_rc.sh <scratch dir>
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$1"
T=$(cd "$1" && pwd)
cp "$HERE/common.sh" "$HERE/gate.sh" "$T/"
printf 'BM1_IN=$BM\nBM1_MODE=docker\nBM1_PY=python3\nBM1_IMAGE=fixture\nBM1_GAME_RUNTIME=.\nBM1_DOCKER_BASE=.\nBM1_GPUS_OK=1\n' > "$T/bm1.env"
cat > "$T/run.sh" <<'STUB'
#!/bin/sh
n=$(cat "$BM/.calls" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "$BM/.calls"
rc=$(echo "$FAKE_RCS" | cut -d' ' -f$n)
for e in 1 2 3; do echo "EXT e$e ${FAKE_EXT:-ok}"; done
echo "1 passed in 0.01s"
exit ${rc:-0}
STUB
chmod +x "$T/run.sh"
export BM=$T
case_() {   # <expected exit> <FAKE_RCS> [RUNS] [FAKE_EXT]
  rm -f "$T/.calls" "$T/reports/bm1.log"
  actual=0
  FAKE_RCS=$2 RUNS=${3:-1} FAKE_EXT=${4:-ok} sh "$T/gate.sh" tree > "$T/out" 2>&1 || actual=$?
  [ "$actual" = "$1" ] || { cat "$T/out"; echo "test_gate_rc: FAKE_RCS '$2' RUNS ${3:-1}: expected exit $1, got $actual"; exit 1; }
  if [ "$1" != 2 ] || [ "${3:-1}" != 0 ]; then
    last=$(grep 'gate tree run' "$T/reports/bm1.log" | tail -1 | grep -o 'rc=[0-9]*')
    [ "$last" = "rc=$1" ] || { echo "test_gate_rc: the last bm1.log line says $last, the exit $1"; exit 1; }
  fi
}
case_ 0 0
case_ 1 1
case_ 2 2
case_ 9 0 1 FAIL       # a native component failed: rc 9 although pytest passed
case_ 0 '1 0' 2        # the last run's rc
case_ 1 '0 1' 2
case_ 2 '' 0           # RUNS 0 refused, nothing run
[ ! -e "$T/.calls" ]
echo 'test_gate_rc: PASS'
