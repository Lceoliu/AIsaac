#!/bin/bash
# kit_doctor.sh <version> [--no-smoke]: read-only health check of this host for a kit version. One line per item:
#   PASS|WARN|FAIL <item>: <detail>
# rc 1 when any FAIL. Nothing on the host is changed; the only writes are the kit's own log, the shared bm1 lock file
# while the smoke runs, and one --rm container aisaac-bench-kit-doctor.
# The smoke (last item): in the kit image, NO mount, on the first free GPU of KIT_SMOKE_GPUS: kit_pkgs.py (imports)
# and bm1/cuda_smoke.py (CUDA init, matmul vs CPU, cuDNN conv vs CPU, CUDA graph capture + replay).
set -u
KITDIR=$(cd "$(dirname "$0")" && pwd)
. "$KITDIR/kit_common.sh"
V=${1:?usage: kit_doctor.sh <version> [--no-smoke]}; SMOKE=1; [ "${2:-}" = --no-smoke ] && SMOKE=0
M=$(kit_manifest "$V")
NFAIL=0; NWARN=0
r() { echo "$1 $2: $3"; echo "$(date -u '+%F %T') [$KIT_HOST] doctor $V: $1 $2: $3" >> "$KIT_LOG"
      case $1 in FAIL) NFAIL=$((NFAIL+1)) ;; WARN) NWARN=$((NWARN+1)) ;; esac; }
vge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }   # vge a b: a >= b
echo "== kit_doctor $V on $KIT_HOST ($(hostname)), $(date -u '+%F %T') UTC"

# host and docker
if docker info > /dev/null 2>&1; then r PASS docker "server $(docker version --format '{{.Server.Version}}'), storage $(docker info --format '{{.Driver}}')"
else r FAIL docker "docker info fails (daemon down or no permission)"; fi
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then r PASS nvidia-runtime "registered"
else r FAIL nvidia-runtime "no nvidia runtime in docker info"; fi
CG=$(docker info --format '{{.CgroupVersion}}' 2>/dev/null); r PASS cgroup "v${CG:-?} (docker), $(stat -fc %T /sys/fs/cgroup)"

# driver and GPUs
[ -f "$M" ] && MIN=$(kit_manifest_get "$V" image.min_driver) && REC=$(kit_manifest_get "$V" image.recommended_driver) || { MIN=525.60.13; REC=570.26; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$DRV" ]; then r FAIL driver "nvidia-smi gives no driver version"
elif ! vge "$DRV" "$MIN"; then r FAIL driver "$DRV < $MIN (CUDA 12 minor-version compatibility minimum)"
elif ! vge "$DRV" "$REC"; then r WARN driver "$DRV >= $MIN but < $REC: cu128 runs by minor-version compatibility (no PTX JIT of newer PTX)"
else r PASS driver "$DRV (>= $REC)"; fi
L=$(nvidia-smi -L 2>&1); NG=$(echo "$L" | grep -c '^GPU ' || true); NE=$(echo "$L" | grep -ciE 'error|unable|unknown' || true)
if [ "$NE" -gt 0 ]; then r FAIL gpus "nvidia-smi -L reports errors: $(echo "$L" | grep -iE 'error|unable|unknown' | head -2 | tr '\n' ' ')"
elif [ -n "${KIT_GPU_EXPECT:-}" ] && [ "$NG" -lt "$KIT_GPU_EXPECT" ]; then r FAIL gpus "$NG listed, $KIT_GPU_EXPECT expected"
else r PASS gpus "$NG listed (expected ${KIT_GPU_EXPECT:-?}): $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '{printf "%s:%sMiB ", $1, $2}')"; fi
KL=$(journalctl -k -b --no-pager 2>/dev/null); echo "$KL" | grep -q 'Linux version' || KL=$(dmesg 2>/dev/null)
if ! echo "$KL" | grep -q 'Linux version'; then r WARN xid "kernel log of this boot not readable by $(id -un)"
else NX=$(echo "$KL" | grep -c 'NVRM: Xid' || true)
  if [ "$NX" -gt 0 ]; then r WARN xid "$NX NVRM Xid line(s) since boot, last: $(echo "$KL" | grep 'NVRM: Xid' | tail -1 | cut -c1-160)"
  else r PASS xid "no NVRM Xid line since boot ($(uptime -s 2>/dev/null))"; fi; fi

# resources and tools
FREE=$(df -BG --output=avail "$KIT_ROOT" | tail -1 | tr -dc 0-9)
if [ "$FREE" -lt 25 ]; then r FAIL disk "$FREE GB free under $KIT_ROOT"; elif [ "$FREE" -lt 60 ]; then r WARN disk "$FREE GB free under $KIT_ROOT"
else r PASS disk "$FREE GB free under $KIT_ROOT"; fi
AV=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo); TOT=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
if [ "$AV" -lt 32 ]; then r WARN memory "$AV of $TOT GiB available"; else r PASS memory "$AV of $TOT GiB available"; fi
miss=; for t in docker tar gzip sha256sum python3 taskset setsid nvidia-smi awk; do command -v $t > /dev/null || miss="$miss $t"; done
[ -z "$miss" ] && r PASS tools-required "all present" || r FAIL tools-required "missing:$miss"
miss=; for t in pigz zstd gio numactl ionice; do command -v $t > /dev/null || miss="$miss $t"; done
[ -z "$miss" ] && r PASS tools-optional "all present" || r WARN tools-optional "missing:$miss"
for LD in ${KIT_LOCKDIRS:-${KIT_LOCKDIR:-}}; do
  h=$(for f in "$LD"/X "$LD"/S.*; do [ -f "$f" ] && echo "$(basename "$f")=$(head -1 "$f")"; done | tr '\n' ';')
  r PASS lock "$LD holders now: ${h:-none}"
done

# kit version: manifest, archives, image, data
if [ ! -f "$M" ]; then r FAIL manifest "$M missing"; else
  r PASS manifest "$M"
  for a in image.archive data; do
    f=$KIT_ROOT/kit/$(kit_manifest_get "$V" $a.file); want=$(kit_manifest_get "$V" $a.bytes)
    if [ ! -f "$f" ]; then r WARN archive "$(basename "$f") not in $KIT_ROOT/kit (redundancy: keep it on two hosts)"
    elif [ "$(stat -c %s "$f")" != "$want" ]; then r FAIL archive "$(basename "$f") is $(stat -c %s "$f") bytes, manifest $want"
    elif [ -f "$f.sha256" ] && [ "$(cut -d' ' -f1 "$f.sha256")" != "$(kit_manifest_get "$V" $a.sha256)" ]; then r FAIL archive "$(basename "$f").sha256 differs from the manifest"
    else r PASS archive "$(basename "$f") $want bytes (= manifest; sha256 sidecar $( [ -f "$f.sha256" ] && echo matches || echo absent))"; fi
  done
  TAG=$(kit_manifest_get "$V" image.tag); ID=$(kit_manifest_get "$V" image.id)
  got=$(docker image inspect -f '{{.Id}}' "$TAG" 2>/dev/null || echo none)
  lab=$(docker image inspect -f '{{index .Config.Labels "aisaac.kit.version"}}' "$TAG" 2>/dev/null || true)
  if [ "$got" = "$ID" ] && [ "$lab" = "$V" ]; then r PASS image "$TAG $ID (label $lab)"; else r FAIL image "$TAG: ID $got label ${lab:-none}, manifest $ID"; fi
  D=$KIT_ROOT/data/$V; NF=$(kit_manifest_get "$V" data.files); NB=$(kit_manifest_get "$V" data.content_bytes)
  if [ ! -d "$D" ]; then r WARN data "$D not installed (bundle only)"
  else
    n=$(cd "$D" && find runtime tools payloads checkpoints -type f | wc -l)
    b=$(cd "$D" && find runtime tools payloads checkpoints -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
    w=$(find "$D" -perm /222 | wc -l)
    if [ "$n" != "$NF" ] || [ "$b" != "$NB" ]; then r FAIL data "$n files / $b bytes, manifest $NF / $NB"
    elif ! (cd "$D" && sha256sum --quiet -c FILES.sha256 > /dev/null 2>&1); then r FAIL data "sha256sum -c FILES.sha256 fails"
    elif [ "$w" -gt 0 ]; then r WARN data "$n files / $b bytes ok, but $w entries writable"
    else r PASS data "$D: $n files / $b bytes = manifest, every sha256 matches, read-only"; fi
  fi
fi

# CUDA smoke in the kit image
if [ "$SMOKE" = 1 ] && [ -f "$M" ]; then
  G=
  for g in ${KIT_SMOKE_GPUS:-}; do
    m=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    p=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [ -n "$m" ] && [ "$m" -lt 500 ] && [ "$p" = 0 ]; then G=$g; break; fi
  done
  if [ -z "$G" ]; then r WARN smoke "no free GPU among ${KIT_SMOKE_GPUS:-none}: GPU part of doctor skipped"
  else
    kit_lock "doctor $V smoke (GPU $G, not timed)"
    m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    p=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [ "$m" -ge 500 ] || [ "$p" != 0 ]; then kit_unlock; r WARN smoke "GPU $G became busy: smoke not run"
    else
      RUNC="docker run --rm -i --name aisaac-bench-kit-doctor --runtime=nvidia --network none --read-only --tmpfs /tmp:rw,size=1g
            -m $KIT_MEM --cpuset-cpus $KIT_CPUSET --user $(id -u):$(id -g) -e HOME=/tmp -e NVIDIA_VISIBLE_DEVICES=$G
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility --entrypoint /opt/aisaac/venv/bin/python $TAG"
      o1=$($RUNC - < "$KITDIR/kit_pkgs.py" 2>&1); c1=$?
      o2=$($RUNC - < "$KITDIR/../bm1/cuda_smoke.py" 2>&1); c2=$?
      kit_unlock
      imp=$(echo "$o1" | grep '^IMPORT' | awk '{printf "%s:%s ", $2, $3}')
      [ $c1 = 0 ] && r PASS imports "no mount: $imp$(echo "$o1" | grep '^TORCH')" || r FAIL imports "rc $c1: $imp $(echo "$o1" | tail -2)"
      [ $c2 = 0 ] && r PASS smoke "GPU $G: $(echo "$o2" | tail -1)" || r FAIL smoke "GPU $G rc $c2: $(echo "$o2" | tail -2)"
    fi
  fi
fi
echo "== $NFAIL FAIL, $NWARN WARN"
[ $NFAIL = 0 ]
