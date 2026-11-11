#!/bin/bash
# kit_build.sh <version> <ops commit>: on the build host (base image + venv + data sources, hosts/<hostname>.env).
# Offline. Steps (each heavy one under the shared bm1 lock, on KIT_CPUSET; stops at the first failure):
#   1 image   docker build of aisaac-bench:kit-<version> FROM the base image; context = kit_ctx.py's stream
#   2 verify  kit_pkgs.py + pip freeze + pip check in the OLD setup (base image, /bench mount, PYTHONPATH=bm1/pydeps)
#             and in the NEW image with no mount; the package lines and the freeze must be identical
#   3 data    aisaac-data-<version>.tar: runtime/, tools/, payloads/b45-payloads2/, checkpoints/c99-segment2-final.pt,
#             FILES.sha256 (per-file sha256 list) under aisaac-data-<version>/
#   4 save    docker save | pigz > aisaac-bench-<version>.tar.gz (once; later migrations copy this file)
#   5 manifest kit-<version>.manifest.json (the MANIFEST.json entry), sha256 sidecars
# Everything goes to <ROOT>/kit/; a version that exists already is refused (versions are never rebuilt).
set -eu
KITDIR=$(cd "$(dirname "$0")" && pwd)
. "$KITDIR/kit_common.sh"
V=${1:?usage: kit_build.sh <version> <ops commit>}; OPS=${2:?ops commit}
case $V in kit-[0-9]*.[0-9]*) ;; *) kdie "version must look like kit-YYYYMMDD.N" ;; esac
TAG=aisaac-bench:$V
K=$KIT_ROOT/kit
B=$K/build-$V
IMG=$K/aisaac-bench-$V.tar.gz
DAT=$K/aisaac-data-$V.tar
M=$(kit_manifest "$V")
for f in "$IMG" "$DAT" "$M"; do [ -e "$f" ] && kdie "$f exists: versions are never rebuilt"; done
mkdir -p "$B"
trap 'kit_unlock' EXIT
klog "build $V (ops $OPS) starts; our containers now: $(docker ps --format '{{.Names}}' | grep '^aisaac-bench-' | tr '\n' ' ')"

# 1 image
if docker image inspect "$TAG" > /dev/null 2>&1; then
  klog "image $TAG exists already: not rebuilt ($(docker image inspect -f '{{.Id}}' "$TAG"))"
else
  kit_lock "build $V: docker build"
  t0=$(date +%s)
  pinned python3 "$KITDIR/kit_ctx.py" "$KITDIR/Dockerfile" "$KIT_SRC_VENV" "$KIT_SRC_PYDEPS" $KIT_OLD_VENV_PATHS \
      2> "$B/ctx-fixed.txt" \
    | DOCKER_BUILDKIT=1 docker build --progress=plain --network=none -t "$TAG" \
        --build-arg KIT_BASE="$KIT_BASE_IMAGE" --build-arg KIT_VERSION="$V" --build-arg KIT_OPS_COMMIT="$OPS" - \
        > "$B/docker-build.log" 2>&1 || { tail -30 "$B/docker-build.log"; kdie "docker build"; }
  kit_unlock
  klog "image built in $(( $(date +%s) - t0 )) s: $(docker image inspect -f '{{.Id}}' "$TAG"); repaired files: $(grep -c fixed "$B/ctx-fixed.txt")"
fi
ID=$(docker image inspect -f '{{.Id}}' "$TAG")
SIZE=$(docker image inspect -f '{{.Size}}' "$TAG")

# 2 verify (CPU only, no GPU, no network)
kit_lock "build $V: verify"
RUNC="docker run --rm -i --network none --read-only --tmpfs /tmp:rw,size=1g -m $KIT_MEM --cpuset-cpus $KIT_CPUSET --user $(id -u):$(id -g) -e HOME=/tmp"
OLDPY=/bench/home/isaac-abplus/.venv/bin/python
OLDRUN="$RUNC --name aisaac-bench-kit-verify-old -v $KIT_ROOT:/bench:ro -e PYTHONPATH=/bench/bm1/pydeps --entrypoint $OLDPY $KIT_BASE_IMAGE"
NEWRUN="$RUNC --name aisaac-bench-kit-verify-new --entrypoint /opt/aisaac/venv/bin/python $TAG"
$OLDRUN - < "$KITDIR/kit_pkgs.py" > "$B/pkgs-old.txt" 2>&1 || kdie "kit_pkgs in the old setup: $(tail -3 $B/pkgs-old.txt)"
$NEWRUN - < "$KITDIR/kit_pkgs.py" > "$B/pkgs-new.txt" 2>&1 || kdie "kit_pkgs in the new image: $(tail -3 $B/pkgs-new.txt)"
$OLDRUN -m pip freeze --all < /dev/null > "$B/freeze-old.txt" 2>&1 || true
$NEWRUN -m pip freeze --all < /dev/null > "$B/freeze-new.txt" 2>&1 || true
$OLDRUN -m pip check < /dev/null > "$B/pipcheck-old.txt" 2>&1 || true
$NEWRUN -m pip check < /dev/null > "$B/pipcheck-new.txt" 2>&1 || true
# the new image as a plain shell sees it: PATH / PYTHONPATH from the image, no explicit interpreter
$RUNC --name aisaac-bench-kit-verify-sh $TAG -c 'command -v python pytest py-spy; python -c "import sys,torch,numpy; print(sys.executable, torch.__version__, numpy.__version__)"; pytest --version; py-spy --version' \
  < /dev/null > "$B/shell-new.txt" 2>&1 || kdie "plain shell in the new image: $(tail -3 $B/shell-new.txt)"
# apt package versions of the base (Dockerfile.from-scratch is written from these)
$RUNC --name aisaac-bench-kit-verify-dpkg --entrypoint /usr/bin/dpkg-query $KIT_BASE_IMAGE -W \
  < /dev/null > "$B/dpkg-base.txt" 2>&1 || true
kit_unlock
d1=$(diff <(grep -E '^(PY|PKG|SHADOWED|IMPORT|TORCH) ' "$B/pkgs-old.txt") <(grep -E '^(PY|PKG|SHADOWED|IMPORT|TORCH) ' "$B/pkgs-new.txt") || true)
d2=$(diff "$B/freeze-old.txt" "$B/freeze-new.txt" || true)
[ -z "$d1" ] || { echo "$d1" | head -20; kdie "package set differs between old setup and new image"; }
[ -z "$d2" ] || { echo "$d2" | head -20; kdie "pip freeze differs between old setup and new image"; }
grep -q '^TORCH 2.7.1+cu128 ' "$B/pkgs-new.txt" || kdie "torch in the new image is not 2.7.1+cu128"
klog "verify: $(grep -c '^PKG ' $B/pkgs-new.txt) packages identical (old setup vs image), freeze identical ($(wc -l < $B/freeze-new.txt) lines), imports: $(grep '^IMPORT' $B/pkgs-new.txt | awk '{printf "%s:%s ", $2, $3}'); pip check old: $(head -1 $B/pipcheck-old.txt); new: $(head -1 $B/pipcheck-new.txt)"

# 3 data bundle
kit_lock "build $V: data bundle"
S=$B/data-stage; mkdir -p "$S/checkpoints" "$S/payloads"
# the file list first: paths as they are in the bundle (under aisaac-data-<version>/)
(
  cd "$S"
  ln -sfn "$KIT_SRC_RUNTIME" runtime; ln -sfn "$KIT_SRC_TOOLS" tools
  ln -sfn "$KIT_SRC_PAYLOADS" payloads/b45-payloads2; ln -sfn "$KIT_SRC_CKPT" checkpoints/c99-segment2-final.pt
  find -L runtime tools payloads checkpoints -type f | LC_ALL=C sort | pinned xargs -d '\n' sha256sum > "$B/FILES.sha256"
)
NF=$(wc -l < "$B/FILES.sha256")
NB=$(cd "$S" && cut -c67- "$B/FILES.sha256" | tr '\n' '\0' | xargs -0 stat -L -c %s | awk '{s+=$1} END {print s}')
cp "$B/FILES.sha256" "$S/FILES.sha256"
TOP=aisaac-data-$V
pinned tar --sort=name --numeric-owner --owner=0 --group=0 --format=gnu -cf "$DAT.part" \
  --transform "s,^,$TOP/," -C "$S" FILES.sha256 \
  -C "$(dirname "$KIT_SRC_RUNTIME")" "$(basename "$KIT_SRC_RUNTIME")" \
  -C "$(dirname "$KIT_SRC_TOOLS")" "$(basename "$KIT_SRC_TOOLS")" \
  -C "$(dirname "$(dirname "$KIT_SRC_PAYLOADS")")" "$(basename "$(dirname "$KIT_SRC_PAYLOADS")")/$(basename "$KIT_SRC_PAYLOADS")" \
  -C "$(dirname "$(dirname "$KIT_SRC_CKPT")")" "$(basename "$(dirname "$KIT_SRC_CKPT")")/$(basename "$KIT_SRC_CKPT")"
NT=$(tar -tf "$DAT.part" | grep -v '/$' | wc -l)
[ "$NT" = $((NF + 1)) ] || kdie "data bundle holds $NT files, the list $NF (+ FILES.sha256)"
mv "$DAT.part" "$DAT"
DSHA=$(pinned sha256sum "$DAT" | cut -d' ' -f1); DBYTES=$(stat -c %s "$DAT")
FSHA=$(sha256sum "$B/FILES.sha256" | cut -d' ' -f1)
echo "$DSHA  $(basename "$DAT")" > "$DAT.sha256"
cp "$B/FILES.sha256" "$K/aisaac-data-$V.files.sha256"
kit_unlock
klog "data bundle: $NF files, $NB content bytes, tar $DBYTES bytes, sha256 $DSHA"

# 4 image archive
kit_lock "build $V: docker save"
t0=$(date +%s)
PZ="gzip -c"; command -v pigz > /dev/null && PZ="pigz -p 12 -c"
set -o pipefail
docker save "$TAG" | pinned $PZ > "$IMG.part" || kdie "docker save"
set +o pipefail
mv "$IMG.part" "$IMG"
ISHA=$(pinned sha256sum "$IMG" | cut -d' ' -f1); IBYTES=$(stat -c %s "$IMG")
echo "$ISHA  $(basename "$IMG")" > "$IMG.sha256"
kit_unlock
klog "image archive: $IBYTES bytes in $(( $(date +%s) - t0 )) s, sha256 $ISHA"

# 5 manifest entry
python3 - "$M" <<EOF
import json, sys
e = {
  "version": "$V", "built_on": "$KIT_HOST", "built_at_utc": "$(date -u '+%F %T')", "ops_commit": "$OPS",
  "image": {"tag": "$TAG", "id": "$ID", "size_bytes": $SIZE, "base": "$KIT_BASE_IMAGE",
            "archive": {"file": "$(basename "$IMG")", "bytes": $IBYTES, "sha256": "$ISHA", "compression": "gzip"},
            "python": "3.10.12", "torch": "2.7.1+cu128", "cuda_runtime": "12.8",
            "min_driver": "525.60.13", "recommended_driver": "570.26",
            "venv": "/opt/aisaac/venv", "pydeps": "/opt/aisaac/pydeps", "packages": $(grep -c '^PKG ' $B/pkgs-new.txt)},
  "data": {"file": "$(basename "$DAT")", "bytes": $DBYTES, "sha256": "$DSHA", "top_dir": "$TOP",
           "files": $NF, "content_bytes": $NB, "file_list": "aisaac-data-$V.files.sha256", "file_list_sha256": "$FSHA"},
}
json.dump(e, open(sys.argv[1], "w"), indent=2)
print(json.dumps(e, indent=2))
EOF
cp "$B/pkgs-new.txt" "$K/$V.packages.txt"
klog "build $V done: $M"
