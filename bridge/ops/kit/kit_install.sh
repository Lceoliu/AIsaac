#!/bin/bash
# kit_install.sh <version>: on a target host, after kit_transfer.py put the version's files into <ROOT>/kit/.
# Side by side: never replaces another image tag, never touches <ROOT>/bm1 or <ROOT>/home. Stops at the first failure.
#   1 layout   <ROOT>/kit, <ROOT>/data
#   2 archives sha256 of the image archive and the data bundle = the manifest's (computed on this host)
#   3 image    docker load (skipped when the tag exists with the manifest's ID), then the image ID = the manifest's
#   4 data     unpack into <ROOT>/data/<version>/ (via a .partial dir), file count / bytes = the manifest's,
#              sha256sum -c of FILES.sha256, then everything read-only (a-w)
# Steps 2-4 hold the shared bm1 lock (never during a timed run).
set -eu
KITDIR=$(cd "$(dirname "$0")" && pwd)
. "$KITDIR/kit_common.sh"
V=${1:?usage: kit_install.sh <version>}
K=$KIT_ROOT/kit
M=$(kit_manifest "$V")
[ -f "$M" ] || kdie "no manifest $M (kit_transfer.py copies it)"
TAG=$(kit_manifest_get "$V" image.tag); ID=$(kit_manifest_get "$V" image.id)
IMG=$K/$(kit_manifest_get "$V" image.archive.file); ISHA=$(kit_manifest_get "$V" image.archive.sha256)
DAT=$K/$(kit_manifest_get "$V" data.file); DSHA=$(kit_manifest_get "$V" data.sha256)
TOP=$(kit_manifest_get "$V" data.top_dir); NF=$(kit_manifest_get "$V" data.files); NB=$(kit_manifest_get "$V" data.content_bytes)
D=$KIT_ROOT/data/$V
mkdir -p "$KIT_ROOT/data"
trap 'kit_unlock' EXIT
klog "install $V starts"
kit_lock "install $V"

# 2 archives
h=$(pinned sha256sum "$DAT" | cut -d' ' -f1); [ "$h" = "$DSHA" ] || kdie "data bundle sha256 $h, manifest $DSHA"
klog "data bundle sha256 ok ($h)"
if docker image inspect "$TAG" > /dev/null 2>&1 && [ "$(docker image inspect -f '{{.Id}}' "$TAG")" = "$ID" ]; then
  klog "image $TAG present with the manifest's ID: not loaded again"
else
  h=$(pinned sha256sum "$IMG" | cut -d' ' -f1); [ "$h" = "$ISHA" ] || kdie "image archive sha256 $h, manifest $ISHA"
  klog "image archive sha256 ok ($h)"
  # 3 image
  t0=$(date +%s)
  docker load -i "$IMG" 2>&1 | tail -3 | while read -r l; do klog "docker load: $l"; done
  klog "docker load took $(( $(date +%s) - t0 )) s"
fi
got=$(docker image inspect -f '{{.Id}}' "$TAG" 2>/dev/null || echo none)
[ "$got" = "$ID" ] || kdie "image $TAG has ID $got, manifest $ID"
klog "image ID ok: $TAG $ID"

# 4 data
if [ -d "$D" ]; then
  klog "data $D exists: verifying only"
else
  P=$KIT_ROOT/data/.partial-$V
  [ -e "$P" ] && kit_trash "$P"
  mkdir -p "$P"
  pinned tar -xf "$DAT" -C "$P" --no-same-owner
  [ -d "$P/$TOP" ] || kdie "bundle has no top dir $TOP"
  mv "$P/$TOP" "$D"; rmdir "$P"
fi
n=$(cd "$D" && find runtime tools payloads checkpoints -type f | wc -l)
b=$(cd "$D" && find runtime tools payloads checkpoints -type f -printf '%s\n' | awk '{s+=$1} END {print s+0}')
[ "$n" = "$NF" ] && [ "$b" = "$NB" ] || kdie "data: $n files / $b bytes, manifest $NF / $NB"
(cd "$D" && pinned sha256sum --quiet -c FILES.sha256) || kdie "data: sha256sum -c FILES.sha256 failed"
chmod -R a-w "$D"
klog "data ok: $D, $n files, $b bytes, every sha256 matches, read-only"
kit_unlock
klog "install $V done"
