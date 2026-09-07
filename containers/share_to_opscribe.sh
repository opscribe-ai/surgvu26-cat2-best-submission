#!/bin/bash
set -u
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin
echo "host: $(hostname)  start: $(date)"

SRC=/staging/n/nkalthoff/surgvu26
DST=/staging/groups/bhaskar_opscribe/surgvu26_cat2_share

[ -d "$SRC" ] || { echo "FATAL: $SRC not visible (WantStagingMount?)"; exit 1; }
[ -d "$(dirname "$DST")" ] || { echo "FATAL: shared staging not visible"; exit 1; }

mkdir -p "$DST"/{containers,models,artifacts} || exit 1

# models/ is rebuilt from scratch: job 9713114 flattened six model directories
# into it, and a stale loose adapter_config.json sitting beside real model
# directories is exactly the kind of thing someone loads by accident.
# The originals live in $SRC; this is only ever a copy.
if [ -n "$(ls -A "$DST/models" 2>/dev/null)" ]; then
  echo "clearing $DST/models (previous run flattened it)"
  rm -rf "${DST:?}/models"/* || exit 1
fi

copy () {  # copy <src> <dstdir>
  # STRIP THE TRAILING SLASH. To rsync, "dir/" means "the CONTENTS of dir",
  # so passing the glob expansion of */ (which keeps the slash) copied every
  # model's files into one directory on top of each other -- job 9713114
  # produced a 14GB merge of six models rather than six model directories.
  local s="${1%/}" d="$2"
  if [ ! -e "$s" ]; then echo "SKIP (missing): $s"; return 0; fi
  echo "copying $s -> $d"
  if command -v rsync >/dev/null; then
    rsync -a --no-o --no-g "$s" "$d/" || { echo "FAIL: $s"; return 1; }
  else
    cp -a "$s" "$d/" || { echo "FAIL: $s"; return 1; }
  fi
  # Verify a directory arrived AS a directory, not spilled into the parent.
  if [ -d "$s" ] && [ ! -d "$d/$(basename "$s")" ]; then
    echo "FAIL: $(basename "$s") did not land as a directory under $d"
    return 1
  fi
}

RC=0

# --- containers: needed to actually RUN any of this -------------------------
copy "$SRC/surgvu26-train.sif"          "$DST/containers" || RC=1
copy "$SRC/surgvu26-train-torch26.sif"  "$DST/containers" || RC=1

# --- models: every real artifact. Smoke runs are throwaway; v6_stage2 is
#     BEING WRITTEN by job 9713019 right now, so it is deliberately excluded
#     -- a half-written adapter looks valid and silently isn't.
for m in "$SRC"/models/*/; do
  name=$(basename "$m")
  case "$name" in
    *smoke*)   echo "SKIP (smoke run): $name"; continue ;;
    v6_stage2) echo "SKIP (in progress, job 9713019): $name"; continue ;;
  esac
  copy "$m" "$DST/models" || RC=1
done
copy "$SRC/detector_v2" "$DST/models" || RC=1

# --- artifacts: small but EXPENSIVE. evidence_cache.jsonl alone took 18h of
#     GPU time to build from cold; the manifests took ~6h of extraction.
for f in evidence_cache.jsonl evidence_cache_report.json \
         qa_frames_manifest_v2.jsonl qa_frames_report_v2.json \
         qa_frames_manifest.jsonl qa_frames_report.json \
         qa_pairs_v2.jsonl qa_pairs.jsonl \
         gi_stage1_manifest.jsonl gi_stage1_splits.json \
         gi_train.json gi_val.json \
         serving_thresholds.json serving_probs.npz; do
  copy "$SRC/$f" "$DST/artifacts" || RC=1
done

# Group access. ORDER AND GROUP BOTH MATTER, and getting this wrong is silent:
# chmod g+rwX sets bits for whatever group owns the file, so if the file is
# owned by the private group `nkalthoff` it grants nothing to anyone else. The
# first run left every adapter at -rw-rw---- nkalthoff:nkalthoff -- looked
# permissive, was unreadable to the group it was being shared with.
# chgrp FIRST, then the bits, then setgid so anything added later inherits
# the group instead of repeating this.
chgrp -R bhaskar_opscribe "$DST" 2>/dev/null
chmod -R g+rwX "$DST" 2>/dev/null
find "$DST" -type d -exec chmod g+s {} + 2>/dev/null

# Prove it rather than assume it: no file in the bundle may be unreadable to
# the group, and none may be owned by a group other than bhaskar_opscribe.
BAD=$(find "$DST" ! -group bhaskar_opscribe -o ! -perm -g+r | head -5)
if [ -n "$BAD" ]; then
  echo "FAIL: these are not group-accessible:"; echo "$BAD"; RC=1
else
  echo "group access OK: whole bundle is bhaskar_opscribe-readable"
fi

echo "=== result ==="
du -sh "$DST" 2>/dev/null
find "$DST" -maxdepth 2 -mindepth 1 | sort
echo "exit: $RC  end: $(date)"
exit $RC
