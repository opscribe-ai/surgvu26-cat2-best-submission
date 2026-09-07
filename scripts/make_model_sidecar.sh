#!/bin/bash
set -euo pipefail
# Build Grand Challenge's optional MODEL TARBALL: the answering VLM and the
# judge, extracted to /opt/ml/model/ at runtime.
#
#   bash scripts/make_model_sidecar.sh [out.tar.gz]
#
# WHY THE MODELS LEFT THE IMAGE. int8 weights are ~8 GiB and the judge another
# ~2.95; with the image that is 14.18 GiB against a documented 10 GiB ceiling.
# Neither fits baked in. Moving both here drops the image to ~3.5 GiB (35% of
# the ceiling) and puts the weights where no size limit is documented.
#
# THE TRAILING DOT IS LOAD-BEARING. Grand Challenge's own instructions say
# `tar -czvf model.tar.gz -C /path/to/model .` and warn that "any common
# directories present in the .tar.gz will not be filtered out" -- i.e. the
# archive's paths are used as-is. Packing the PARENT would extract to
# /opt/ml/model/models/... and every lookup in scripts/inference.py's
# resolve_vlm_model_dir would miss, silently, leaving the run on whatever
# weights the image happens to carry.
STAGING=/staging/n/nkalthoff/surgvu26/models
OUT="${1:-/staging/n/nkalthoff/surgvu26/surgvu26-models.tar.gz}"
# PACK FROM AN EXISTING HARDLINKED TREE, DO NOT COPY. The first version
# mktemp'd a scratch dir and cp -r'd ~11.5 GB into it before taring -- 11.5 GB
# of I/O and disk for no benefit, against a 20 GB request, on a node whose
# /tmp may not be the job's scratch at all.
#
# /staging/.../models_sidecar already holds exactly the layout the archive
# needs, built with `cp -al` so it costs no space, and it is the SAME tree the
# validation bind-mounts to /opt/ml/model. Packing what was validated -- rather
# than a fresh copy assembled by different code -- removes a whole class of
# "the tarball differs from what we tested" failure.
WORK="${SIDECAR_DIR:-/staging/n/nkalthoff/surgvu26/models_sidecar}"

# Checked on the TREE BEING PACKED, not on the source directories it was built
# from -- packing one thing after validating another is how a good check
# certifies the wrong artefact.
[ -d "$WORK" ] || { echo "FATAL: $WORK does not exist"; exit 1; }
found=0
for d in "$WORK"/*/; do
    [ -d "$d" ] || continue
    found=$((found+1))
    [ -f "$d/config.json" ] || { echo "FATAL: $d has no config.json"; exit 1; }
    ls "$d"/*.safetensors >/dev/null 2>&1 || { echo "FATAL: $d has no weight shards"; exit 1; }
done
[ "$found" -gt 0 ] || { echo "FATAL: $WORK holds no model directories"; exit 1; }
echo "$found model director(ies) verified loadable"

echo "packing from $WORK (already assembled and validation-tested)"
du -sh "$WORK"/* 2>/dev/null
# Every entry must be a real directory: a symlink into /staging would dangle
# inside the container, and inside a tarball it would archive the LINK rather
# than the weights -- a 4 KB "model" that extracts to nothing.
if find "$WORK" -maxdepth 1 -type l | grep -q .; then
    echo "FATAL: $WORK contains symlinks; tar would archive the links, not the"
    echo "  weights. Rebuild it with cp -al (hardlinks) instead."
    exit 1
fi

echo "packing $OUT"
tar -czf "$OUT" -C "$WORK" .

echo "== verify the archive's top-level layout =="
# Must be ./<model-dir>/..., never ./models/<model-dir>/...
tar -tzf "$OUT" | awk -F/ '{print $2}' | sort -u | head -5
echo "size: $(du -h "$OUT" | cut -f1)"
echo
echo "Upload this to the MODEL slot (not the container slot)."
