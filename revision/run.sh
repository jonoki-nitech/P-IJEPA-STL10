#!/usr/bin/env bash
# =============================================================================
# revision/run.sh
#
#   Two experiments, end to end, for the hand-off run:
#     Step 0: uncertainty (sigma) calibration           -> images/calibration/
#     Step 1: 7 selection strategies x block sizes b=1..4
#             (patch selection until saturation, evaluation aligned at the
#              smallest budget) -> per-variant index JSONs + MAE
#              reconstruction-error vs t
#
#   7 strategies generated per block size (all inside ONE script,
#   patch_selection_reconstruction_eval.py handles selection + MAE sweep):
#     sigma_greedy    : sigma-based selection, random initial block
#     sigma_center    : sigma-based selection, center initial block
#     random_greedy   : random selection, random initial block
#     random_center   : random selection, center initial block
#     center_outward  : fixed, center-outward order
#     raster          : fixed, raster order (top-left initial)
#     raster_center   : fixed, raster order (center initial block)
#
#   Each variant is truncated at its observable budget (no fully-unobserved
#   b x b window remains -> the image stops observing, even if steps are
#   left); the MAE evaluation aligns all strategies at
#   aligned_max_t = min(144/b^2 - 1, min over (variant, image) budgets),
#   i.e. plotting stops where duplicated observations would begin.
#   Outputs go to images/rec_images_tmax32/b{b}/ (rec_error_vs_t.png/csv/pt,
#   gallery/nXXXX.png for the first --gallery_n samples) and the selection
#   JSONs to selected_patch_index/block_{b}_tmax_32_{variant}.json
#   (the earlier tmax_20 artifacts are left untouched).
#
#   Step 0 (calibration) outputs images/calibration/:
#   reliability diagram + metrics CSV + calibration_points.pt (per-point
#   sigma / prediction error + ids, for later re-analysis:
#   python calibration_eval.py --points_pt <path>),
#   context (b=4 at "4:4") patches excluded from the calibration points.
#   NOTE: this script OVERWRITES images/calibration/ and the
#   block_*_tmax_32_*.json files.
#
# Usage:
#   PYTHON=/path/to/python bash revision/run.sh
#   (PY defaults to "python". Scripts auto-fall back to CPU when CUDA is
#    unavailable, so no device flag is needed unless you want to pin one.)
#
#   Overrides:
#     N=100 PYTHON=... bash revision/run.sh      # first 100 test images only
#     DEVICE=cpu ...                             # pin the device
#     GALLERY_N=0 ...                            # skip gallery PNGs
#     SKIP_ZIP=1 ...                             # skip packaging
#     ZIP=result.zip ...                         # zip name
#
#   Requires: torch, torchvision, numpy, matplotlib, einops, tqdm.
#   Models: I-JEPA at
#   models/embdim=512_epoch=300_patchsize=8/training_model.pth and MAE at
#   models/mae/latest.ckpt (both --model_path / --mae_model_path overridable
#   in the patch-selection call below). STL10 data is resolved by
#   data_utils.py: if the server data directory exists it is used directly,
#   otherwise only the required splits are streamed into data/stl10_binary/.
# =============================================================================

set -e
set -u

PY=${PYTHON:-python}
N=${N:-0}               # number of test images (0 = all 8000)
SEED=${SEED:-0}          # seed for the stochastic strategies
DEVICE=${DEVICE:-cuda:0} # override with e.g. DEVICE=cpu
GALLERY_N=${GALLERY_N:-10}

echo "python : $($PY --version 2>&1)"
echo "params : N=$N   seed=$SEED   device=$DEVICE   gallery_n=$GALLERY_N"

# -----------------------------------------------------------------------------
# Step 0: calibration of the sigma predictor
#   default: b=4, one b x b-patch evaluation position per image ("4:4"),
#   context patches excluded from the calibration points.
# -----------------------------------------------------------------------------
echo ""
echo "=== Step 0: calibration ==========================================="
$PY calibration_eval.py \
    --b 4 \
    --positions 4:4 \
    --N "$N" \
    --device "$DEVICE"

# -----------------------------------------------------------------------------
# Step 1: block sizes b = 1..4 (selection + MAE error-vs-t, one script)
# -----------------------------------------------------------------------------
echo ""
echo "=== Step 1: selection (7 variants) + MAE reconstruction vs t ======="
$PY patch_selection_reconstruction_eval.py \
    --bs 1 2 3 4 \
    --N "$N" \
    --seed "$SEED" \
    --gallery_n "$GALLERY_N" \
    --device "$DEVICE"

# -----------------------------------------------------------------------------
# Summary of the produced artefacts
# -----------------------------------------------------------------------------
echo ""
echo "=== Done. Produced artefacts ======================================"
echo "--- calibration -------------------------------------------------"
ls -d images/calibration 2>/dev/null || echo "(missing: images/calibration)"
echo "--- selection index JSONs (block_*_tmax_32_*) --------------------"
ls selected_patch_index/*tmax_32*.json 2>/dev/null || echo "(missing)"
echo "--- MAE error-vs-t results ---------------------------------------"
ls -d images/rec_images_tmax32/b[1-4] 2>/dev/null
for b in 1 2 3 4; do
    ls -d "images/rec_images_tmax32/b$b/gallery" 2>/dev/null || true
done

# -----------------------------------------------------------------------------
# Package the artefacts into a single zip (path overridable with ZIP=<name>)
# -----------------------------------------------------------------------------
if [ "${SKIP_ZIP:-0}" != "1" ]; then
    ZIP=${ZIP:-result.zip}
    echo ""
    echo "=== Pack artefacts into $ZIP ===================================="
    rm -f "$ZIP"
    "$PY" - "$ZIP" <<'PYEOF'
import os, sys, zipfile
zip_name = sys.argv[1]
with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for root in ("images/calibration", "images/rec_images_tmax32"):
        for dirpath, _, files in os.walk(root):
            for f in files:
                zf.write(os.path.join(dirpath, f))
    for f in os.listdir("selected_patch_index"):
        if "tmax_32" in f:
            zf.write(os.path.join("selected_patch_index", f))
print(f"created: {zip_name} ({os.path.getsize(zip_name) / 1e6:.1f} MB)")
PYEOF
fi

exit 0
