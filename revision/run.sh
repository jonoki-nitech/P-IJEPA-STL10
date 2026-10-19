#!/usr/bin/env bash
# =============================================================================
# revision/run.sh
#
#   Two experiments, end to end, for the hand-off run:
#     Step 0: uncertainty (sigma) calibration           -> images/calibration/
#     Step 1: 10 selection strategies (7 baselines + AME + the 2 lookahead
#             variants) x block sizes b=1..4 in ONE script
#             (patch_selection_reconstruction_eval.py handles selection +
#              MAE sweep; all strategies share one aligned t axis)
#
#   10 strategies generated per block size:
#     sigma_greedy    : sigma-based selection, random initial block
#     sigma_center    : sigma-based selection, center initial block
#     random_greedy   : random selection, random initial block
#     random_center   : random selection, center initial block
#     center_outward  : fixed, center-outward order
#     raster          : fixed, raster order (top-left initial)
#     raster_center   : fixed, raster order (center initial block)
#     ame             : Attention-Map Entropy baseline (Pardyl et al.,
#                       IJCAI 2023) on the MAE decoder attention;
#                       random initial block (seeded), duplicate-free
#                       truncation
#     sigma_lookahead : 1-step lookahead sigma minimisation, random initial
#                       block (simulate every top-M candidate with its
#                       mu-predicted features, pick the one that minimises
#                       the total sigma left unobserved; M=16 default)
#     sigma_lookahead_center : same with the center initial block
#
#   While selecting, the sigma-based and lookahead variants record, per image
#   and step, the total sigma still unobserved; the lookahead variants also
#   record the sigma rank of each chosen block. All of it lands in
#   images/rec_images_tmax32/b{b}/sigma_sum_traj.pt.
#
#   Each variant is truncated at its observable budget (no fully-unobserved
#   b x b window remains -> the image stops observing, even if steps are
#   left); the MAE evaluation aligns all strategies at
#   aligned_max_t = min(144/b^2 - 1, min over (variant, image) budgets),
#   i.e. plotting stops where duplicated observations would begin.
#   The random-initial variants (sigma_greedy / random_greedy / ame /
#   sigma_lookahead) share ONE initial block per image (paired comparison;
#   different images keep independent draws).
#   Outputs go to images/rec_images_tmax32/b{b}/ (rec_error_vs_t.png/csv/pt,
#   sigma_sum_traj.pt, gallery/nXXXX.png for the first --gallery_n samples)
#   and the selection JSONs to selected_patch_index/_
#   block_{b}_tmax_32_{variant}.json (the earlier tmax_20 artifacts are left
#   untouched).
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
#     BS="2 4" ...                               # block sizes (default 1 2 3 4)
#     DEVICE=cpu ...                             # pin the device
#     GALLERY_N=0 ...                            # skip gallery PNGs
#     ATTENTION_LAYER=2 ...                      # AME decoder attention block
#     CHUNK=512 ...                              # chunk size (default 1024)
#     M=8 ...                                    # lookahead candidate prescreen
#                                                #  (default 16; ALL_CANDIDATES=1
#                                                #  disables the prescreen,
#                                                #  VERY slow for b=1)
#     SKIP_ZIP=1 ...                             # skip packaging
#     ZIP=result.zip ...                         # zip name
#
#   Runtime on one 24 GB GPU at N=8000: calibration a few minutes, the 8 fast
#   variants ~2.5-4 h, each lookahead variant ~4-5 h (b=1 dominates), so the
#   full run is roughly ~12-14 h including the MAE sweeps.
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

export PYTHONUNBUFFERED=1 # real-time log lines even when piped through tee

PY=${PYTHON:-python}
N=${N:-0}               # number of test images (0 = all 8000)
BS=${BS:-"1 2 3 4"}      # block sizes to evaluate
SEED=${SEED:-0}          # seed for the stochastic strategies
DEVICE=${DEVICE:-cuda:0} # override with e.g. DEVICE=cpu
GALLERY_N=${GALLERY_N:-10}
AME_LAYER=${AME_LAYER:--1} # AME attention layer (-1 = last decoder block)
LOOKAHEAD_M=${M:-16}     # lookahead candidate prescreen size
export CHUNK=${CHUNK:-1024} # chunk size of the batched forward passes

echo "python : $($PY --version 2>&1)"
echo "params : N=$N   bs=$BS   seed=$SEED   device=$DEVICE   gallery_n=$GALLERY_N   ame_attention_layer=$AME_LAYER   lookahead_M=$LOOKAHEAD_M"

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
# Step 1: block sizes b = 1..4 (selection of all 10 strategies + MAE
#         error-vs-t + uncertainty trajectories, one script)
# -----------------------------------------------------------------------------
echo ""
echo "=== Step 1: selection (10 variants) + MAE reconstruction vs t ======"
LOOKAHEAD_FLAGS=""
if [ "${ALL_CANDIDATES:-0}" = "1" ]; then
    LOOKAHEAD_FLAGS="--all-candidates"
fi
$PY patch_selection_reconstruction_eval.py \
    --bs $BS \
    --N "$N" \
    --seed "$SEED" \
    --attention-layer "$AME_LAYER" \
    --M "$LOOKAHEAD_M" \
    $LOOKAHEAD_FLAGS \
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
